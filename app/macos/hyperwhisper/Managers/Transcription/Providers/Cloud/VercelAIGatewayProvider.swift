//
//  VercelAIGatewayProvider.swift
//  hyperwhisper
//
//  Batch transcription through Vercel AI Gateway with the user's own
//  AI Gateway API key. One key reaches every transcription model the gateway
//  lists (OpenAI, ElevenLabs, Google, Microsoft, Mistral, Fish Audio, ...);
//  the model id is the gateway's `creator/model` form, e.g.
//  `openai/gpt-4o-transcribe`.
//
//  WIRE FORMAT (https://vercel.com/docs/ai-gateway/modalities/speech-to-text):
//    POST https://ai-gateway.vercel.sh/v4/ai/transcription-model
//    Authorization: Bearer <AI_GATEWAY_API_KEY>
//    ai-gateway-protocol-version: 0.0.1
//    ai-transcription-model-specification-version: 4
//    ai-model-id: <creator/model>
//    { "audio": <base64>, "mediaType": <mime>, "providerOptions": {...} }
//  → { "text": ..., "segments": [...], "language": ..., "durationInSeconds": ... }
//
//  NOTE: unlike the other direct vendors, this request is built in Swift, not
//  in the Rust shared core. It is a local, macOS-only addition.
//

import Foundation

final class VercelAIGatewayProvider: TranscriptionProvider {
    /// A stateless HTTP call: nothing per call is kept on the instance.
    var supportsConcurrentPasses: Bool { true }

    enum Constants {
        static let baseURL = URL(string: "https://ai-gateway.vercel.sh")!
        static let transcriptionPath = "/v4/ai/transcription-model"
        /// OpenAI-compatible chat endpoint, for general audio models that
        /// transcribe from a prompt (`isPromptedModel`).
        static let chatCompletionsPath = "/v1/chat/completions"
        static let creditsPath = "/v1/credits"
        static let protocolVersion = "0.0.1"
        static let specificationVersion = "4"
        static let defaultModelID = "openai/gpt-4o-transcribe"
        /// Raw audio cap. The body is base64 JSON, which inflates it by ~33%.
        static let maxUploadBytes: Int64 = 25 * 1024 * 1024
        static let maxPromptTerms = 100
    }

    private var apiKey = ""
    private lazy var session: URLSession = {
        let configuration = URLSessionConfiguration.credentialBearing
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 600
        configuration.waitsForConnectivity = false
        return URLSession(configuration: configuration)
    }()

    var isAvailable: Bool { !apiKey.isEmpty }
    var name: String { "Vercel AI Gateway" }

    func configure(apiKey: String) {
        self.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        AppLogger.network.debug("Vercel AI Gateway API key configured · nonEmpty=\(!self.apiKey.isEmpty, privacy: .public)")
    }

    // MARK: - Transcription

    func transcribe(audioURL: URL, language: String?, mode: Mode?, vocabulary: [Vocabulary]) async throws -> String {
        guard !apiKey.isEmpty else {
            throw TranscriptionError.apiKeyMissing(provider: name)
        }
        guard NetworkStatus.shared.isOnline else {
            throw TranscriptionError.transientNetwork(details: nil)
        }
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw TranscriptionError.audioFileNotFound
        }

        let selected = mode?.cloudTranscriptionModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let model = selected.isEmpty ? Constants.defaultModelID : selected

        let mime = AudioMimeTypeResolver.infer(for: audioURL, fallback: "audio/wav")

        // A general audio model (Gemini Flash / Flash Lite) transcribes from a prompt,
        // which can carry the whole screen text and every vocabulary word.
        // Gemini reads WAV, MP3, AAC, FLAC, OGG and AIFF, not an MP4 container.
        if Self.isPromptedModel(model) {
            if mime == "audio/mp4" {
                let wavURL = CloudAudioFormatRecovery.makeTemporaryWAVURL()
                defer { CloudAudioFormatRecovery.deleteFile(at: wavURL) }
                try await CloudAudioFormatRecovery.reencodeToWAV(source: audioURL, destination: wavURL)
                return try await sendPrompted(audioURL: wavURL, mime: CloudAudioFormatRecovery.wavContentType,
                                              model: model, language: language, vocabulary: vocabulary,
                                              instruction: mode?.geminiCustomPrompt)
            }
            return try await sendPrompted(audioURL: audioURL, mime: mime, model: model,
                                          language: language, vocabulary: vocabulary,
                                          instruction: mode?.geminiCustomPrompt)
        }

        // Microsoft MAI models reject MP4/M4A containers; send them a WAV.
        if model.hasPrefix("microsoft/"), mime == "audio/mp4" {
            let wavURL = CloudAudioFormatRecovery.makeTemporaryWAVURL()
            defer { CloudAudioFormatRecovery.deleteFile(at: wavURL) }
            try await CloudAudioFormatRecovery.reencodeToWAV(source: audioURL, destination: wavURL)
            return try await send(audioURL: wavURL, mime: CloudAudioFormatRecovery.wavContentType,
                                  model: model, language: language, vocabulary: vocabulary)
        }
        return try await send(audioURL: audioURL, mime: mime, model: model,
                              language: language, vocabulary: vocabulary)
    }

    private func send(
        audioURL: URL,
        mime: String,
        model: String,
        language: String?,
        vocabulary: [Vocabulary]
    ) async throws -> String {
        let fileBytes = try audioURL.fileSize()
        guard fileBytes <= Constants.maxUploadBytes else {
            throw TranscriptionError.audioFileTooLarge(
                fileSize: fileBytes,
                limit: Constants.maxUploadBytes,
                providerName: name
            )
        }

        let audio = try Data(contentsOf: audioURL)
        // Custom Vocabulary first (the user chose it), then the names picked
        // off the screen when the mode has Screen Text on.
        var terms = RustCoreMapping.boostVocabularyTerms(from: vocabulary)
        let screenTerms = await Self.screenTerms()
        if !screenTerms.isEmpty {
            var seen = Set(terms.map { $0.lowercased() })
            terms += screenTerms.filter { seen.insert($0.lowercased()).inserted }
            AppLogger.network.info("Vercel AI Gateway vocabulary includes screen terms · screen=\(screenTerms.count, privacy: .public) · total=\(min(terms.count, Constants.maxPromptTerms), privacy: .public)")
        }
        let request = try Self.buildRequest(
            apiKey: apiKey,
            model: model,
            audio: audio,
            mediaType: mime,
            language: Self.normalizedLanguage(language),
            vocabulary: Array(terms.prefix(Constants.maxPromptTerms))
        )

        AppLogger.network.info("Vercel AI Gateway transcription started · model=\(model, privacy: .public) · bytes=\(fileBytes, privacy: .public) · language=\(language ?? "auto", privacy: .public)")

        let (data, response) = try await performWithRetry(request)
        try Task.checkCancellation()

        let text = try Self.parseResponse(data: data, response: response, providerName: name)
        AppLogger.network.info("Vercel AI Gateway transcription completed · chars=\(text.count, privacy: .public)")
        return text
    }

    /// Transcription through a general audio model: the recording plus a prompt,
    /// sent to the gateway's Chat Completions endpoint. Unlike the transcription
    /// endpoint, the prompt has room for the whole screen text and every
    /// vocabulary word.
    private func sendPrompted(
        audioURL: URL,
        mime: String,
        model: String,
        language: String?,
        vocabulary: [Vocabulary],
        instruction: String?
    ) async throws -> String {
        let fileBytes = try audioURL.fileSize()
        guard fileBytes <= Constants.maxUploadBytes else {
            throw TranscriptionError.audioFileTooLarge(
                fileSize: fileBytes,
                limit: Constants.maxUploadBytes,
                providerName: name
            )
        }

        let audio = try Data(contentsOf: audioURL)
        let terms = RustCoreMapping.boostVocabularyTerms(from: vocabulary)
        let screenText = ScreenTextTranscriptionContext.screenText
        let prompt = Self.transcriptionPrompt(
            instruction: instruction,
            language: Self.normalizedLanguage(language),
            vocabulary: terms,
            screenText: screenText
        )
        let request = try Self.buildPromptedRequest(
            apiKey: apiKey, model: model, audio: audio, mediaType: mime, prompt: prompt)

        let usesCustomInstruction = !(instruction?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        AppLogger.network.info("Vercel AI Gateway prompted transcription started · model=\(model, privacy: .public) · bytes=\(fileBytes, privacy: .public) · vocabulary=\(terms.count, privacy: .public) · screenChars=\(screenText?.count ?? 0, privacy: .public) · customPrompt=\(usesCustomInstruction, privacy: .public)")

        let (data, response) = try await performWithRetry(request)
        try Task.checkCancellation()

        let text = try Self.parsePromptedResponse(data: data, response: response, providerName: name)
        AppLogger.network.info("Vercel AI Gateway prompted transcription completed · chars=\(text.count, privacy: .public)")
        return text
    }

    /// Retries only transport failures and 429/5xx, with a short backoff.
    private func performWithRetry(_ request: URLRequest) async throws -> (Data, URLResponse) {
        let maxAttempts = 3
        var attempt = 0
        while true {
            attempt += 1
            do {
                let (data, response) = try await session.data(for: request)
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                let retryable = status == 429 || (500...599).contains(status)
                if retryable && attempt < maxAttempts {
                    AppLogger.network.warning("Vercel AI Gateway retrying · status=\(status, privacy: .public) · attempt=\(attempt, privacy: .public)")
                    try await Task.sleep(nanoseconds: UInt64(attempt) * 750_000_000)
                    continue
                }
                return (data, response)
            } catch let error as URLError where error.code != .cancelled && attempt < maxAttempts {
                AppLogger.network.warning("Vercel AI Gateway retrying · urlError=\(error.code.rawValue, privacy: .public) · attempt=\(attempt, privacy: .public)")
                try await Task.sleep(nanoseconds: UInt64(attempt) * 750_000_000)
            } catch let error as URLError {
                if error.code == .cancelled { throw CancellationError() }
                if error.code == .timedOut { throw TranscriptionError.timeout(operation: "Vercel AI Gateway transcription") }
                throw TranscriptionError.transientNetwork(details: error.localizedDescription)
            }
        }
    }

    // MARK: - Request / response (static so they can be tested in isolation)

    static func buildRequest(
        apiKey: String,
        model: String,
        audio: Data,
        mediaType: String,
        language: String?,
        vocabulary: [String]
    ) throws -> URLRequest {
        var request = URLRequest(url: Constants.baseURL.appendingPathComponent(Constants.transcriptionPath))
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue(Constants.protocolVersion, forHTTPHeaderField: "ai-gateway-protocol-version")
        request.setValue(Constants.specificationVersion, forHTTPHeaderField: "ai-transcription-model-specification-version")
        request.setValue(model, forHTTPHeaderField: "ai-model-id")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var body: [String: Any] = [
            "audio": audio.base64EncodedString(),
            "mediaType": mediaType,
        ]
        let options = providerOptions(model: model, language: language, vocabulary: vocabulary)
        if !options.isEmpty {
            body["providerOptions"] = options
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return request
    }

    /// Language and vocabulary hints, in each upstream's own option namespace.
    /// Unknown creators get no options: the gateway reports unsupported ones
    /// as warnings, but sending none is the safe default.
    static func providerOptions(model: String, language: String?, vocabulary: [String]) -> [String: Any] {
        let creator = model.split(separator: "/").first.map(String.init) ?? ""
        var options: [String: Any] = [:]
        switch creator {
        case "openai":
            if let language { options["language"] = language }
            if !vocabulary.isEmpty, !model.hasSuffix("whisper-1") || vocabulary.count <= 50 {
                options["prompt"] = "Vocabulary: " + vocabulary.joined(separator: ", ")
            }
            return options.isEmpty ? [:] : ["openai": options]
        case "microsoft":
            if !vocabulary.isEmpty { options["phraseList"] = ["phrases": vocabulary] }
            if let language { options["locales"] = [language] }
            return options.isEmpty ? [:] : ["azure": options]
        case "elevenlabs":
            if let language { options["languageCode"] = language }
            return options.isEmpty ? [:] : ["elevenlabs": options]
        case "google":
            // Gemini 3.5 Transcribe: Google's `AudioTranscriptionConfig`
            // (`language_codes`, `custom_vocabulary`) through the AI SDK names.
            if let language { options["languageCodes"] = [language] }
            if !vocabulary.isEmpty { options["customVocabulary"] = vocabulary }
            return options.isEmpty ? [:] : ["google": options]
        default:
            return [:]
        }
    }

    // MARK: - Prompted transcription (general audio models)

    /// A general model that takes audio plus a prompt, rather than a dedicated
    /// transcription model: Gemini Flash / Pro. `…-transcribe` ids stay on the
    /// transcription endpoint.
    static func isPromptedModel(_ model: String) -> Bool {
        let id = model.lowercased()
        return id.hasPrefix("google/gemini-") && !id.contains("transcribe") && !id.contains("tts")
    }

    /// The instruction sent with the recording when the mode has none of its
    /// own. Kept plain and strict: a chat model that answers, summarizes or
    /// repeats on-screen text is worse than a transcription model that
    /// misspells a name. The mode editor shows it as the starting point.
    static let defaultTranscriptionInstruction = """
        Transcribe the speech in this recording. Output only the words that were spoken, \
        exactly as said, with normal punctuation and capitalization. Do not answer, summarize, \
        translate or add anything: no introduction, labels, quotes, notes or timestamps. \
        If nothing is said, output nothing.
        """

    /// The full prompt: the mode's instruction (or the default), then the
    /// language, the vocabulary and the screen text, which are always added.
    static func transcriptionPrompt(
        instruction: String? = nil,
        language: String?,
        vocabulary: [String],
        screenText: String?
    ) -> String {
        let custom = instruction?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var parts = [custom.isEmpty ? defaultTranscriptionInstruction : custom]
        if let language {
            parts.append("The speech is in language \(language).")
        }
        if !vocabulary.isEmpty {
            parts.append(
                "These words may come up. When one is said, spell it exactly like this: "
                    + vocabulary.joined(separator: ", ") + ".")
        }
        if let screenSection = ScreenTextTranscriptionContext.promptSection(for: screenText) {
            parts.append(screenSection)
        }
        return parts.joined(separator: "\n\n")
    }

    /// Chat Completions with the audio as a `file` part
    /// (https://vercel.com/docs/ai-gateway/inputs-and-tools/audio-input).
    static func buildPromptedRequest(
        apiKey: String,
        model: String,
        audio: Data,
        mediaType: String,
        prompt: String
    ) throws -> URLRequest {
        var request = URLRequest(url: Constants.baseURL.appendingPathComponent(Constants.chatCompletionsPath))
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let fileExtension = mediaType.split(separator: "/").last.map(String.init) ?? "wav"
        let body: [String: Any] = [
            "model": model,
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "text", "text": prompt],
                    ["type": "file", "file": [
                        "filename": "recording.\(fileExtension == "mpeg" ? "mp3" : fileExtension)",
                        "file_data": "data:\(mediaType);base64,\(audio.base64EncodedString())",
                    ]],
                ],
            ]],
            "temperature": 0,
            // Gemini 3 Flash thinks before answering; `low` is its fastest
            // level the gateway passes through (it maps other values to high).
            "reasoning_effort": "low",
            "stream": false,
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return request
    }

    static func parsePromptedResponse(data: Data, response: URLResponse, providerName: String) throws -> String {
        let json = try checkStatus(data: data, response: response, providerName: providerName)
        let message = ((json?["choices"] as? [[String: Any]])?.first?["message"]) as? [String: Any]
        let text: String?
        if let content = message?["content"] as? String {
            text = content
        } else if let parts = message?["content"] as? [[String: Any]] {
            text = parts.compactMap { $0["text"] as? String }.joined()
        } else {
            text = nil
        }
        guard let text else {
            throw TranscriptionError.invalidResponse(details: "Vercel AI Gateway chat response has no text")
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { throw TranscriptionError.noSpeechDetected }
        return trimmed
    }

    /// Names picked off the screen for this call: empty unless the mode has
    /// Screen Text on and the capture worked (`ScreenTextTranscriptionContext`).
    /// The picker uses the system spelling dictionary, which is main-actor only.
    static func screenTerms() async -> [String] {
        guard let screenText = ScreenTextTranscriptionContext.screenText, !screenText.isEmpty else {
            return []
        }
        return await MainActor.run {
            ScreenTextTermExtractor.terms(
                from: screenText,
                isDictionaryWord: ScreenTextTermExtractor.englishDictionaryLookup()
            )
        }
    }

    static func normalizedLanguage(_ language: String?) -> String? {
        guard let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !trimmed.isEmpty, trimmed != "auto" else { return nil }
        return trimmed
    }

    static func parseResponse(data: Data, response: URLResponse, providerName: String) throws -> String {
        let json = try checkStatus(data: data, response: response, providerName: providerName)

        guard let text = json?["text"] as? String else {
            throw TranscriptionError.invalidResponse(details: "Vercel AI Gateway response has no text")
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { throw TranscriptionError.noSpeechDetected }
        return trimmed
    }

    /// Maps a non-2xx reply to a `TranscriptionError`; returns the JSON body otherwise.
    static func checkStatus(data: Data, response: URLResponse, providerName: String) throws -> [String: Any]? {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]

        guard (200...299).contains(status) else {
            let message = ((json?["error"] as? [String: Any])?["message"] as? String)
                ?? String(data: data.prefix(500), encoding: .utf8)
                ?? "HTTP \(status)"
            switch status {
            case 401, 403:
                throw TranscriptionError.unauthorized(provider: providerName, statusCode: status)
            case 402:
                throw TranscriptionError.quotaExceeded(provider: providerName, message: message)
            case 429:
                throw TranscriptionError.rateLimited(retryAfter: nil)
            default:
                throw TranscriptionError.serverError(statusCode: status, message: message)
            }
        }
        return json
    }

    // MARK: - Health

    /// Content-free key check: `GET /v1/credits` answers 200 for a valid key.
    static func healthCheck(apiKey: String) async -> ProviderHealth {
        var request = URLRequest(url: Constants.baseURL.appendingPathComponent(Constants.creditsPath))
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 15
        do {
            let (_, response) = try await URLSession(configuration: .ephemeral).data(for: request)
            switch (response as? HTTPURLResponse)?.statusCode ?? 0 {
            case 200...299: return .healthy
            case 401, 403: return .unauthorized
            default: return .unreachable
            }
        } catch is CancellationError {
            return .unknown
        } catch {
            return .unreachable
        }
    }
}
