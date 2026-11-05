//
//  ScreenTextTranscriptionContext.swift
//  HyperWhisper
//
//  Hands the screen text captured at recording start to a transcription model
//  that can use it, so it spells on-screen names correctly without a
//  post-processing LLM:
//  - Gemini's generateContent path (`GeminiTranscriptionProvider`) takes a
//    free-form prompt, so the whole capture goes in as context.
//  - Gemini 3.5 Transcribe through Vercel AI Gateway
//    (`VercelAIGatewayProvider`) takes only a word list (`customVocabulary`),
//    so `ScreenTextTermExtractor` picks the names off the screen first.
//
//  A task-local, not a property on the provider: Gemini runs a dictation and a
//  file job at the same time on one instance (#1425), and a property would let
//  one pass read the other's screen.
//

import Foundation

enum ScreenTextTranscriptionContext {

    /// The screen text for the transcription running in this task, or nil.
    /// Set by `TranscriptionPipeline` only when the mode has Screen Text on.
    @TaskLocal static var screenText: String?

    /// The prompt paragraph a model receives with the screen text, or nil when
    /// there is none. The text is context only: the model is told never to
    /// transcribe it, because a model that repeats on-screen text into the
    /// transcript is worse than one that misspells a name.
    static func promptSection(for screenText: String?) -> String? {
        guard let text = screenText?.trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty
        else { return nil }
        // The capture is arbitrary on-screen text; it must not be able to close
        // the block it sits in.
        let contained = text
            .replacingOccurrences(of: "<screen_text>", with: "")
            .replacingOccurrences(of: "</screen_text>", with: "")
        return """
        The text between the screen_text tags was visible on the speaker's screen when they \
        started talking. It is reference material only. Use it to spell names, products, \
        people and technical terms correctly when the speaker says them. Never transcribe it, \
        never quote it, and never add a word from it that the speaker did not say.
        <screen_text>
        \(contained)
        </screen_text>
        """
    }

    /// The mode's own custom prompt followed by the screen-text section.
    static func prompt(customPrompt: String?, screenText: String?) -> String? {
        let parts = [customPrompt, promptSection(for: screenText)]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return parts.isEmpty ? nil : parts.joined(separator: "\n\n")
    }
}
