//
//  ScreenTextTermExtractor.swift
//  HyperWhisper
//
//  Picks the words on the screen a speech model is likely to mishear (names,
//  brands, identifiers, acronyms) for a model that takes a word list rather
//  than free text: Gemini 3.5 Transcribe's `customVocabulary`. Ordinary
//  dictionary words (menus, buttons, prose) are dropped: the model already
//  knows them, and they would crowd the names out of the 100-word list.
//
//  Runs on the device. The rules are heuristics, not tuned against a
//  benchmark yet.
//

import AppKit
import Foundation

enum ScreenTextTermExtractor {

    /// The most terms handed to the speech engine: Gemini 3.5 Transcribe's
    /// `customVocabulary` takes 100, and the caller puts the user's Custom
    /// Vocabulary ahead of these and caps the total.
    static let maximumTerms = 100

    /// Shorter words collide with ordinary speech ("CRAN" ~ "ran").
    static let minimumLength = 4
    static let maximumLength = 32

    /// Returns at most `limit` terms from `ocrText`, best first.
    ///
    /// - Parameter isDictionaryWord: whether a word, as spelled, is ordinary
    ///   language the speech model already knows. Injected so tests do not
    ///   depend on the system dictionary; the app passes
    ///   `englishDictionaryLookup()`.
    static func terms(
        from ocrText: String,
        limit: Int = maximumTerms,
        isDictionaryWord: (String) -> Bool
    ) -> [String] {
        guard limit > 0, !ocrText.isEmpty else { return [] }

        struct Candidate {
            var spelling: String
            var spellingCount: Int
            var count: Int
            let firstIndex: Int
            var score: Int
        }

        var candidates: [String: Candidate] = [:]
        var spellingCounts: [String: [String: Int]] = [:]
        var dictionaryCache: [String: Bool] = [:]
        func lookUp(_ word: String) -> Bool {
            if let cached = dictionaryCache[word] { return cached }
            let value = isDictionaryWord(word)
            dictionaryCache[word] = value
            return value
        }

        for (index, raw) in tokens(in: ocrText).enumerated() {
            guard let token = normalize(raw) else { continue }
            let lowered = token.lowercased()
            guard !stopwords.contains(lowered) else { continue }
            guard let score = score(token, lowered: lowered, isKnown: lookUp) else { continue }

            spellingCounts[lowered, default: [:]][token, default: 0] += 1
            if var existing = candidates[lowered] {
                existing.count += 1
                existing.score = max(existing.score, score)
                let spellingCount = spellingCounts[lowered]?[token] ?? 0
                if spellingCount > existing.spellingCount {
                    existing.spelling = token
                    existing.spellingCount = spellingCount
                }
                candidates[lowered] = existing
            } else {
                candidates[lowered] = Candidate(
                    spelling: token, spellingCount: 1, count: 1, firstIndex: index, score: score)
            }
        }

        return candidates.values
            // No minimum count: a name shown once ("zooly" in a sidebar) is
            // exactly what the model needs. A plain lowercase word the
            // dictionary does not know can be an OCR misread, so it ranks last
            // and is the first cut at the limit. The list is a hint the model
            // weighs against the audio, not a rewrite rule.
            .sorted {
                if $0.score != $1.score { return $0.score > $1.score }
                if $0.count != $1.count { return $0.count > $1.count }
                return $0.firstIndex < $1.firstIndex
            }
            .prefix(limit)
            .map(\.spelling)
    }

    /// A dictionary check against the system's English spelling dictionary.
    /// Call on the main actor: `NSSpellChecker` is not thread-safe.
    @MainActor
    static func englishDictionaryLookup() -> (String) -> Bool {
        let checker = NSSpellChecker.shared
        return { word in
            let range = checker.checkSpelling(
                of: word, startingAt: 0, language: "en", wrap: false,
                inSpellDocumentWithTag: 0, wordCount: nil)
            return range.location == NSNotFound
        }
    }

    // MARK: - Rules

    private enum Score {
        static let unknownLowercase = 1
        static let properNoun = 2
        static let unknownCapitalized = 3
        static let acronym = 3
        static let mixedCase = 4
    }

    /// nil drops the token. Higher is more likely to be a term worth boosting.
    private static func score(_ token: String, lowered: String, isKnown: (String) -> Bool) -> Int? {
        let letters = token.unicodeScalars.filter { CharacterSet.letters.contains($0) }
        let uppercase = letters.filter { CharacterSet.uppercaseLetters.contains($0) }.count
        let lowercase = letters.count - uppercase
        let first = token.unicodeScalars.first.map { CharacterSet.uppercaseLetters.contains($0) } ?? false

        // HyperWhisper, macOS, getUserId: an inner capital is never ordinary prose.
        let innerCapital = token.unicodeScalars.dropFirst().contains { CharacterSet.uppercaseLetters.contains($0) }
        if innerCapital && lowercase > 0 {
            return Score.mixedCase
        }
        // Every other shape is ordinary language when the dictionary knows its
        // lowercase form, including SETTINGS shouted by a UI.
        if isKnown(lowered) { return nil }
        if lowercase == 0 && uppercase >= minimumLength && uppercase <= 8 {
            return Score.acronym  // NVIDIA, JSON
        }
        if first && lowercase > 0 {
            // The dictionary also holds proper nouns (Kubernetes, Weiss) and
            // knows only their capitalized form. They are still names a speech
            // model can mishear, so they stay, ranked below unknown names.
            return isKnown(token) ? Score.properNoun : Score.unknownCapitalized
        }
        if uppercase == 0 {
            return Score.unknownLowercase
        }
        return nil
    }

    private static func tokens(in text: String) -> [Substring] {
        text.split { character in
            !(character.isLetter || character.isNumber || character == "-" || character == "'"
                || character == "’" || character == "_")
        }
    }

    private static let edgeCharacters = CharacterSet(charactersIn: "-'’_")

    /// Trims edge punctuation and a possessive, and drops what is not a word.
    private static func normalize(_ raw: Substring) -> String? {
        var token = String(raw).trimmingCharacters(in: edgeCharacters)
        for suffix in ["'s", "’s"] where token.hasSuffix(suffix) {
            token = String(token.dropLast(suffix.count))
        }
        // "zooly2", "project3": a short number on the end names a copy of the
        // same thing, so it counts toward the name it is attached to.
        let trailingDigits = token.reversed().prefix { $0.isNumber }.count
        if trailingDigits > 0, trailingDigits <= 2, token.count - trailingDigits >= minimumLength {
            token = String(token.dropLast(trailingDigits))
        }
        guard token.count >= minimumLength, token.count <= maximumLength else { return nil }

        var letterCount = 0
        var digitCount = 0
        for character in token {
            if character.isLetter { letterCount += 1 } else if character.isNumber { digitCount += 1 }
        }
        // Version strings, dates, hashes and IDs are not words anyone dictates.
        guard letterCount >= minimumLength - 1, digitCount <= 2 else { return nil }
        // Identifiers with underscores are read as separate words; skip them.
        guard !token.contains("_") else { return nil }
        return token
    }

    /// Function words, kept out even when OCR splits them oddly.
    private static let stopwords: Set<String> = [
        "about", "after", "also", "been", "before", "being", "could", "does", "done", "each",
        "from", "have", "here", "into", "just", "like", "more", "most", "much", "only", "over",
        "same", "should", "some", "such", "than", "that", "their", "them", "then", "there",
        "these", "they", "this", "those", "very", "were", "what", "when", "where", "which",
        "while", "will", "with", "would", "your",
    ]
}
