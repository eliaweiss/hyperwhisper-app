//
//  ScreenTextTermExtractorTests.swift
//  hyperwhisperTests
//
//  The extractor picks names, brands, identifiers and acronyms off the screen
//  and drops ordinary words. A fixed word list stands in for the system
//  dictionary so these tests do not depend on the Mac's spelling settings.
//

import Foundation
import Testing
@testable import HyperWhisper

struct ScreenTextTermExtractorTests {
    private static let dictionary: Set<String> = [
        "file", "edit", "view", "window", "help", "settings", "inbox", "message", "send",
        "monday", "Monday", "meeting", "notes", "project", "deploy", "cluster", "today", "review",
    ]

    private func terms(_ text: String, limit: Int = ScreenTextTermExtractor.maximumTerms) -> [String] {
        ScreenTextTermExtractor.terms(from: text, limit: limit) { Self.dictionary.contains($0) }
    }

    @Test func keepsNamesBrandsAndIdentifiersAndDropsDictionaryWords() {
        let result = terms("File Edit View Window Help\nMeeting notes Monday: deploy Kubernetes cluster with HyperWhisper and NVIDIA")
        #expect(result.contains("Kubernetes"))
        #expect(result.contains("HyperWhisper"))
        #expect(result.contains("NVIDIA"))
        for ordinary in ["File", "Edit", "View", "Window", "Help", "Meeting", "Monday", "deploy", "cluster"] {
            #expect(!result.contains(ordinary), "kept the dictionary word \(ordinary)")
        }
    }

    @Test func ranksMixedCaseFirst() {
        let result = terms("Weiss Weiss Weiss macOS")
        #expect(result.first == "macOS")
        #expect(result.contains("Weiss"))
    }

    @Test func keepsProperNounsTheDictionaryKnowsBelowUnknownNames() {
        let result = ScreenTextTermExtractor.terms(from: "Weiss Zooly") { $0 == "Weiss" }
        #expect(result == ["Zooly", "Weiss"])
    }

    @Test func dropsShortNumericAndUnderscoreTokens() {
        let result = terms("API v1.2.3 2026-11-04 a1b2c3d4e5 user_id_field Elia")
        #expect(result == ["Elia"], "got \(result)")
    }

    /// A name shown once is kept (the "zooly" sidebar entry), but an unknown
    /// lowercase word, which can be an OCR misread, ranks below real names.
    @Test func keepsAnUnknownLowercaseWordSeenOnceButRanksItLast() {
        #expect(terms("review zooly today") == ["zooly"])
        #expect(terms("zooly Kubernetes") == ["Kubernetes", "zooly"])
    }

    @Test func mergesCaseVariantsAndStripsPossessives() {
        let result = terms("Zooly's dashboard, Zooly, ZOOLY")
        #expect(result.filter { $0.lowercased() == "zooly" } == ["Zooly"])
    }

    /// A workspace sidebar listing "zooly2" and "zooly" names one thing twice.
    @Test func countsANumberedCopyTowardItsName() {
        #expect(terms("hyperwhisper-app zooly2 zooly zdocs VoiceMem").contains("zooly"))
    }

    @Test func honoursTheLimit() {
        let text = (1...20).map { "Termname\(String(UnicodeScalar(UInt8(64 + $0))))x" }.joined(separator: " ")
        #expect(terms(text, limit: 5).count == 5)
    }

    @Test func emptyInputGivesNoTerms() {
        #expect(terms("").isEmpty)
        #expect(terms("Kubernetes", limit: 0).isEmpty)
    }
}
