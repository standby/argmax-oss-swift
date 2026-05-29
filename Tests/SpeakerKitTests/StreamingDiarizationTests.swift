//  StreamingDiarizationTests.swift
//  SpeakerKitTests
//
//  Copyright © 2026 Stephen Chu.

import XCTest
import WhisperKit
@testable import SpeakerKit

/// Unit coverage for the streaming audio source behind `diarize(audioFile:)`.
/// No model download or GPU needed — just verifies the file-backed source
/// returns the same samples as decoding the whole file, which is what
/// guarantees `diarize(audioFile:)` produces the same result as
/// `diarize(audioArray:)`.
final class StreamingDiarizationSourceTests: XCTestCase {

    private func fixtureURL() throws -> URL {
        guard let url = Bundle.module.url(forResource: "VADAudio", withExtension: "wav") else {
            throw XCTSkip("VADAudio.wav not found in test bundle")
        }
        return url
    }

    func testFileSourceMatchesDecodedArray() throws {
        let url = try fixtureURL()
        let reference = AudioProcessor.convertBufferToArray(
            buffer: try AudioProcessor.loadAudio(fromPath: url.path))
        let source = try FileAudioChunkSource(url: url, sampleRate: 16_000)

        XCTAssertLessThanOrEqual(
            abs(source.sampleCount - reference.count), 2,
            "Streaming source length should match the decoded array")

        // Read in odd-sized windows to exercise window boundaries, concatenate.
        var streamed: [Float] = []
        let window = 1601
        var start = 0
        while start < source.sampleCount {
            let chunk = source.samples(start: start, count: window)
            if chunk.isEmpty { break }
            streamed.append(contentsOf: chunk)
            start += chunk.count
        }

        let n = min(streamed.count, reference.count)
        XCTAssertGreaterThan(n, 16_000, "Should read a meaningful amount of audio")
        var maxDiff: Float = 0
        for i in 0..<n { maxDiff = max(maxDiff, abs(streamed[i] - reference[i])) }
        XCTAssertLessThan(maxDiff, 1e-4,
            "Windowed file reads must match the fully-decoded samples")
    }

    func testEmptyAndOutOfRangeReads() throws {
        let source = try FileAudioChunkSource(url: try fixtureURL(), sampleRate: 16_000)
        XCTAssertTrue(source.samples(start: source.sampleCount, count: 100).isEmpty)
        XCTAssertTrue(source.samples(start: 0, count: 0).isEmpty)
        XCTAssertEqual(source.samples(start: 0, count: 10).count, 10)
    }
}

/// End-to-end check that diarizing straight from a file gives the same speakers
/// and turns as diarizing the decoded array. Downloads the Pyannote models.
final class E2EStreamingDiarizationTests: XCTestCase {

    func testDiarizeFromFileMatchesArray() async throws {
        guard let url = Bundle.module.url(forResource: "VADAudio", withExtension: "wav") else {
            throw XCTSkip("VADAudio.wav not found in test bundle")
        }
        let speakerKit = try await SpeakerKit()

        let audioArray = AudioProcessor.convertBufferToArray(
            buffer: try AudioProcessor.loadAudio(fromPath: url.path))
        let arrayResult = try await speakerKit.diarize(audioArray: audioArray)
        let fileResult = try await speakerKit.diarize(audioFile: url)

        XCTAssertEqual(fileResult.speakerCount, arrayResult.speakerCount,
                       "File path should find the same number of speakers")
        XCTAssertEqual(fileResult.segments.count, arrayResult.segments.count,
                       "File path should produce the same number of segments")
        for (a, b) in zip(arrayResult.segments, fileResult.segments) {
            XCTAssertEqual(a.startTime, b.startTime, accuracy: 0.05)
            XCTAssertEqual(a.endTime, b.endTime, accuracy: 0.05)
        }
    }
}
