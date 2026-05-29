//  AudioChunkSource.swift
//  SpeakerKit
//
//  Copyright © 2026 Stephen Chu.

import AVFoundation
import Foundation

/// Supplies 16 kHz mono PCM windows to the diarization segmenter on demand.
///
/// The segmenter processes audio in 30-second windows across several
/// concurrent workers. Rather than handing it the whole recording as one
/// `[Float]` (≈768 MB for a 200-minute meeting), a source lets the segmenter
/// pull just the window it needs and discard it after inference — so the full
/// PCM need never be resident.
///
/// `samples(start:count:)` MUST be safe to call concurrently from multiple
/// tasks.
public protocol AudioChunkSource: Sendable {
    /// Total number of samples available, at 16 kHz mono.
    var sampleCount: Int { get }

    /// Return up to `count` samples starting at sample index `start`.
    /// Returns fewer samples only when the range runs past `sampleCount`,
    /// and an empty array when `start` is out of range.
    func samples(start: Int, count: Int) -> [Float]
}

/// In-memory source backing the existing `diarize(audioArray:)` path. Slicing
/// a read-only `[Float]` is safe to do concurrently.
public struct ArrayAudioChunkSource: AudioChunkSource {
    public let audio: [Float]
    public init(_ audio: [Float]) { self.audio = audio }

    public var sampleCount: Int { audio.count }

    public func samples(start: Int, count: Int) -> [Float] {
        guard start >= 0, start < audio.count, count > 0 else { return [] }
        let end = Swift.min(start + count, audio.count)
        return Array(audio[start..<end])
    }
}

/// A window of another source, used when a seek-clip covers only part of the
/// recording (the default whole-file clip passes the base source through
/// directly, so this is only hit when `clipTimestamps` are supplied).
public struct OffsetAudioChunkSource: AudioChunkSource {
    public let base: any AudioChunkSource
    public let offset: Int
    public let sampleCount: Int

    public init(base: any AudioChunkSource, offset: Int, count: Int) {
        self.base = base
        self.offset = offset
        self.sampleCount = count
    }

    public func samples(start: Int, count: Int) -> [Float] {
        guard start >= 0, start < sampleCount, count > 0 else { return [] }
        let clamped = Swift.min(count, sampleCount - start)
        return base.samples(start: offset + start, count: clamped)
    }
}

/// File-backed source: reads each window from disk so the full decoded PCM is
/// never held in memory. Each `samples(_:)` call opens its own `AVAudioFile`,
/// which makes concurrent reads from the segmenter's worker pool safe.
public struct FileAudioChunkSource: AudioChunkSource {
    public let url: URL
    public let sampleCount: Int
    private let targetSampleRate: Double
    private let sourceSampleRate: Double

    /// - Parameters:
    ///   - url: An audio file readable by `AVAudioFile`.
    ///   - sampleRate: Target rate the segmenter expects (16 kHz).
    public init(url: URL, sampleRate: Int = 16_000) throws {
        self.url = url
        self.targetSampleRate = Double(sampleRate)
        let file = try AVAudioFile(forReading: url)
        self.sourceSampleRate = file.processingFormat.sampleRate
        // Sample count at the *target* rate.
        let ratio = Double(sampleRate) / file.processingFormat.sampleRate
        self.sampleCount = Int((Double(file.length) * ratio).rounded())
    }

    public func samples(start: Int, count: Int) -> [Float] {
        guard start >= 0, start < sampleCount, count > 0 else { return [] }
        let wanted = Swift.min(count, sampleCount - start)
        do {
            let file = try AVAudioFile(forReading: url)
            let format = file.processingFormat

            if Int(sourceSampleRate) == Int(targetSampleRate) {
                // Exact path: target indices == file frames, no resampling.
                file.framePosition = AVAudioFramePosition(start)
                return Self.read(file: file, format: format, frameCount: wanted)
            }

            // Resample path: read the native frames covering [start, start+wanted)
            // at the target rate, then convert that window to the target rate.
            let ratio = sourceSampleRate / targetSampleRate
            let nativeStart = AVAudioFramePosition((Double(start) * ratio).rounded(.down))
            let nativeCount = Int((Double(wanted) * ratio).rounded(.up)) + 1
            file.framePosition = nativeStart
            let native = Self.read(file: file, format: format, frameCount: nativeCount)
            guard !native.isEmpty else { return [] }
            let resampled = Self.resample(native, from: sourceSampleRate, to: targetSampleRate)
            return Array(resampled.prefix(wanted))
        } catch {
            return []
        }
    }

    private static func read(file: AVAudioFile, format: AVAudioFormat, frameCount: Int) -> [Float] {
        guard frameCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            return []
        }
        do {
            try file.read(into: buffer, frameCount: AVAudioFrameCount(frameCount))
        } catch {
            return []
        }
        guard let channelData = buffer.floatChannelData else { return [] }
        let n = Int(buffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData[0], count: n))
    }

    private static func resample(_ samples: [Float], from inputRate: Double, to outputRate: Double) -> [Float] {
        guard inputRate != outputRate, !samples.isEmpty,
              let sourceFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inputRate, channels: 1, interleaved: false),
              let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: outputRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: sourceFormat, to: targetFormat),
              let inBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(samples.count)) else {
            return samples
        }
        inBuffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            inBuffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        let outCapacity = AVAudioFrameCount(Double(samples.count) * outputRate / inputRate) + 1
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else { return samples }
        var fed = false
        var error: NSError?
        converter.convert(to: outBuffer, error: &error) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return inBuffer
        }
        guard error == nil, outBuffer.frameLength > 0 else { return samples }
        return Array(UnsafeBufferPointer(start: outBuffer.floatChannelData![0], count: Int(outBuffer.frameLength)))
    }
}
