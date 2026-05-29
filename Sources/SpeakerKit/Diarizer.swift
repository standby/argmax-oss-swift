//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import ArgmaxCore
import WhisperKit

/// Protocol for diarization backends (Pyannote, Sortformer, etc.)
///
/// This protocol defines the interface for speaker diarization implementations.
/// Concrete implementations (e.g., `PyannoteDiarizer`, `SortformerDiarizer`) handle
/// model loading, downloading, and diarization processing.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol Diarizer: Sendable {
    /// The folder containing the diarization models
    var modelFolder: URL? { get }
    
    /// Current state of the models (loaded, unloaded, etc.)
    var modelState: ModelState { get }
    
    /// Download the diarization models if needed
    func downloadModels() async throws

    /// Load the diarization models into memory
    func loadModels() async throws
    
    /// Unload the diarization models from memory
    func unloadModels() async
    
    /// Perform speaker diarization on audio
    ///
    /// - Parameters:
    ///   - audioArray: Audio samples as floating-point array (16kHz, mono)
    ///   - options: Optional diarization configuration options
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Diarization result with speaker segments
    func diarize(
        audioArray: [Float],
        options: (any DiarizationOptions)?,
        progressCallback: (@Sendable (Progress) -> Void)?
    ) async throws -> DiarizationResult

    /// Perform speaker diarization streaming windows from an audio file, so the
    /// full decoded PCM need not be resident. Backends that support it (e.g.
    /// `PyannoteDiarizer`) override this; others fall back to decoding the file.
    func diarize(
        audioFile url: URL,
        sampleRate: Int,
        options: (any DiarizationOptions)?,
        progressCallback: (@Sendable (Progress) -> Void)?
    ) async throws -> DiarizationResult
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public extension Diarizer {
    /// Default: decode the whole file, then diarize from the in-memory array.
    /// Streaming backends override this to avoid materializing the full PCM.
    func diarize(
        audioFile url: URL,
        sampleRate: Int = 16_000,
        options: (any DiarizationOptions)? = nil,
        progressCallback: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> DiarizationResult {
        let buffer = try AudioProcessor.loadAudio(fromPath: url.path)
        let samples = AudioProcessor.convertBufferToArray(buffer: buffer)
        return try await diarize(audioArray: samples, options: options, progressCallback: progressCallback)
    }
}
