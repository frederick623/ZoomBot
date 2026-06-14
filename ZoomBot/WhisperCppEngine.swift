import Foundation

/// Bridges the original whisper.cpp `WhisperContext` actor to the app's `WhisperEngine` protocol.
/// `WhisperContext` is an actor, so Swift enforces single-threaded access automatically —
/// no manual DispatchQueue needed.
final class WhisperCppEngine: WhisperEngine {
    private let whisperContext: WhisperContext

    /// - Parameters:
    ///   - modelURL: Path to the GGML `.bin` model file.
    ///   - language: BCP-47 language code passed to whisper.cpp (e.g. `"yue"` for Cantonese).
    init(modelURL: URL, language: String = "yue") throws {
        self.whisperContext = try WhisperContext.createContext(
            path: modelURL.path(),
            language: language
        )
    }

    func transcribe(samples: [Float]) async throws -> String {
        guard !samples.isEmpty else { return "" }
        await whisperContext.fullTranscribe(samples: samples)
        return await whisperContext.getTranscription()
    }
}
