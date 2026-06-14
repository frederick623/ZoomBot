import Foundation
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers

final class MeetingSessionController: MeetingCaptureDelegate {
    private let captureManager: MeetingCaptureManager
    private let sceneDetector: SceneDetector
    private let whisper: WhisperStreamingService
    private let transcripts = TranscriptStore()

    private let outputDir: URL
    private let screenshotsDir: URL
    private var screenshotCount = 0

    var onTranscriptChanged: ((String) -> Void)?

    init(outputDir: URL, whisperEngine: WhisperEngine) throws {
        self.outputDir = outputDir
        self.screenshotsDir = outputDir.appendingPathComponent("screenshots", isDirectory: true)
        self.captureManager = MeetingCaptureManager()
        self.sceneDetector = SceneDetector(threshold: 0.25, minInterval: 2.0)
        self.whisper = WhisperStreamingService(engine: whisperEngine)

        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: screenshotsDir, withIntermediateDirectories: true)

        captureManager.delegate = self
        whisper.onUpdate = { [weak self] update in
            guard let self else { return }
            self.transcripts.apply(update)
            self.onTranscriptChanged?(self.transcripts.fullTranscript())
        }
    }

    func start() {
        whisper.start()
        captureManager.start()
    }

    func stop() {
        captureManager.stop()
        whisper.stop()
        saveTranscript()
    }

    func didCaptureVideoFrame(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        if sceneDetector.detectChange(pixelBuffer: pixelBuffer, at: timestamp) {
            saveScreenshot(pixelBuffer)
        }
    }

    func didCaptureAudioSamples(_ samples: [Float], sampleRate: Double) {
        // Sample rate expected 16k from capture manager in this draft.
        whisper.appendAudio(samples)
    }

    private func saveScreenshot(_ pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }

        screenshotCount += 1
        let ts = Int(Date().timeIntervalSince1970)
        let url = screenshotsDir.appendingPathComponent("slide_\(String(format: "%04d", screenshotCount))_\(ts).png")

        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dest, cgImage, nil)
        CGImageDestinationFinalize(dest)
    }

    private func saveTranscript() {
        let text = transcripts.fullTranscript()
        let url = outputDir.appendingPathComponent("transcript.txt")
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }
}
