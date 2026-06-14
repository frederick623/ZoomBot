
import Foundation
import QuartzCore

struct TranscriptUpdate {
    let text: String
    let isFinal: Bool
    let startTime: TimeInterval
    let endTime: TimeInterval
}

protocol WhisperEngine {
    /// Input: mono float32 PCM at 16kHz
    func transcribe(samples: [Float]) async throws -> String
}

final class WhisperStreamingService {
    var onUpdate: ((TranscriptUpdate) -> Void)?

    private let engine: WhisperEngine
    private let queue = DispatchQueue(label: "whisper.streaming.queue")

    private let sampleRate: Double = 16000
    private let emitInterval: TimeInterval = 2.0
    private let windowSeconds: TimeInterval = 12.0
    private let stableCommitEvery: Int = 3

    private var rollingSamples: [Float] = []
    private var isRunning = false
    private var timer: DispatchSourceTimer?
    private var tickCount = 0
    private var sessionStartTime: TimeInterval = CACurrentMediaTime()

    init(engine: WhisperEngine) {
        self.engine = engine
    }

    func start() {
        queue.async {
            guard !self.isRunning else { return }
            self.isRunning = true
            self.sessionStartTime = CACurrentMediaTime()
            self.startTimerLocked()
        }
    }

    func stop() {
        queue.async {
            self.isRunning = false
            self.timer?.cancel()
            self.timer = nil
        }
    }

    /// Append mono float32 PCM at 16kHz.
    func appendAudio(_ samples: [Float]) {
        queue.async {
            self.rollingSamples.append(contentsOf: samples)

            let maxSamples = Int(self.sampleRate * (self.windowSeconds + 8))
            if self.rollingSamples.count > maxSamples {
                self.rollingSamples.removeFirst(self.rollingSamples.count - maxSamples)
            }
        }
    }

    private func startTimerLocked() {
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + emitInterval, repeating: emitInterval)
        t.setEventHandler { [weak self] in
            guard let self else { return }
            Task { await self.runInferenceTick() }
        }
        t.resume()
        timer = t
    }

    private func runInferenceTick() async {
        guard isRunning else { return }

        let needed = Int(windowSeconds * sampleRate)
        guard rollingSamples.count >= needed / 2 else { return }

        let window = Array(rollingSamples.suffix(needed))
        let tickTime = CACurrentMediaTime() - sessionStartTime

        do {
            let text = try await engine.transcribe(samples: window).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return }

            tickCount += 1
            let final = tickCount % stableCommitEvery == 0
            let update = TranscriptUpdate(
                text: text,
                isFinal: final,
                startTime: max(0, tickTime - windowSeconds),
                endTime: tickTime
            )
            DispatchQueue.main.async { self.onUpdate?(update) }
        } catch {
            // Keep streaming even if one decode fails.
        }
    }
}

final class MockWhisperEngine: WhisperEngine {
    private var utteranceIndex: Int = 0

    func transcribe(samples: [Float]) async throws -> String {
        guard !samples.isEmpty else { return "" }

        let durationSec = Double(samples.count) / 16000.0

        var sumAbs: Float = 0
        var peak: Float = 0
        for s in samples {
            let v = abs(s)
            sumAbs += v
            if v > peak { peak = v }
        }
        let avg = sumAbs / Float(samples.count)

        // Very simple "speech activity" gate so silent chunks don't keep spamming text.
        if avg < 0.003 && peak < 0.02 {
            return ""
        }

        utteranceIndex += 1

        let intensity: String
        switch avg {
        case ..<0.010: intensity = "soft"
        case ..<0.030: intensity = "normal"
        default: intensity = "strong"
        }

        return String(
            format: "[mock-%03d] %@ speech detected (%.1fs window, avg=%.4f, peak=%.4f)",
            utteranceIndex,
            intensity,
            durationSec,
            avg,
            peak
        )
    }
}
