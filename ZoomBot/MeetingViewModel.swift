import AVFoundation
import Combine
import Foundation
import SwiftUI

@MainActor
final class MeetingViewModel: ObservableObject {
    @Published var transcript: String = ""
    @Published var isRunning: Bool = false
    @Published var isTranscribingFile: Bool = false
    @Published var statusText: String = "Idle"
    @Published var meetingLink: String = ""
    @Published var meetingPassword: String = ""

    private var session: MeetingSessionController?
    private var engine: WhisperEngine?

    init() {
        setupSession()
    }

    // MARK: - Zoom meeting join

    func joinMeeting() {
        let link = meetingLink.trimmingCharacters(in: .whitespaces)
        guard !link.isEmpty else {
            statusText = "Enter a meeting link or ID"
            return
        }
        guard let zoomURL = buildZoomURL(from: link, password: meetingPassword) else {
            statusText = "Invalid meeting link or ID"
            return
        }
        NSWorkspace.shared.open(zoomURL)
        statusText = "Joining meeting…"
        // Give Zoom time to launch and show the meeting window, then start capture.
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            self.start()
        }
    }

    /// Build a `zoommtg://` deep-link URL from a Zoom meeting link or bare meeting ID.
    private func buildZoomURL(from input: String, password: String) -> URL? {
        let trimmed = input.trimmingCharacters(in: .whitespaces)
        var meetingID: String?
        var pwd = password.trimmingCharacters(in: .whitespaces)

        // Try parsing as a URL (with or without scheme).
        let urlString = trimmed.hasPrefix("http") ? trimmed : "https://\(trimmed)"
        if let url = URL(string: urlString),
           let host = url.host, host.contains("zoom.us") {
            let components = url.pathComponents
            if let jIdx = components.firstIndex(of: "j"), jIdx + 1 < components.count {
                meetingID = components[jIdx + 1]
            }
            // Pull embedded password from the URL query string if the user didn't supply one.
            if pwd.isEmpty,
               let queryItems = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems {
                pwd = queryItems.first(where: { $0.name == "pwd" })?.value ?? ""
            }
        }

        // Fall back: treat the input as a bare numeric meeting ID (9–11 digits).
        if meetingID == nil {
            let digits = trimmed.filter(\.isNumber)
            if (9...11).contains(digits.count) {
                meetingID = digits
            }
        }

        guard let id = meetingID else { return nil }

        var urlComponents = URLComponents()
        urlComponents.scheme = "zoommtg"
        urlComponents.host = "zoom.us"
        urlComponents.path = "/join"
        var queryItems = [
            URLQueryItem(name: "action", value: "join"),
            URLQueryItem(name: "confno", value: id)
        ]
        if !pwd.isEmpty {
            queryItems.append(URLQueryItem(name: "pwd", value: pwd))
        }
        urlComponents.queryItems = queryItems
        return urlComponents.url
    }

    func start() {
        guard !isRunning else { return }
        session?.start()
        isRunning = true
        statusText = "Capturing"
    }

    func stop() {
        guard isRunning else { return }
        session?.stop()
        isRunning = false
        statusText = "Stopped"
    }

    // MARK: - File transcription

    func transcribeFile(url: URL) {
        guard !isTranscribingFile, !isRunning else { return }
        Task { await transcribeFileAsync(url: url) }
    }

    private func transcribeFileAsync(url: URL) async {
        guard let engine else {
            statusText = "Engine not ready"
            return
        }
        isTranscribingFile = true
        transcript = ""
        statusText = "Loading audio…"

        defer { isTranscribingFile = false }

        do {
            let samples = try loadAudioFile(url: url)
            let chunkSize = 30 * 16_000          // 30-second chunks at 16 kHz
            var result = ""
            var offset = 0

            while offset < samples.count {
                let end = min(offset + chunkSize, samples.count)
                let chunk = Array(samples[offset..<end])
                let progress = min(100, Int(Double(end) / Double(samples.count) * 100))
                statusText = "Transcribing… \(progress)%"

                let text = try await engine.transcribe(samples: chunk)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty {
                    result += (result.isEmpty ? "" : "\n") + text
                    transcript = result
                }
                offset += chunkSize
            }
            statusText = result.isEmpty ? "No speech detected" : "Done"
        } catch {
            statusText = "Error: \(error.localizedDescription)"
        }
    }

    /// Decode any audio file to mono Float32 PCM at 16 kHz.
    private nonisolated func loadAudioFile(url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let inputFormat = audioFile.processingFormat

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ) else {
            throw CocoaError(.fileReadUnknown)
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw CocoaError(.fileReadUnknown)
        }

        // Read the whole file into an input buffer.
        let inputFrameCount = AVAudioFrameCount(audioFile.length)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrameCount) else {
            throw CocoaError(.fileReadUnknown)
        }
        try audioFile.read(into: inputBuffer)

        // Allocate an output buffer sized for the resampled audio.
        let ratio = 16_000.0 / inputFormat.sampleRate
        let outputFrameCount = AVAudioFrameCount(ceil(Double(inputFrameCount) * ratio)) + 1
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
            throw CocoaError(.fileReadUnknown)
        }

        // Single-pass conversion (the entire input is already in memory).
        var inputProvided = false
        var conversionError: NSError?
        converter.convert(to: outputBuffer, error: &conversionError) { _, statusPtr in
            if inputProvided {
                statusPtr.pointee = .endOfStream
                return nil
            }
            inputProvided = true
            statusPtr.pointee = .haveData
            return inputBuffer
        }
        if let err = conversionError { throw err }

        guard let channelData = outputBuffer.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: channelData[0], count: Int(outputBuffer.frameLength)))
    }

    // MARK: - Private setup

    private func setupSession() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let out = docs.appendingPathComponent("output", isDirectory: true)

        do {
            let whisperEngine: WhisperEngine
            if let modelURL = WhisperModelLocator.defaultModelURL() {
                whisperEngine = try WhisperCppEngine(modelURL: modelURL)
                statusText = "Ready · Whisper \(modelURL.lastPathComponent)"
            } else {
                whisperEngine = MockWhisperEngine()
                statusText = "Ready · Mock (add ggml-*.bin to bundle or Documents/models)"
            }

            self.engine = whisperEngine

            let created = try MeetingSessionController(outputDir: out, whisperEngine: whisperEngine)
            created.onTranscriptChanged = { [weak self] text in
                self?.transcript = text
            }
            self.session = created
        } catch {
            self.statusText = "Init failed: \(error.localizedDescription)"
        }
    }
}
