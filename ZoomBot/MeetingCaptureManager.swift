import Foundation
import ReplayKit
import AVFoundation

protocol MeetingCaptureDelegate: AnyObject {
    func didCaptureVideoFrame(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval)
    func didCaptureAudioSamples(_ samples: [Float], sampleRate: Double)
}

final class MeetingCaptureManager: NSObject {
    weak var delegate: MeetingCaptureDelegate?

    private let recorder = RPScreenRecorder.shared()

    func start() {
        recorder.isMicrophoneEnabled = true
        recorder.startCapture(handler: { [weak self] sampleBuffer, sampleType, error in
            if error != nil { return }
            guard let self else { return }

            switch sampleType {
            case .video:
                if let pb = CMSampleBufferGetImageBuffer(sampleBuffer) {
                    let ts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
                    self.delegate?.didCaptureVideoFrame(pb, timestamp: ts)
                }
            case .audioMic, .audioApp:
                guard let mono16k = Self.extractMono16kFloatPCM(sampleBuffer: sampleBuffer) else { return }
                self.delegate?.didCaptureAudioSamples(mono16k, sampleRate: 16000)
            @unknown default:
                break
            }
        }, completionHandler: { _ in })
    }

    func stop() {
        recorder.stopCapture { _ in }
    }

    private static func extractMono16kFloatPCM(sampleBuffer: CMSampleBuffer) -> [Float]? {
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return nil }
        let length = CMBlockBufferGetDataLength(blockBuffer)
        guard length > 0 else { return nil }

        var data = Data(count: length)
        data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            _ = CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: length, destination: base)
        }

        // Draft simplification: assumes Float32 PCM from source format path.
        let floats = data.withUnsafeBytes { raw -> [Float] in
            let p = raw.bindMemory(to: Float.self)
            return Array(p)
        }

        // If source isn't already 16k mono, insert AVAudioConverter here.
        return floats
    }
}
