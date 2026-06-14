import Foundation
import CoreImage
import CoreVideo

final class SceneDetector {
    private let threshold: Float
    private let minInterval: TimeInterval
    private var lastFrameLuma: [UInt8]?
    private var lastCaptureTime: TimeInterval = 0

    init(threshold: Float = 0.25, minInterval: TimeInterval = 2.0) {
        self.threshold = threshold
        self.minInterval = minInterval
    }

    func detectChange(pixelBuffer: CVPixelBuffer, at timestamp: TimeInterval) -> Bool {
        guard timestamp - lastCaptureTime >= minInterval else { return false }

        let current = Self.downsampleLuma(pixelBuffer: pixelBuffer, width: 320, height: 240)

        guard let previous = lastFrameLuma else {
            lastFrameLuma = current
            lastCaptureTime = timestamp
            return true
        }

        let score = Self.normalizedAbsDiff(previous, current)
        if score > threshold {
            lastFrameLuma = current
            lastCaptureTime = timestamp
            return true
        }
        return false
    }

    private static func normalizedAbsDiff(_ a: [UInt8], _ b: [UInt8]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var total: Float = 0
        for i in 0..<a.count {
            total += Float(abs(Int(a[i]) - Int(b[i])))
        }
        return total / Float(a.count) / 255.0
    }

    private static func downsampleLuma(pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> [UInt8] {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return Array(repeating: 0, count: width * height)
        }

        let ptr = base.assumingMemoryBound(to: UInt8.self)
        var out = Array(repeating: UInt8(0), count: width * height)

        // Approximate luma from BGRA: Y ~= 0.299R + 0.587G + 0.114B
        for y in 0..<height {
            let sy = y * srcH / height
            for x in 0..<width {
                let sx = x * srcW / width
                let o = sy * bytesPerRow + sx * 4
                let b = Float(ptr[o + 0])
                let g = Float(ptr[o + 1])
                let r = Float(ptr[o + 2])
                let yVal = UInt8(max(0, min(255, Int(0.114 * b + 0.587 * g + 0.299 * r))))
                out[y * width + x] = yVal
            }
        }

        return out
    }
}
