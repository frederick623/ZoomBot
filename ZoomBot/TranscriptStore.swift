import Foundation

final class TranscriptStore {
    private(set) var finalSegments: [String] = []
    private(set) var livePartial: String = ""

    func apply(_ update: TranscriptUpdate) {
        if update.isFinal {
            finalSegments.append(update.text)
            livePartial = ""
        } else {
            livePartial = update.text
        }
    }

    func fullTranscript() -> String {
        if livePartial.isEmpty {
            return finalSegments.joined(separator: "\n")
        }
        if finalSegments.isEmpty {
            return livePartial
        }
        return finalSegments.joined(separator: "\n") + "\n" + livePartial
    }
}
