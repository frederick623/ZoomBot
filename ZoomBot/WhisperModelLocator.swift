import Foundation

enum WhisperModelLocator {
    private static let preferredModelName = "ggml-large-v3"
    private static let modelExt = "bin"

    static func defaultModelURL(fileManager: FileManager = .default) -> URL? {
        if let repoModel = preferredRepositoryModel(fileManager: fileManager) {
            return repoModel
        }

        if let bundled = firstBundledModel(fileManager: fileManager) {
            return bundled
        }

        return firstDocumentModel(fileManager: fileManager)
    }

    private static func preferredRepositoryModel(fileManager: FileManager) -> URL? {
        // whisper.cpp expects the GGML .bin path and derives the paired
        // CoreML encoder path by replacing `.bin` with `-encoder.mlmodelc`.
        let relativePaths = [
            "whisper.cpp/models/\(preferredModelName)."+modelExt,
            "models/\(preferredModelName)."+modelExt
        ]

        for root in candidateSearchRoots(fileManager: fileManager) {
            for relativePath in relativePaths {
                let candidate = root.appendingPathComponent(relativePath, isDirectory: false)
                if fileManager.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
        }

        return nil
    }

    private static func firstBundledModel(fileManager: FileManager) -> URL? {
        let namedCandidates = [
            preferredModelName,
            "ggml-base.en",
            "ggml-base",
            "ggml-small.en",
            "ggml-small",
            "ggml-medium.en",
            "ggml-tiny.en"
        ]

        for name in namedCandidates {
            if let url = Bundle.main.url(forResource: name, withExtension: modelExt),
               fileManager.fileExists(atPath: url.path) {
                return url
            }
        }

        guard let resourceURL = Bundle.main.resourceURL,
              let contents = try? fileManager.contentsOfDirectory(
                at: resourceURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
              ) else {
            return nil
        }

        return contents.first(where: {
            $0.pathExtension == modelExt && $0.lastPathComponent.hasPrefix("ggml-")
        })
    }

    private static func firstDocumentModel(fileManager: FileManager) -> URL? {
        guard let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }

        let directories = [
            documentsURL,
            documentsURL.appendingPathComponent("models", isDirectory: true),
            documentsURL.appendingPathComponent("Models", isDirectory: true)
        ]

        for directory in directories {
            let preferred = directory.appendingPathComponent("\(preferredModelName)."+modelExt, isDirectory: false)
            if fileManager.fileExists(atPath: preferred.path) {
                return preferred
            }

            guard let contents = try? fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            if let model = contents.first(where: {
                $0.pathExtension == modelExt && $0.lastPathComponent.hasPrefix("ggml-")
            }) {
                return model
            }
        }

        return nil
    }

    private static func candidateSearchRoots(fileManager: FileManager) -> [URL] {
        var baseCandidates: [URL?] = [
            URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true),
            Bundle.main.resourceURL,
            Bundle.main.bundleURL.deletingLastPathComponent(),
            fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        ]
        #if os(macOS)
        baseCandidates.append(fileManager.homeDirectoryForCurrentUser)
        #endif

        var roots: [URL] = []
        for base in baseCandidates.compactMap({ $0?.standardizedFileURL }) {
            appendAncestors(of: base, to: &roots)
        }

        return roots
    }

    private static func appendAncestors(of base: URL, to roots: inout [URL]) {
        var current = base
        while true {
            if !roots.contains(where: { $0.path == current.path }) {
                roots.append(current)
            }

            let parent = current.deletingLastPathComponent().standardizedFileURL
            if parent.path == current.path {
                break
            }

            current = parent
        }
    }
}
