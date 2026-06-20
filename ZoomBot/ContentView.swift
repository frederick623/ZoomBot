import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var vm = MeetingViewModel()
    @State private var showFileImporter = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Live Captions")
                .font(.title2).bold()

            // MARK: – Meeting input
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Join Zoom Meeting")
                        .font(.headline)

                    TextField("Meeting link or ID (e.g. https://zoom.us/j/123… or 123 456 789)", text: $vm.meetingLink)
                        .textFieldStyle(.roundedBorder)
                        .disabled(vm.isRunning || vm.isTranscribingFile)

                    SecureField("Password (if required)", text: $vm.meetingPassword)
                        .textFieldStyle(.roundedBorder)
                        .disabled(vm.isRunning || vm.isTranscribingFile)

                    Button {
                        vm.joinMeeting()
                    } label: {
                        Label("Join & Transcribe", systemImage: "video.badge.plus")
                    }
                    .buttonStyle(CustomHeightButtonStyle())
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .disabled(vm.meetingLink.trimmingCharacters(in: .whitespaces).isEmpty
                              || vm.isRunning || vm.isTranscribingFile)
                }
                .padding(4)
            }

            ScrollView {
                if vm.transcript.isEmpty {
                    Text("Captions will appear here...")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                } else {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(splitSentences(vm.transcript).enumerated()), id: \.offset) { _, sentence in
                            Text(sentence)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .textSelection(.enabled)
                    .padding(12)
                }
            }
            .frame(maxHeight: .infinity)
            .background(Color.gray.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 12))

            HStack(spacing: 12) {
                Button {
                    vm.isRunning ? vm.stop() : vm.start()
                } label: {
                    Label(vm.isRunning ? "Stop" : "Start",
                          systemImage: vm.isRunning ? "stop.fill" : "play.fill")
                }
                .buttonStyle(CustomHeightButtonStyle())
                .tint(vm.isRunning ? .red : .accentColor)
                .disabled(vm.isTranscribingFile)

                Button {
                    showFileImporter = true
                } label: {
                    Label("Open File", systemImage: "doc.badge.arrow.up")
                }
                .buttonStyle(CustomHeightButtonStyle())
                .disabled(vm.isRunning || vm.isTranscribingFile)

                Button {
                    saveCaptions()
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(CustomHeightButtonStyle())
                .disabled(vm.transcript.isEmpty)

                Spacer()

                if vm.isTranscribingFile {
                    ProgressView()
                        .scaleEffect(0.6)
                        .frame(width: 16, height: 16)
                }

                Text(vm.statusText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio, .wav, .aiff, .mp3,
                                  UTType(filenameExtension: "m4a") ?? .audio,
                                  UTType(filenameExtension: "caf") ?? .audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                // Start accessing the security-scoped resource.
                guard url.startAccessingSecurityScopedResource() else { return }
                vm.transcribeFile(url: url)
                // Release after the async task picks it up (slight delay is fine for local files).
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                    url.stopAccessingSecurityScopedResource()
                }
            case .failure:
                break
            }
        }
    }

    struct CustomHeightButtonStyle: ButtonStyle {
        var height: CGFloat = 50
        
        func makeBody(configuration: Configuration) -> some View {
            configuration.label
                .font(.body)
                .bold()
                .foregroundColor(.white)
                .frame(maxWidth: 120) // Fills width
                .frame(height: 40)      // Sets exact height
                .background(Color.black)
                .cornerRadius(8)
                .opacity(configuration.isPressed ? 0.8 : 1.0) // Tap animation
        }
    }

    private func saveCaptions() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "captions.txt"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        // Transcript already has \n between each paragraph from TranscriptStore.
        try? vm.transcript.write(to: url, atomically: true, encoding: .utf8)
    }
    /// Splits a transcript into individual sentences for line-by-line display.
    /// Handles both Western (`.`, `?`, `!`) and CJK (`。`, `？`, `！`) terminators,
    /// and preserves existing newline-separated segments from the transcript store.
    private func splitSentences(_ text: String) -> [String] {
        let terminators: Set<Character> = [".", "?", "!", "。", "？", "！", "…"]
        var sentences: [String] = []
        var current = ""

        for line in text.split(whereSeparator: { $0.isNewline }) {
            for ch in line {
                current.append(ch)
                if terminators.contains(ch) {
                    let trimmed = current.trimmingCharacters(in: .whitespaces)
                    if !trimmed.isEmpty { sentences.append(trimmed) }
                    current = ""
                }
            }
            let trimmed = current.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { sentences.append(trimmed) }
            current = ""
        }
        return sentences
    }
}

#Preview {
    ContentView()
}
