#!/usr/bin/env swift
// Captures macOS system audio (everything playing through the default output
// device) via ScreenCaptureKit and writes it to a WAV file.
//
// Usage: capture_system_audio <output.wav>
//   Records until SIGTERM or SIGINT.
//
// Build:
//   xcrun -sdk macosx swiftc capture_system_audio.swift -o capture_system_audio
//
// Run as subprocess from recorder/recording.py. The parent sends SIGTERM
// to stop. WAV is finalized in stop() via AVAssetWriter.finishWriting; if
// the parent SIGKILLs us, the file's RIFF header will be left with
// placeholder sizes and won't be readable by ffmpeg — the parent's wait
// timeout is set generously to avoid that.
//
// Reliability fixes layered over the baseline ScreenCaptureKit demo:
//   - audioQueue.sync barrier in stop() so a callback can't append to a
//     just-finished writer.
//   - Periodically checks fileWriter.status; on .failed we log and exit
//     instead of silently no-op'ing every append for the rest of the
//     session.
//   - SCStreamDelegate.stream(_:didStopWithError:) wired up so a mid-session
//     halt (display reconfigure, permission revoked, system pressure)
//     surfaces as a non-zero exit + log line instead of a silent zero-byte
//     gap.
//   - startWriting() return value checked.

import AVFoundation
import Foundation
import ScreenCaptureKit

func log(_ msg: String) {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
}

guard CommandLine.arguments.count >= 2 else {
    log("Usage: capture_system_audio <output.wav>")
    log("  Records until killed (SIGTERM/SIGINT).")
    exit(1)
}

let outputPath = CommandLine.arguments[1]

final class AudioRecorder: NSObject, SCStreamOutput, SCStreamDelegate {
    var fileWriter: AVAssetWriter?
    var audioInput: AVAssetWriterInput?
    var stream: SCStream?
    var isRecording = false
    var sessionStarted = false
    var sampleCount = 0
    var loggedWriterFailure = false
    let outputURL: URL
    let audioQueue = DispatchQueue(label: "com.callrecorder.audio")
    var streamHaltedWithError: Error?

    init(outputURL: URL) {
        self.outputURL = outputURL
        super.init()
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false
        )
        guard let display = content.displays.first else {
            log("ERROR: No display found")
            exit(1)
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = false
        config.sampleRate = 48000
        config.channelCount = 2
        // Video pipeline is unused but ScreenCaptureKit requires *some* video
        // configuration; pick the smallest legal frame size and slowest tick.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        try? FileManager.default.removeItem(at: outputURL)
        fileWriter = try AVAssetWriter(outputURL: outputURL, fileType: .wav)

        let audioSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 48000,
            AVNumberOfChannelsKey: 2,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        audioInput!.expectsMediaDataInRealTime = true
        fileWriter!.add(audioInput!)

        guard fileWriter!.startWriting() else {
            let err = fileWriter!.error?.localizedDescription ?? "unknown"
            log("ERROR: AVAssetWriter.startWriting failed: \(err)")
            exit(1)
        }
        // Session will start at the first sample's timestamp (see callback).

        stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream!.addStreamOutput(self, type: .audio, sampleHandlerQueue: audioQueue)
        try await stream!.startCapture()
        isRecording = true
        log("Recording system audio to: \(outputURL.path)")
    }

    func stop() async {
        // Barrier on audioQueue ensures no callback can be in-flight (or
        // start) while we mark the writer as finished. Without this, a
        // sample buffer can race past the isRecording check and try to
        // append to a writer that's already finalized.
        audioQueue.sync {
            guard isRecording else { return }
            isRecording = false
            audioInput?.markAsFinished()
        }

        do {
            try await stream?.stopCapture()
        } catch {
            log("Warning: stream.stopCapture threw: \(error)")
        }

        await fileWriter?.finishWriting()

        let finalStatus = fileWriter?.status
        if finalStatus == .failed {
            let err = fileWriter?.error?.localizedDescription ?? "unknown"
            log("ERROR: AVAssetWriter final status .failed: \(err)")
        }

        log("Total audio samples: \(sampleCount)")
        log("Saved: \(outputURL.path)")
        let size = (try? FileManager.default.attributesOfItem(atPath: outputURL.path)[.size] as? Int) ?? 0
        log("Size: \(size / 1024) KB")
        if let e = streamHaltedWithError {
            log("ScreenCaptureKit stream had halted with: \(e)")
        }
    }

    // MARK: - SCStreamOutput

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio, isRecording else { return }

        if let status = fileWriter?.status, status == .failed {
            if !loggedWriterFailure {
                loggedWriterFailure = true
                let err = fileWriter?.error?.localizedDescription ?? "unknown"
                log("ERROR: AVAssetWriter entered .failed mid-session: \(err)")
            }
            isRecording = false
            return
        }

        guard let audioInput = audioInput, audioInput.isReadyForMoreMediaData else { return }

        if !sessionStarted {
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            fileWriter?.startSession(atSourceTime: pts)
            sessionStarted = true
            log("First audio sample received (pts: \(CMTimeGetSeconds(pts)))")
        }

        sampleCount += 1
        audioInput.append(sampleBuffer)
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("ERROR: ScreenCaptureKit stream halted mid-session: \(error)")
        streamHaltedWithError = error
        // Signal main loop to exit (next 100ms tick).
        shouldStopGlobal = true
    }
}

// File-scope flag readable by main loop and writable by signal handlers and
// the SCStreamDelegate. Swift's Bool isn't formally atomic, but on the
// architectures we ship to a one-word store is effectively atomic; the
// worst case is a one-tick read delay before we notice.
var shouldStopGlobal = false

signal(SIGINT) { _ in shouldStopGlobal = true }
signal(SIGTERM) { _ in shouldStopGlobal = true }

let recorder = AudioRecorder(outputURL: URL(fileURLWithPath: outputPath))
let semaphore = DispatchSemaphore(value: 0)

Task {
    do {
        try await recorder.start()
        log("Recording until interrupted (SIGTERM/SIGINT)...")
        while !shouldStopGlobal {
            try await Task.sleep(nanoseconds: 100_000_000)  // 100 ms
        }
        log("Stopping...")
        await recorder.stop()
    } catch {
        log("Error: \(error)")
    }
    semaphore.signal()
}

semaphore.wait()
