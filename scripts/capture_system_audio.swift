#!/usr/bin/env swift
// Captures system audio using ScreenCaptureKit (macOS 13+).
// No BlackHole or Multi-Output Device needed.
// Usage: swift capture_system_audio.swift <output.wav> [duration_seconds]

import AVFoundation
import Foundation
import ScreenCaptureKit

guard CommandLine.arguments.count >= 2 else {
    print("Usage: capture_system_audio <output.wav> [duration_seconds]")
    print("  If duration is omitted, records until killed (Ctrl+C or SIGTERM)")
    exit(1)
}

let outputPath = CommandLine.arguments[1]
let duration: Double? = CommandLine.arguments.count >= 3 ? Double(CommandLine.arguments[2]) : nil

class AudioRecorder: NSObject, SCStreamOutput {
    var fileWriter: AVAssetWriter?
    var audioInput: AVAssetWriterInput?
    var stream: SCStream?
    var isRecording = false
    var sessionStarted = false
    var sampleCount = 0
    let outputURL: URL

    init(outputURL: URL) {
        self.outputURL = outputURL
        super.init()
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            print("ERROR: No display found")
            exit(1)
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = false
        config.sampleRate = 48000
        config.channelCount = 2

        // We only want audio, minimize video overhead
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1) // 1 fps min

        // Set up file writer
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
        fileWriter!.startWriting()
        // Session will start at first sample's timestamp

        let audioQueue = DispatchQueue(label: "com.callrecorder.audio")
        stream = SCStream(filter: filter, configuration: config, delegate: nil)
        try stream!.addStreamOutput(self, type: .audio, sampleHandlerQueue: audioQueue)
        try await stream!.startCapture()
        isRecording = true
        print("Recording system audio to: \(outputURL.path)")
    }

    func stop() async {
        guard isRecording else { return }
        isRecording = false
        try? await stream?.stopCapture()
        audioInput?.markAsFinished()
        await fileWriter?.finishWriting()
        print("Total audio samples: \(sampleCount)")
        print("Saved: \(outputURL.path)")
        let size = (try? FileManager.default.attributesOfItem(atPath: outputURL.path)[.size] as? Int) ?? 0
        print("Size: \(size / 1024) KB")
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, isRecording else { return }
        guard let audioInput = audioInput, audioInput.isReadyForMoreMediaData else { return }

        if !sessionStarted {
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            fileWriter?.startSession(atSourceTime: pts)
            sessionStarted = true
            print("First audio sample received (pts: \(CMTimeGetSeconds(pts)))")
        }

        sampleCount += 1
        audioInput.append(sampleBuffer)
    }
}

let recorder = AudioRecorder(outputURL: URL(fileURLWithPath: outputPath))
let semaphore = DispatchSemaphore(value: 0)

// Handle SIGINT/SIGTERM for graceful stop
var shouldStop = false

signal(SIGINT) { _ in shouldStop = true }
signal(SIGTERM) { _ in shouldStop = true }

Task {
    do {
        try await recorder.start()

        if let dur = duration {
            print("Recording for \(Int(dur)) seconds...")
            try await Task.sleep(nanoseconds: UInt64(dur * 1_000_000_000))
        } else {
            print("Recording until interrupted (Ctrl+C)...")
            while !shouldStop {
                try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }

        print("\nStopping...")
        await recorder.stop()
    } catch {
        print("Error: \(error)")
    }
    semaphore.signal()
}

semaphore.wait()
