#!/usr/bin/env swift
// Captures system audio using ScreenCaptureKit (macOS 13+).
// No BlackHole or Multi-Output Device needed.
// Usage: swift capture_system_audio.swift <output.wav> [duration_seconds]

import AVFoundation
import Foundation
import ScreenCaptureKit

func log(_ msg: String) {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
}

guard CommandLine.arguments.count >= 2 else {
    log("Usage: capture_system_audio <output.wav> [duration_seconds]")
    log("  If duration is omitted, records until killed (Ctrl+C or SIGTERM)")
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
    let pipePCM: Bool  // Write raw PCM (int16 mono 16kHz) to stdout

    init(outputURL: URL, pipePCM: Bool = false) {
        self.outputURL = outputURL
        self.pipePCM = pipePCM
        super.init()
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
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
        log("Recording system audio to: \(outputURL.path)")
    }

    func stop() async {
        guard isRecording else { return }
        isRecording = false
        try? await stream?.stopCapture()
        audioInput?.markAsFinished()
        await fileWriter?.finishWriting()
        log("Total audio samples: \(sampleCount)")
        log("Saved: \(outputURL.path)")
        let size = (try? FileManager.default.attributesOfItem(atPath: outputURL.path)[.size] as? Int) ?? 0
        log("Size: \(size / 1024) KB")
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, isRecording else { return }
        guard let audioInput = audioInput, audioInput.isReadyForMoreMediaData else { return }

        if !sessionStarted {
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            fileWriter?.startSession(atSourceTime: pts)
            sessionStarted = true
            log("First audio sample received (pts: \(CMTimeGetSeconds(pts)))")
        }

        sampleCount += 1
        audioInput.append(sampleBuffer)

        // Write raw PCM to stdout for streaming transcription
        if pipePCM {
            if let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) {
                let length = CMBlockBufferGetDataLength(blockBuffer)
                var data = Data(count: length)
                data.withUnsafeMutableBytes { ptr in
                    CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: length, destination: ptr.baseAddress!)
                }
                FileHandle.standardOutput.write(data)
            }
        }
    }
}

let pipePCM = CommandLine.arguments.contains("--pipe")
let recorder = AudioRecorder(outputURL: URL(fileURLWithPath: outputPath), pipePCM: pipePCM)
let semaphore = DispatchSemaphore(value: 0)

// Handle SIGINT/SIGTERM for graceful stop
var shouldStop = false

signal(SIGINT) { _ in shouldStop = true }
signal(SIGTERM) { _ in shouldStop = true }

Task {
    do {
        try await recorder.start()

        if let dur = duration {
            log("Recording for \(Int(dur)) seconds...")
            try await Task.sleep(nanoseconds: UInt64(dur * 1_000_000_000))
        } else {
            log("Recording until interrupted (Ctrl+C)...")
            while !shouldStop {
                try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }

        log("\nStopping...")
        await recorder.stop()
    } catch {
        log("Error: \(error)")
    }
    semaphore.signal()
}

semaphore.wait()
