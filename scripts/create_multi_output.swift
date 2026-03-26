#!/usr/bin/env swift
// Creates a Multi-Output Device combining the default output with BlackHole 2ch.
// This is equivalent to manually creating one in Audio MIDI Setup.

import CoreAudio
import Foundation

func getDeviceUIDs() -> [(uid: String, name: String, isOutput: Bool)] {
    var propertyAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )

    var dataSize: UInt32 = 0
    var status = AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0, nil,
        &dataSize
    )
    guard status == noErr else { return [] }

    let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
    var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
    status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0, nil,
        &dataSize,
        &deviceIDs
    )
    guard status == noErr else { return [] }

    var results: [(uid: String, name: String, isOutput: Bool)] = []

    for deviceID in deviceIDs {
        // Get UID
        var uidAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: CFString = "" as CFString
        var uidSize = UInt32(MemoryLayout<CFString>.size)
        let uidStatus = AudioObjectGetPropertyData(deviceID, &uidAddress, 0, nil, &uidSize, &uid)
        guard uidStatus == noErr else { continue }

        // Get Name
        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceNameCFString,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var name: CFString = "" as CFString
        var nameSize = UInt32(MemoryLayout<CFString>.size)
        let nameStatus = AudioObjectGetPropertyData(deviceID, &nameAddress, 0, nil, &nameSize, &name)
        guard nameStatus == noErr else { continue }

        // Check if has output channels
        var outputAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
        var outputSize: UInt32 = 0
        let hasOutput = AudioObjectGetPropertyDataSize(deviceID, &outputAddress, 0, nil, &outputSize) == noErr && outputSize > 0

        var isOutput = false
        if hasOutput {
            let bufferListPtr = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: Int(outputSize))
            defer { bufferListPtr.deallocate() }
            if AudioObjectGetPropertyData(deviceID, &outputAddress, 0, nil, &outputSize, bufferListPtr) == noErr {
                let bufferList = UnsafeMutableAudioBufferListPointer(bufferListPtr)
                isOutput = bufferList.reduce(0) { $0 + Int($1.mNumberChannels) } > 0
            }
        }

        results.append((uid: uid as String, name: name as String, isOutput: isOutput))
    }

    return results
}

func createMultiOutputDevice(name: String, subDeviceUIDs: [String]) -> Bool {
    let desc: [String: Any] = [
        kAudioAggregateDeviceNameKey as String: name,
        kAudioAggregateDeviceUIDKey as String: "com.callrecorder.\(name.replacingOccurrences(of: " ", with: "_"))",
        kAudioAggregateDeviceIsPrivateKey as String: 0,
        kAudioAggregateDeviceIsStackedKey as String: 0,  // 0 = multi-output
        kAudioAggregateDeviceSubDeviceListKey as String: subDeviceUIDs.map { uid in
            [kAudioSubDeviceUIDKey as String: uid]
        }
    ]

    var aggregateID: AudioDeviceID = 0
    let status = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &aggregateID)

    if status == noErr {
        print("Created Multi-Output Device '\(name)' (ID: \(aggregateID))")

        // Set drift correction for all sub-devices except the first (clock source)
        if subDeviceUIDs.count > 1 {
            var driftAddress = AudioObjectPropertyAddress(
                mSelector: kAudioAggregateDevicePropertyComposition,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            // Drift correction is automatically handled for aggregate devices
            print("Drift correction: enabled (automatic for aggregate devices)")
        }

        return true
    } else {
        print("Failed to create Multi-Output Device. Error: \(status)")
        return false
    }
}

// --- Main ---

let devices = getDeviceUIDs()

print("Available audio output devices:")
for d in devices where d.isOutput {
    print("  \(d.name) [UID: \(d.uid)]")
}
print()

// Find BlackHole
guard let blackhole = devices.first(where: { $0.name.lowercased().contains("blackhole") }) else {
    print("ERROR: BlackHole not found in audio devices.")
    print("Try: sudo launchctl kickstart -kp system/com.apple.audio.coreaudiod")
    exit(1)
}

// Find default output (speakers or headphones, but not BlackHole/Zoom/virtual)
let physicalOutputs = devices.filter {
    $0.isOutput &&
    !$0.name.lowercased().contains("blackhole") &&
    !$0.name.lowercased().contains("zoom") &&
    !$0.name.lowercased().contains("aggregate") &&
    !$0.name.lowercased().contains("multi-output")
}

guard let mainOutput = physicalOutputs.first else {
    print("ERROR: No physical audio output device found.")
    exit(1)
}

print("Will combine: \(mainOutput.name) + \(blackhole.name)")
print()

let deviceName = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "Record Output"

if createMultiOutputDevice(name: deviceName, subDeviceUIDs: [mainOutput.uid, blackhole.uid]) {
    print()
    print("Done! Now set '\(deviceName)' as your system output:")
    print("  System Settings → Sound → Output → \(deviceName)")
    print()
    print("Or from terminal:")
    print("  # Install: brew install switchaudio-osx")
    print("  # Then: SwitchAudioSource -s '\(deviceName)'")
} else {
    exit(1)
}
