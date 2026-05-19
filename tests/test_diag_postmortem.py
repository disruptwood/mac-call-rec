"""Tests for postmortem diagnostics."""

from __future__ import annotations

from scripts.diag_postmortem import analyze_ffmpeg_log, first_existing_log, parse_timecode


def test_parse_timecode():
    assert parse_timecode("01:05:15.40") == 3915.4
    assert parse_timecode("05:15.40") == 315.4
    assert parse_timecode("bad") is None


def test_analyze_ffmpeg_log_reads_input_stream_not_output_stream(tmp_path):
    log = tmp_path / "_mic_seg000.wav.ffmpeg.log"
    log.write_text(
        "\n".join([
            "Input #0, avfoundation, from ':1':",
            "  Duration: N/A, start: 221870.309688, bitrate: 1536 kb/s",
            "  Stream #0:0: Audio: pcm_f32le, 48000 Hz, mono, flt, 1536 kb/s",
            "Stream mapping:",
            "Output #0, wav, to '_mic_seg000.wav':",
            "  Stream #0:0: Audio: pcm_s16le, 48000 Hz, mono, s16, 768 kb/s",
            "size=1KiB time=00:00:01.00 elapsed=0:00:01.00",
            "size=2KiB time=01:05:15.40 elapsed=1:05:15.44",
        ]),
    )

    info = analyze_ffmpeg_log(log)

    assert info["input_format"] == "pcm_f32le, 48000 Hz, mono, flt, 1536 kb/s"
    assert info["ffmpeg_time_seconds"] == 3915.4
    assert info["ffmpeg_elapsed_seconds"] == 3915.44


def test_first_existing_log_falls_back_to_segment_log(tmp_path):
    segment_log = tmp_path / "_mic_seg000.wav.ffmpeg.log"
    segment_log.write_text("log")

    log_path, segment_name = first_existing_log(
        tmp_path, "_mic.wav.ffmpeg.log", "_mic_seg*.wav.ffmpeg.log",
    )

    assert log_path == segment_log
    assert segment_name == "_mic_seg000.wav.ffmpeg.log"
