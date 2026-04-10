"""
tests/test_malibu_sync.py — Unit tests for MalibuSync components.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from scipy.io import wavfile

import subtitle_tools
import sync_engine


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def noise_wavs():
    """Return (ref_wav_path, hq_wav_path, expected_offset_ms) with a 1.5 s shift."""
    np.random.seed(0)
    rate   = 16000
    length = 130  # seconds — long enough for scale estimation (>2*SEGMENT_SEC)
    noise  = (np.random.randn(rate * length) * 10000).astype(np.int16)

    shift = int(1.5 * rate)  # HQ is 1.5 s ahead of reference
    hq    = np.concatenate([noise[shift:], np.zeros(shift, dtype=np.int16)])

    tmpdir = tempfile.mkdtemp()
    ref_path = os.path.join(tmpdir, "ref.wav")
    hq_path  = os.path.join(tmpdir, "hq.wav")
    wavfile.write(ref_path, rate, noise)
    wavfile.write(hq_path,  rate, hq)

    yield ref_path, hq_path, 1500.0


@pytest.fixture()
def sample_srt(tmp_path):
    """Write a minimal SRT file and return its path."""
    content = (
        "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n"
        "2\n00:00:04,500 --> 00:00:06,000\nWorld\n\n"
        "3\n00:00:07,000 --> 00:00:09,000\nEnd\n\n"
    )
    p = tmp_path / "sample.srt"
    p.write_text(content, encoding="utf-8")
    return str(p)


# ── sync_engine tests ─────────────────────────────────────────────────────────

class TestComputeSyncParams:
    def test_correct_offset(self, noise_wavs):
        ref, hq, expected = noise_wavs
        offset_ms, scale = sync_engine.compute_sync_params(ref, hq)
        assert abs(offset_ms - expected) < 50, (
            f"offset {offset_ms:.1f} ms not close to expected {expected} ms"
        )

    def test_scale_is_one_when_no_drift(self, noise_wavs):
        ref, hq, _ = noise_wavs
        _, scale = sync_engine.compute_sync_params(ref, hq)
        assert abs(scale - 1.0) < 0.005, f"scale {scale:.6f} deviates too much from 1.0"

    def test_zero_offset_when_identical(self, noise_wavs):
        ref, _, _ = noise_wavs
        offset_ms, scale = sync_engine.compute_sync_params(ref, ref)
        assert abs(offset_ms) < 50
        assert abs(scale - 1.0) < 0.005


# ── subtitle_tools tests ──────────────────────────────────────────────────────

class TestParseSrt:
    def test_entry_count(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        assert len(entries) == 3

    def test_timestamps(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        assert entries[0].start_ms == 1000
        assert entries[0].end_ms   == 3000
        assert entries[1].start_ms == 4500

    def test_text(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        assert entries[0].text == "Hello"
        assert entries[1].text == "World"

    def test_windows_line_endings(self, tmp_path):
        content = "1\r\n00:00:01,000 --> 00:00:02,000\r\nTest\r\n\r\n"
        p = tmp_path / "win.srt"
        p.write_bytes(content.encode("utf-8"))
        entries = subtitle_tools.parse_srt(str(p))
        assert len(entries) == 1
        assert entries[0].start_ms == 1000

    def test_bom_marker(self, tmp_path):
        # write_bytes with a manual BOM prefix simulates UTF-8-BOM files
        content = "1\n00:00:01,000 --> 00:00:02,000\nBOM test\n\n"
        p = tmp_path / "bom.srt"
        p.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
        entries = subtitle_tools.parse_srt(str(p))
        assert len(entries) == 1


class TestApplyLinearSync:
    def test_offset_only(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        synced  = subtitle_tools.apply_linear_sync(entries, offset_ms=500)
        assert synced[0].start_ms == 1500
        assert synced[0].end_ms   == 3500
        assert synced[0].text == entries[0].text

    def test_scale_only(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        synced  = subtitle_tools.apply_linear_sync(entries, offset_ms=0, scale=2.0)
        assert synced[0].start_ms == 2000
        assert synced[0].end_ms   == 6000

    def test_cues_before_zero_are_dropped(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        # offset -8000 ms: entries 1 and 2 end before 0; entry 3 survives
        synced = subtitle_tools.apply_linear_sync(entries, offset_ms=-8000)
        assert len(synced) == 1
        assert synced[0].end_ms == 1000

    def test_start_clamped_to_zero(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        synced  = subtitle_tools.apply_linear_sync(entries, offset_ms=-8000)
        assert synced[0].start_ms == 0

    def test_renumbering(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        synced  = subtitle_tools.apply_linear_sync(entries, offset_ms=-8000)
        assert synced[0].index == 1

    def test_identity_transform(self, sample_srt):
        entries = subtitle_tools.parse_srt(sample_srt)
        synced  = subtitle_tools.apply_linear_sync(entries, offset_ms=0, scale=1.0)
        for orig, new in zip(entries, synced):
            assert orig.start_ms == new.start_ms
            assert orig.end_ms   == new.end_ms


class TestWriteSrt:
    def test_roundtrip(self, sample_srt, tmp_path):
        entries  = subtitle_tools.parse_srt(sample_srt)
        out_path = str(tmp_path / "out.srt")
        subtitle_tools.write_srt(entries, out_path)
        reloaded = subtitle_tools.parse_srt(out_path)
        assert len(reloaded) == len(entries)
        for a, b in zip(entries, reloaded):
            assert a.start_ms == b.start_ms
            assert a.end_ms   == b.end_ms
            assert a.text     == b.text


class TestSyncSubtitleFile:
    def test_returns_cue_count(self, sample_srt, tmp_path):
        out = str(tmp_path / "out.srt")
        n   = subtitle_tools.sync_subtitle_file(sample_srt, out, offset_ms=0)
        assert n == 3

    def test_applies_offset(self, sample_srt, tmp_path):
        out     = str(tmp_path / "out.srt")
        subtitle_tools.sync_subtitle_file(sample_srt, out, offset_ms=1000)
        entries = subtitle_tools.parse_srt(out)
        assert entries[0].start_ms == 2000
