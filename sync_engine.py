"""
sync_engine.py — Linear audio synchronisation engine.

Computes a global (offset_ms, scale) pair from two English-audio WAV files:
  - offset_ms : milliseconds to ADD to HQ timestamps to align with the reference
                (positive → HQ content appears earlier than reference)
  - scale     : multiplicative factor for the HQ time axis
                (>1 → HQ runs faster than reference)

The same parameters are then applied externally to PT-BR audio and subtitles.
No segment-based or non-linear strategies are used here.
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly

# ── tuneable constants ──────────────────────────────────────────────────────
ANALYSIS_RATE = 8000   # Hz  – low rate keeps FFT fast while preserving speech
SEGMENT_SEC   = 60     # seconds of audio used per correlation window


# ── internal helpers ────────────────────────────────────────────────────────

def _load_mono_float(path: str, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Read a WAV file, mix to mono, return (float32 array, sample_rate)."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if max_seconds is not None:
        data = data[: int(rate * max_seconds)]
    return data, rate


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample from src_rate to dst_rate."""
    from math import gcd
    g = gcd(src_rate, dst_rate)
    return resample_poly(data, dst_rate // g, src_rate // g).astype(np.float32)


def _normalize(data: np.ndarray) -> np.ndarray:
    """Scale to unit RMS; return zero array unchanged."""
    rms = float(np.sqrt(np.mean(data ** 2)))
    return data / rms if rms > 1e-6 else data


def _window(data: np.ndarray, rate: int, start_sec: float, dur_sec: float) -> np.ndarray:
    """Extract a time window from an audio array."""
    a = int(rate * start_sec)
    b = int(rate * (start_sec + dur_sec))
    return data[a:b]


def _xcorr_offset_sec(ref: np.ndarray, hq: np.ndarray, rate: int) -> float:
    """
    Compute the offset (seconds) of *hq* relative to *ref* via cross-correlation.

    Positive return value means HQ content appears `offset` seconds *before*
    the same content in the reference, so adding `offset` to HQ timestamps
    aligns them with the reference.

    Formula:
        correlate(ref, hq)[k] = sum_m ref[m+k] * hq[m]
        Peak at lag k  →  ref[m + k] ≈ hq[m]  →  hq is k samples behind ref
                       →  to align: shift hq *forward* by k samples
        offset = (best_k - (len(hq) - 1)) / rate
    """
    corr = correlate(ref, hq, mode="full")
    # In 'full' mode the zero-lag position is at index (len(hq) - 1).
    # Subtracting it converts the absolute argmax index to a signed lag.
    lag = int(np.argmax(corr)) - (len(hq) - 1)
    return lag / rate


# ── public API ───────────────────────────────────────────────────────────────

def compute_sync_params(
    ref_audio_path: str,
    hq_audio_path: str,
) -> tuple[float, float]:
    """
    Compute linear sync parameters (offset_ms, scale) from two WAV files.

    Parameters
    ----------
    ref_audio_path : WAV of the English audio from the *reference* file.
    hq_audio_path  : WAV of the English audio from the *HQ* file.

    Returns
    -------
    offset_ms : float
        Milliseconds to add to HQ timestamps so that HQ content lines up with
        the reference.  Positive → HQ was ahead; negative → HQ was behind.
    scale : float
        Multiplicative factor for the HQ time axis.  1.0 means no drift.
        Applied as:  t_ref = t_hq * scale + offset_ms / 1000
    """
    # Load and normalise sample rates
    ref_data, ref_rate = _load_mono_float(ref_audio_path)
    hq_data,  hq_rate  = _load_mono_float(hq_audio_path)

    if ref_rate != ANALYSIS_RATE:
        ref_data = _resample(ref_data, ref_rate, ANALYSIS_RATE)
    if hq_rate != ANALYSIS_RATE:
        hq_data = _resample(hq_data, hq_rate, ANALYSIS_RATE)

    rate = ANALYSIS_RATE
    total_ref = len(ref_data) / rate
    total_hq  = len(hq_data)  / rate

    # ── start-window offset ──
    win_sec   = min(SEGMENT_SEC, total_ref * 0.4, total_hq * 0.4)
    ref_start = _normalize(_window(ref_data, rate, 0.0, win_sec))
    hq_start  = _normalize(_window(hq_data,  rate, 0.0, win_sec))
    offset_start_sec = _xcorr_offset_sec(ref_start, hq_start, rate)

    # ── scale via end-window drift ──
    scale = 1.0
    min_total = min(total_ref, total_hq)
    if min_total > 2.0 * SEGMENT_SEC:
        end_start_sec = min_total - SEGMENT_SEC
        ref_end = _normalize(_window(ref_data, rate, end_start_sec, SEGMENT_SEC))
        hq_end  = _normalize(_window(hq_data,  rate, end_start_sec, SEGMENT_SEC))
        offset_end_sec = _xcorr_offset_sec(ref_end, hq_end, rate)

        # Linear model:  t_ref = scale * t_hq + offset_start
        #   At t_hq = end_start_sec:
        #     scale * end_start_sec + offset_start = end_start_sec + offset_end
        #   => scale = 1 + (offset_end - offset_start) / end_start_sec
        drift = offset_end_sec - offset_start_sec
        if abs(end_start_sec) > 0.1:
            scale = 1.0 + drift / end_start_sec

    offset_ms = offset_start_sec * 1000.0
    return offset_ms, scale
