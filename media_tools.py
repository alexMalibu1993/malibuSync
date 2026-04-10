"""
media_tools.py — ffmpeg/ffprobe utilities for MalibuSync.

All subprocess calls go through a single _run() helper so that error messages
are always captured and can be surfaced to the caller.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Callable

# ── constants ────────────────────────────────────────────────────────────────

# Offsets below this threshold are ignored (avoids adding unnecessary -itsoffset
# to the ffmpeg command when the drift is below 1 ms).
_MIN_OFFSET_THRESHOLD_SEC = 0.001

# ── internal helper ─────────────────────────────────────────────────────────

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run *cmd*, capture stdout/stderr, optionally raise on non-zero exit."""
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


# ── stream probing ───────────────────────────────────────────────────────────

def probe_streams(file_path: str) -> list[dict]:
    """
    Return the stream list reported by ffprobe for *file_path*.

    Each dict contains at minimum the keys returned by ffprobe JSON output:
    ``index``, ``codec_type``, ``codec_name``, ``tags`` (may be absent).
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        file_path,
    ]
    result = _run(cmd)
    return json.loads(result.stdout).get("streams", [])


def find_stream(
    streams: list[dict],
    codec_type: str,
    language: str | None = None,
) -> int | None:
    """
    Return the stream index matching *codec_type* (and optional *language* tag).

    Falls back to the first stream of the given type when no language match is
    found.  Returns *None* when no stream of *codec_type* exists at all.
    """
    fallback: int | None = None
    for s in streams:
        if s.get("codec_type") != codec_type:
            continue
        if fallback is None:
            fallback = s["index"]
        if language:
            lang = s.get("tags", {}).get("language", "").lower()
            if lang == language.lower():
                return s["index"]
    return fallback


# ── extraction helpers ───────────────────────────────────────────────────────

def extract_audio_stream(
    file_path: str,
    stream_index: int,
    out_wav: str,
    max_seconds: float | None = None,
) -> None:
    """
    Extract *stream_index* from *file_path* to a mono 16 kHz WAV file.

    16 kHz mono is sufficient for cross-correlation analysis and keeps the
    temporary file small.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", file_path,
        "-map", f"0:{stream_index}",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
    ]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd.append(out_wav)
    _run(cmd)


def extract_subtitle_stream(
    file_path: str,
    stream_index: int,
    out_path: str,
) -> bool:
    """
    Extract a subtitle stream to *out_path* (SRT format preferred).

    Returns *True* on success, *False* when extraction fails entirely.
    Two attempts are made:
      1. Explicit ``-f srt`` (works for most text-based formats).
      2. No format flag — lets ffmpeg infer from the output extension.
    """
    for extra in (["-f", "srt"], []):
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", file_path,
            "-map", f"0:{stream_index}",
            *extra,
            out_path,
        ]
        result = _run(cmd, check=False)
        if result.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return True
    return False


# ── batch subtitle extraction ────────────────────────────────────────────────

_PTBR_LANG_PRIORITY = ("por", "pt", "pt-br")
_MEDIA_EXTENSIONS   = (".mkv", ".mp4", ".avi", ".ts", ".m2ts")


def batch_extract_ptbr_subtitles(
    input_dir: str,
    output_dir: str,
    extensions: tuple[str, ...] = _MEDIA_EXTENSIONS,
    log_fn: Callable[[str], None] | None = None,
) -> list[str]:
    """
    Extract PT-BR subtitle streams from every media file in *input_dir*.

    Fallback strategy per file:
      1. Stream whose ``language`` tag is ``por``.
      2. Stream whose ``language`` tag is ``pt``.
      3. Stream whose ``language`` tag is ``pt-br``.
      4. First subtitle stream found (any language).

    Returns the list of successfully created subtitle files.
    """
    os.makedirs(output_dir, exist_ok=True)
    created: list[str] = []

    for fname in sorted(os.listdir(input_dir)):
        if not any(fname.lower().endswith(ext) for ext in extensions):
            continue

        src  = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        out  = os.path.join(output_dir, base + ".srt")

        streams     = probe_streams(src)
        sub_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

        if not sub_streams:
            if log_fn:
                log_fn(f"[SKIP] {fname} — no subtitle streams found")
            continue

        # Build candidate list in fallback order
        candidates: list[int] = []
        for lang_tag in _PTBR_LANG_PRIORITY:
            for s in sub_streams:
                if s.get("tags", {}).get("language", "").lower() == lang_tag:
                    candidates.append(s["index"])
                    break  # one match per language tag is enough

        if not candidates:
            # No PT-BR tag found — use the first available subtitle stream
            candidates.append(sub_streams[0]["index"])

        success = False
        for idx in candidates:
            if extract_subtitle_stream(src, idx, out):
                success = True
                break

        if success:
            if log_fn:
                log_fn(f"[OK]   {fname} → {os.path.basename(out)}")
            created.append(out)
        else:
            if log_fn:
                log_fn(f"[FAIL] {fname} — could not extract any subtitle stream")

    return created


# ── remux / export ───────────────────────────────────────────────────────────

def remux_with_sync(
    hq_file: str,
    video_stream_index: int,
    ptbr_audio_index: int,
    subtitle_srt_path: str | None,
    output_path: str,
    audio_offset_ms: float,
) -> None:
    """
    Remux *hq_file* applying a time offset to the PT-BR audio stream.

    Video is always copied without re-encoding.  The PT-BR audio is
    time-shifted using ffmpeg's ``-itsoffset``; no audio re-encoding takes
    place (remux-only).  The subtitle, if provided, must already be
    pre-synced externally (see :func:`subtitle_tools.sync_subtitle_file`).

    Parameters
    ----------
    hq_file             : Source media file.
    video_stream_index  : Index of the video stream to copy.
    ptbr_audio_index    : Index of the PT-BR audio stream to time-shift.
    subtitle_srt_path   : Path to an already-synced SRT file, or *None*.
    output_path         : Destination file.
    audio_offset_ms     : Milliseconds to shift the PT-BR audio
                          (positive → delay; negative → advance).
    """
    offset_sec = audio_offset_ms / 1000.0

    # Build inputs: the HQ file is read twice so each copy can have its own
    # -itsoffset without affecting the other streams.
    cmd = ["ffmpeg", "-y", "-v", "quiet"]

    # Input 0 — video (no offset)
    cmd += ["-i", hq_file]

    # Input 1 — PT-BR audio (with offset)
    if abs(offset_sec) > _MIN_OFFSET_THRESHOLD_SEC:
        cmd += ["-itsoffset", f"{offset_sec:.6f}"]
    cmd += ["-i", hq_file]

    input_idx = 2

    if subtitle_srt_path:
        cmd += ["-i", subtitle_srt_path]
        sub_input = input_idx
        input_idx += 1

    # Map streams
    cmd += ["-map", f"0:{video_stream_index}"]
    cmd += ["-map", f"1:{ptbr_audio_index}"]
    if subtitle_srt_path:
        cmd += ["-map", f"{sub_input}:0"]

    # Copy all streams (remux-only)
    cmd += ["-c", "copy"]
    if subtitle_srt_path:
        cmd += ["-c:s", "srt"]

    cmd.append(output_path)
    _run(cmd)


def apply_full_sync(
    hq_file: str,
    video_stream_index: int,
    ptbr_audio_index: int,
    subtitle_srt_path: str | None,
    output_path: str,
    audio_offset_ms: float,
    scale: float,
    tmp_audio: str,
) -> None:
    """
    Full sync: re-encode PT-BR audio with offset + speed correction, then mux.

    The ``atempo`` filter adjusts playback speed.  For *scale* values outside
    [0.5, 2.0] the filter is chained automatically.

    Parameters
    ----------
    tmp_audio : Temporary path for the re-encoded audio (e.g. ``/tmp/ptbr.aac``).
    """
    # ── Step 1: re-encode PT-BR audio with atempo + adelay ──────────────────
    filters: list[str] = []

    # atempo only accepts [0.5, 2.0]; chain if necessary
    remaining = scale
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")

    # adelay adds milliseconds of silence at the front; the N|N syntax applies
    # the same delay to all channels (works for both mono and stereo).
    if audio_offset_ms >= 0:
        filters.append(f"adelay={audio_offset_ms:.0f}|{audio_offset_ms:.0f}")
        trim_start = None
    else:
        # Advance: trim the leading silence that would result from a negative delay
        trim_start = abs(audio_offset_ms) / 1000.0
        filters.append(f"atrim=start={trim_start:.6f},asetpts=PTS-STARTPTS")

    af = ",".join(filters)

    cmd_encode = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", hq_file,
        "-map", f"0:{ptbr_audio_index}",
        "-af", af,
        tmp_audio,
    ]
    _run(cmd_encode)

    # ── Step 2: mux video + synced audio (+ subtitle) ───────────────────────
    cmd_mux = ["ffmpeg", "-y", "-v", "quiet",
               "-i", hq_file,
               "-i", tmp_audio]

    input_idx = 2
    if subtitle_srt_path:
        cmd_mux += ["-i", subtitle_srt_path]
        sub_input = input_idx
        input_idx += 1

    cmd_mux += ["-map", f"0:{video_stream_index}"]
    cmd_mux += ["-map", "1:0"]
    if subtitle_srt_path:
        cmd_mux += ["-map", f"{sub_input}:0", "-c:s", "srt"]

    cmd_mux += ["-c:v", "copy", "-c:a", "copy"]
    cmd_mux.append(output_path)
    _run(cmd_mux)
