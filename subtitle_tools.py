"""
subtitle_tools.py — SRT subtitle parsing and linear time synchronisation.

Applies a global linear transformation  t_out = t_in * scale + offset_ms
to every cue in an SRT file.  No segment-based or non-linear strategies
are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class SrtEntry:
    """A single SRT subtitle cue."""
    index: int
    start_ms: int
    end_ms: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


# ── timestamp helpers ────────────────────────────────────────────────────────

_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _ts_to_ms(ts: str) -> int:
    """Convert ``HH:MM:SS,mmm`` or ``HH:MM:SS.mmm`` to milliseconds."""
    m = _TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Unrecognised timestamp: {ts!r}")
    h, mn, s, ms = (int(x) for x in m.groups())
    return ((h * 3600 + mn * 60 + s) * 1000) + ms


def _ms_to_ts(ms: int) -> str:
    """Convert milliseconds to ``HH:MM:SS,mmm``."""
    ms = max(0, ms)
    h   = ms // 3_600_000;  ms %= 3_600_000
    mn  = ms // 60_000;     ms %= 60_000
    s   = ms // 1_000;      ms %= 1_000
    return f"{h:02d}:{mn:02d}:{s:02d},{ms:03d}"


# ── parsing ──────────────────────────────────────────────────────────────────

_ARROW_RE = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}[,.]\d{3})")


def parse_srt(path: str) -> list[SrtEntry]:
    """
    Parse an SRT file and return a list of :class:`SrtEntry` objects.

    Tolerates Windows line endings and BOM markers.  Blank lines between
    cues are used as separators; malformed cues are skipped silently.
    """
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        content = fh.read()

    entries: list[SrtEntry] = []
    blocks = re.split(r"\r?\n\r?\n", content.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines_raw = re.split(r"\r?\n", block)

        # First non-empty line should be the cue index
        idx_line = lines_raw[0].strip() if lines_raw else ""
        if not idx_line.isdigit():
            continue
        cue_index = int(idx_line)

        if len(lines_raw) < 2:
            continue

        m = _ARROW_RE.match(lines_raw[1].strip())
        if not m:
            continue

        start_ms = _ts_to_ms(m.group(1))
        end_ms   = _ts_to_ms(m.group(2))
        text     = lines_raw[2:] if len(lines_raw) > 2 else []

        entries.append(SrtEntry(
            index    = cue_index,
            start_ms = start_ms,
            end_ms   = end_ms,
            lines    = text,
        ))

    return entries


# ── sync ─────────────────────────────────────────────────────────────────────

def apply_linear_sync(
    entries: list[SrtEntry],
    offset_ms: float,
    scale: float = 1.0,
) -> list[SrtEntry]:
    """
    Apply a linear time transformation to every cue:

        t_out = t_in * scale + offset_ms

    Cues whose *end* time would fall at or before zero are dropped.
    The cue index sequence is re-numbered from 1.
    """
    result: list[SrtEntry] = []
    new_idx = 1
    for e in entries:
        new_start = int(round(e.start_ms * scale + offset_ms))
        new_end   = int(round(e.end_ms   * scale + offset_ms))
        if new_end <= 0:
            continue
        new_start = max(0, new_start)
        result.append(SrtEntry(
            index    = new_idx,
            start_ms = new_start,
            end_ms   = new_end,
            lines    = list(e.lines),
        ))
        new_idx += 1
    return result


# ── writing ──────────────────────────────────────────────────────────────────

def write_srt(entries: list[SrtEntry], path: str) -> None:
    """Write *entries* to *path* in standard SRT format (UTF-8)."""
    with open(path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(f"{e.index}\n")
            fh.write(f"{_ms_to_ts(e.start_ms)} --> {_ms_to_ts(e.end_ms)}\n")
            fh.write(e.text)
            fh.write("\n\n")


# ── convenience wrapper ───────────────────────────────────────────────────────

def sync_subtitle_file(
    input_path: str,
    output_path: str,
    offset_ms: float,
    scale: float = 1.0,
) -> int:
    """
    Read *input_path*, apply the linear sync, write *output_path*.

    Returns the number of subtitle cues written.
    """
    entries = parse_srt(input_path)
    synced  = apply_linear_sync(entries, offset_ms=offset_ms, scale=scale)
    write_srt(synced, output_path)
    return len(synced)
