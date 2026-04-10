#!/usr/bin/env python3
"""
malibu_sync.py — MalibuSync: Audio & Subtitle Synchroniser (PT-BR)

Tkinter GUI that:
  • Detects linear sync parameters (offset + scale) from English audio tracks.
  • Applies the same parameters to PT-BR audio and subtitles.
  • Supports remux-only mode (no video/audio re-encoding) and full-sync mode.
  • Provides a batch PT-BR subtitle extractor with language-tag fallbacks.
"""

from __future__ import annotations

import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import media_tools
import subtitle_tools
import sync_engine


# ── helpers ──────────────────────────────────────────────────────────────────

def _ask_file(title: str, filetypes: list[tuple[str, str]], var: tk.StringVar) -> None:
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if path:
        var.set(path)


def _ask_dir(title: str, var: tk.StringVar) -> None:
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def _ask_save(title: str, filetypes: list[tuple[str, str]], var: tk.StringVar) -> None:
    path = filedialog.asksaveasfilename(title=title, filetypes=filetypes)
    if path:
        var.set(path)


# ── stream helpers ────────────────────────────────────────────────────────────

def _stream_label(s: dict) -> str:
    """Return a human-readable label for a media stream."""
    idx   = s.get("index", "?")
    ctype = s.get("codec_type", "?")
    codec = s.get("codec_name", "?")
    lang  = s.get("tags", {}).get("language", "")
    title = s.get("tags", {}).get("title", "")
    parts = [f"#{idx}", codec.upper()]
    if lang:
        parts.append(lang.upper())
    if title:
        parts.append(f'"{title}"')
    return f"[{ctype.upper()}] " + " ".join(parts)


def _streams_of_type(streams: list[dict], codec_type: str) -> list[dict]:
    return [s for s in streams if s.get("codec_type") == codec_type]


# ── main application ──────────────────────────────────────────────────────────

class MalibuSyncApp(tk.Tk):
    """Root window for MalibuSync."""

    def __init__(self) -> None:
        super().__init__()
        self.title("MalibuSync — Sincronizador de Áudio e Legendas PT-BR")
        self.resizable(True, True)
        self.minsize(620, 520)

        # ── shared state ──
        self._hq_streams:  list[dict] = []
        self._ref_streams: list[dict] = []

        self._offset_ms = tk.DoubleVar(value=0.0)
        self._scale     = tk.DoubleVar(value=1.0)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        sync_frame  = ttk.Frame(notebook)
        batch_frame = ttk.Frame(notebook)
        notebook.add(sync_frame,  text="  Sincronizar  ")
        notebook.add(batch_frame, text="  Extração em Lote  ")

        self._build_sync_tab(sync_frame)
        self._build_batch_tab(batch_frame)

    # ────────────────────────────────────────────────────────────────────────
    # Sync tab
    # ────────────────────────────────────────────────────────────────────────

    def _build_sync_tab(self, parent: ttk.Frame) -> None:
        pad = {"padx": 6, "pady": 3}

        # ── File inputs ──────────────────────────────────────────────────────
        files_frame = ttk.LabelFrame(parent, text="Arquivos")
        files_frame.pack(fill="x", **pad)
        files_frame.columnconfigure(1, weight=1)

        self._hq_path  = tk.StringVar()
        self._ref_path = tk.StringVar()

        for row, (label, var, cmd) in enumerate([
            ("Arquivo HQ:",         self._hq_path,  self._load_hq_file),
            ("Arquivo Referência:", self._ref_path, self._load_ref_file),
        ]):
            ttk.Label(files_frame, text=label).grid(row=row, column=0, sticky="e", **pad)
            ttk.Entry(files_frame, textvariable=var, width=50).grid(
                row=row, column=1, sticky="ew", **pad
            )
            ttk.Button(files_frame, text="…", width=3, command=cmd).grid(
                row=row, column=2, **pad
            )

        # ── Stream selection ──────────────────────────────────────────────────
        streams_frame = ttk.LabelFrame(parent, text="Seleção de Streams")
        streams_frame.pack(fill="x", **pad)
        streams_frame.columnconfigure(1, weight=1)

        self._hq_en_audio_var   = tk.StringVar()
        self._ref_en_audio_var  = tk.StringVar()
        self._hq_ptbr_audio_var = tk.StringVar()
        self._hq_video_var      = tk.StringVar()
        self._hq_sub_var        = tk.StringVar()

        self._hq_en_audio_cb   = ttk.Combobox(streams_frame, textvariable=self._hq_en_audio_var,  state="readonly", width=48)
        self._ref_en_audio_cb  = ttk.Combobox(streams_frame, textvariable=self._ref_en_audio_var, state="readonly", width=48)
        self._hq_ptbr_audio_cb = ttk.Combobox(streams_frame, textvariable=self._hq_ptbr_audio_var, state="readonly", width=48)
        self._hq_video_cb      = ttk.Combobox(streams_frame, textvariable=self._hq_video_var,      state="readonly", width=48)
        self._hq_sub_cb        = ttk.Combobox(streams_frame, textvariable=self._hq_sub_var,        state="readonly", width=48)

        stream_rows = [
            ("Áudio EN (HQ):",        self._hq_en_audio_cb),
            ("Áudio EN (Referência):", self._ref_en_audio_cb),
            ("Áudio PT-BR (HQ):",     self._hq_ptbr_audio_cb),
            ("Vídeo (HQ):",           self._hq_video_cb),
            ("Legenda (HQ):",         self._hq_sub_cb),
        ]
        for row, (lbl, cb) in enumerate(stream_rows):
            ttk.Label(streams_frame, text=lbl).grid(row=row, column=0, sticky="e", **pad)
            cb.grid(row=row, column=1, sticky="ew", **pad)

        # ── External subtitle (optional standalone sync) ─────────────────────
        ext_sub_frame = ttk.LabelFrame(parent, text="Legenda Avulsa (opcional)")
        ext_sub_frame.pack(fill="x", **pad)
        ext_sub_frame.columnconfigure(1, weight=1)

        self._ext_sub_path = tk.StringVar()
        ttk.Label(ext_sub_frame, text="Arquivo SRT:").grid(row=0, column=0, sticky="e", **pad)
        ttk.Entry(ext_sub_frame, textvariable=self._ext_sub_path, width=50).grid(
            row=0, column=1, sticky="ew", **pad
        )
        ttk.Button(
            ext_sub_frame, text="…", width=3,
            command=lambda: _ask_file(
                "Selecionar legenda SRT",
                [("SRT", "*.srt"), ("Todos", "*.*")],
                self._ext_sub_path,
            ),
        ).grid(row=0, column=2, **pad)

        ttk.Button(
            ext_sub_frame, text="Sincronizar Legenda Avulsa",
            command=self._sync_standalone_subtitle,
        ).grid(row=1, column=0, columnspan=3, **pad)

        # ── Sync parameters ──────────────────────────────────────────────────
        params_frame = ttk.LabelFrame(parent, text="Parâmetros de Sync")
        params_frame.pack(fill="x", **pad)

        ttk.Button(
            params_frame, text="Calcular Sync (EN Audio)",
            command=self._calculate_sync,
        ).grid(row=0, column=0, columnspan=4, **pad)

        for col, (lbl, var, fmt) in enumerate([
            ("Offset (ms):", self._offset_ms, "%.1f"),
            ("Escala:",      self._scale,     "%.6f"),
        ]):
            ttk.Label(params_frame, text=lbl).grid(row=1, column=col * 2,     sticky="e", **pad)
            ttk.Label(params_frame, textvariable=var).grid(
                row=1, column=col * 2 + 1, sticky="w", **pad
            )

        # ── Output ───────────────────────────────────────────────────────────
        output_frame = ttk.LabelFrame(parent, text="Saída")
        output_frame.pack(fill="x", **pad)
        output_frame.columnconfigure(1, weight=1)

        self._output_path = tk.StringVar()
        self._remux_only  = tk.BooleanVar(value=True)

        ttk.Label(output_frame, text="Arquivo de saída:").grid(row=0, column=0, sticky="e", **pad)
        ttk.Entry(output_frame, textvariable=self._output_path, width=50).grid(
            row=0, column=1, sticky="ew", **pad
        )
        ttk.Button(
            output_frame, text="…", width=3,
            command=lambda: _ask_save(
                "Salvar arquivo de saída",
                [("MKV", "*.mkv"), ("MP4", "*.mp4"), ("Todos", "*.*")],
                self._output_path,
            ),
        ).grid(row=0, column=2, **pad)

        mode_frame = ttk.Frame(output_frame)
        mode_frame.grid(row=1, column=0, columnspan=3, sticky="w", **pad)
        ttk.Radiobutton(mode_frame, text="Remux Only (sem re-encode de áudio)",
                        variable=self._remux_only, value=True).pack(side="left")
        ttk.Radiobutton(mode_frame, text="Full Sync (re-encode + scale)",
                        variable=self._remux_only, value=False).pack(side="left", padx=12)

        ttk.Button(
            output_frame, text="Aplicar Sync e Exportar",
            command=self._apply_sync,
        ).grid(row=2, column=0, columnspan=3, **pad)

        # ── Log ──────────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)

        self._sync_log = tk.Text(log_frame, height=6, state="disabled", wrap="word")
        scroll = ttk.Scrollbar(log_frame, command=self._sync_log.yview)
        self._sync_log.configure(yscrollcommand=scroll.set)
        self._sync_log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ────────────────────────────────────────────────────────────────────────
    # Batch extraction tab
    # ────────────────────────────────────────────────────────────────────────

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        pad = {"padx": 6, "pady": 3}

        dirs_frame = ttk.LabelFrame(parent, text="Pastas")
        dirs_frame.pack(fill="x", **pad)
        dirs_frame.columnconfigure(1, weight=1)

        self._batch_in_dir  = tk.StringVar()
        self._batch_out_dir = tk.StringVar()

        for row, (lbl, var, title) in enumerate([
            ("Pasta de entrada:", self._batch_in_dir,  "Selecionar pasta de entrada"),
            ("Pasta de saída:",   self._batch_out_dir, "Selecionar pasta de saída"),
        ]):
            ttk.Label(dirs_frame, text=lbl).grid(row=row, column=0, sticky="e", **pad)
            ttk.Entry(dirs_frame, textvariable=var, width=50).grid(
                row=row, column=1, sticky="ew", **pad
            )
            ttk.Button(
                dirs_frame, text="…", width=3,
                command=lambda v=var, t=title: _ask_dir(t, v),
            ).grid(row=row, column=2, **pad)

        ttk.Button(
            dirs_frame, text="Extrair Legendas PT-BR em Lote",
            command=self._run_batch_extract,
        ).grid(row=2, column=0, columnspan=3, **pad)

        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)

        self._batch_log = tk.Text(log_frame, height=12, state="disabled", wrap="word")
        scroll = ttk.Scrollbar(log_frame, command=self._batch_log.yview)
        self._batch_log.configure(yscrollcommand=scroll.set)
        self._batch_log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ────────────────────────────────────────────────────────────────────────
    # File loading / stream population
    # ────────────────────────────────────────────────────────────────────────

    def _load_hq_file(self) -> None:
        _ask_file(
            "Selecionar arquivo HQ",
            [("Vídeo", "*.mkv *.mp4 *.avi *.ts"), ("Todos", "*.*")],
            self._hq_path,
        )
        path = self._hq_path.get()
        if not path:
            return
        try:
            self._hq_streams = media_tools.probe_streams(path)
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao ler streams do arquivo HQ:\n{exc}")
            return
        self._populate_hq_dropdowns()

    def _load_ref_file(self) -> None:
        _ask_file(
            "Selecionar arquivo de referência",
            [("Vídeo", "*.mkv *.mp4 *.avi *.ts"), ("Todos", "*.*")],
            self._ref_path,
        )
        path = self._ref_path.get()
        if not path:
            return
        try:
            self._ref_streams = media_tools.probe_streams(path)
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao ler streams do arquivo de referência:\n{exc}")
            return
        self._populate_ref_dropdowns()

    def _populate_hq_dropdowns(self) -> None:
        streams = self._hq_streams

        audio_streams = _streams_of_type(streams, "audio")
        video_streams = _streams_of_type(streams, "video")
        sub_streams   = _streams_of_type(streams, "subtitle")

        audio_labels = [_stream_label(s) for s in audio_streams]
        video_labels = [_stream_label(s) for s in video_streams]
        sub_labels   = [_stream_label(s) for s in sub_streams]

        self._hq_en_audio_cb["values"]   = audio_labels
        self._hq_ptbr_audio_cb["values"] = audio_labels
        self._hq_video_cb["values"]      = video_labels
        self._hq_sub_cb["values"]        = sub_labels

        # Auto-select: EN audio → first stream with 'eng'/'en' language tag
        en_idx   = media_tools.find_stream(streams, "audio", "eng")
        ptbr_idx = media_tools.find_stream(streams, "audio", "por")
        vid_idx  = media_tools.find_stream(streams, "video")
        sub_idx  = media_tools.find_stream(streams, "subtitle", "por")

        def _pick(candidates: list[dict], preferred_idx: int | None) -> str:
            if preferred_idx is None or not candidates:
                return candidates[0] if candidates else ""
            # Find label for preferred index
            for s, lbl in zip(candidates, [_stream_label(c) for c in candidates]):
                if s["index"] == preferred_idx:
                    return lbl
            return _stream_label(candidates[0]) if candidates else ""

        if audio_labels:
            self._hq_en_audio_var.set(  _pick(audio_streams, en_idx))
            self._hq_ptbr_audio_var.set(_pick(audio_streams, ptbr_idx or en_idx))
        if video_labels:
            self._hq_video_var.set(video_labels[0])
        if sub_labels:
            self._hq_sub_var.set(_pick(sub_streams, sub_idx))

    def _populate_ref_dropdowns(self) -> None:
        streams = self._ref_streams
        audio_streams = _streams_of_type(streams, "audio")
        audio_labels  = [_stream_label(s) for s in audio_streams]

        self._ref_en_audio_cb["values"] = audio_labels
        en_idx = media_tools.find_stream(streams, "audio", "eng")

        def _pick(candidates: list[dict], preferred_idx: int | None) -> str:
            if preferred_idx is None or not candidates:
                return candidates[0] if candidates else ""
            for s, lbl in zip(candidates, [_stream_label(c) for c in candidates]):
                if s["index"] == preferred_idx:
                    return lbl
            return _stream_label(candidates[0]) if candidates else ""

        if audio_labels:
            self._ref_en_audio_var.set(_pick(audio_streams, en_idx))

    # ────────────────────────────────────────────────────────────────────────
    # Sync calculation
    # ────────────────────────────────────────────────────────────────────────

    def _resolve_stream_index(self, label: str, streams: list[dict]) -> int | None:
        """Return the stream index for a combobox label."""
        for s in streams:
            if _stream_label(s) == label:
                return s["index"]
        return None

    def _calculate_sync(self) -> None:
        hq_path  = self._hq_path.get()
        ref_path = self._ref_path.get()

        if not hq_path or not ref_path:
            messagebox.showwarning("Atenção", "Selecione os arquivos HQ e de Referência primeiro.")
            return

        hq_en_label  = self._hq_en_audio_var.get()
        ref_en_label = self._ref_en_audio_var.get()

        if not hq_en_label or not ref_en_label:
            messagebox.showwarning("Atenção", "Selecione os streams de áudio EN para ambos os arquivos.")
            return

        hq_en_idx  = self._resolve_stream_index(hq_en_label,  self._hq_streams)
        ref_en_idx = self._resolve_stream_index(ref_en_label, self._ref_streams)

        if hq_en_idx is None or ref_en_idx is None:
            messagebox.showerror("Erro", "Stream de áudio EN não identificado.")
            return

        self._log_sync("Calculando sync… (isso pode levar alguns segundos)")
        self.update_idletasks()

        def _work() -> None:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    hq_wav  = os.path.join(tmpdir, "hq_en.wav")
                    ref_wav = os.path.join(tmpdir, "ref_en.wav")

                    self._log_sync("  Extraindo áudio EN do arquivo HQ…")
                    media_tools.extract_audio_stream(hq_path,  hq_en_idx,  hq_wav)

                    self._log_sync("  Extraindo áudio EN do arquivo de Referência…")
                    media_tools.extract_audio_stream(ref_path, ref_en_idx, ref_wav)

                    self._log_sync("  Computando correlação…")
                    offset_ms, scale = sync_engine.compute_sync_params(ref_wav, hq_wav)

                self.after(0, lambda: self._on_sync_done(offset_ms, scale))
            except Exception as exc:
                self.after(0, lambda: self._on_sync_error(str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    def _on_sync_done(self, offset_ms: float, scale: float) -> None:
        self._offset_ms.set(round(offset_ms, 1))
        self._scale.set(round(scale, 6))
        self._log_sync(f"✔ Sync calculado: offset={offset_ms:.1f} ms  escala={scale:.6f}")

    def _on_sync_error(self, msg: str) -> None:
        self._log_sync(f"✘ Erro ao calcular sync: {msg}")
        messagebox.showerror("Erro de Sync", msg)

    # ────────────────────────────────────────────────────────────────────────
    # Apply sync + export
    # ────────────────────────────────────────────────────────────────────────

    def _apply_sync(self) -> None:
        hq_path     = self._hq_path.get()
        output_path = self._output_path.get()
        offset_ms   = self._offset_ms.get()
        scale       = self._scale.get()
        remux_only  = self._remux_only.get()

        if not hq_path:
            messagebox.showwarning("Atenção", "Selecione o arquivo HQ.")
            return
        if not output_path:
            messagebox.showwarning("Atenção", "Selecione o arquivo de saída.")
            return

        video_label    = self._hq_video_var.get()
        ptbr_aud_label = self._hq_ptbr_audio_var.get()
        sub_label      = self._hq_sub_var.get()

        video_idx    = self._resolve_stream_index(video_label,    self._hq_streams)
        ptbr_aud_idx = self._resolve_stream_index(ptbr_aud_label, self._hq_streams)
        sub_idx      = self._resolve_stream_index(sub_label,      self._hq_streams) if sub_label else None

        if video_idx is None:
            messagebox.showerror("Erro", "Stream de vídeo não identificado.")
            return
        if ptbr_aud_idx is None:
            messagebox.showerror("Erro", "Stream de áudio PT-BR não identificado.")
            return

        self._log_sync("Aplicando sync e exportando…")
        self.update_idletasks()

        def _work() -> None:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    synced_srt: str | None = None

                    if sub_idx is not None:
                        # Extract embedded subtitle and apply sync
                        raw_srt    = os.path.join(tmpdir, "raw.srt")
                        synced_srt = os.path.join(tmpdir, "synced.srt")
                        self._log_sync("  Extraindo e sincronizando legenda…")
                        ok = media_tools.extract_subtitle_stream(hq_path, sub_idx, raw_srt)
                        if ok:
                            n = subtitle_tools.sync_subtitle_file(
                                raw_srt, synced_srt, offset_ms=offset_ms, scale=scale
                            )
                            self._log_sync(f"  {n} cues sincronizados.")
                        else:
                            self._log_sync("  ⚠ Falha ao extrair legenda — exportando sem legenda.")
                            synced_srt = None

                    if remux_only:
                        self._log_sync("  Modo Remux Only: aplicando offset ao áudio…")
                        media_tools.remux_with_sync(
                            hq_file             = hq_path,
                            video_stream_index  = video_idx,
                            ptbr_audio_index    = ptbr_aud_idx,
                            subtitle_srt_path   = synced_srt,
                            output_path         = output_path,
                            audio_offset_ms     = offset_ms,
                        )
                    else:
                        tmp_audio = os.path.join(tmpdir, "ptbr_synced.aac")
                        self._log_sync("  Modo Full Sync: re-codificando áudio PT-BR…")
                        media_tools.apply_full_sync(
                            hq_file             = hq_path,
                            video_stream_index  = video_idx,
                            ptbr_audio_index    = ptbr_aud_idx,
                            subtitle_srt_path   = synced_srt,
                            output_path         = output_path,
                            audio_offset_ms     = offset_ms,
                            scale               = scale,
                            tmp_audio           = tmp_audio,
                        )

                self.after(0, lambda: self._log_sync(f"✔ Exportado: {output_path}"))
            except Exception as exc:
                self.after(0, lambda: self._on_sync_error(str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────────
    # Standalone subtitle sync
    # ────────────────────────────────────────────────────────────────────────

    def _sync_standalone_subtitle(self) -> None:
        """Apply current offset+scale to an external SRT file."""
        in_path = self._ext_sub_path.get()
        if not in_path:
            messagebox.showwarning("Atenção", "Selecione um arquivo SRT avulso primeiro.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Salvar legenda sincronizada",
            defaultextension=".srt",
            filetypes=[("SRT", "*.srt"), ("Todos", "*.*")],
        )
        if not out_path:
            return

        offset_ms = self._offset_ms.get()
        scale     = self._scale.get()

        try:
            n = subtitle_tools.sync_subtitle_file(
                in_path, out_path, offset_ms=offset_ms, scale=scale
            )
            self._log_sync(f"✔ Legenda avulsa sincronizada: {n} cues → {out_path}")
        except Exception as exc:
            self._log_sync(f"✘ Erro ao sincronizar legenda avulsa: {exc}")
            messagebox.showerror("Erro", str(exc))

    # ────────────────────────────────────────────────────────────────────────
    # Batch extraction
    # ────────────────────────────────────────────────────────────────────────

    def _run_batch_extract(self) -> None:
        in_dir  = self._batch_in_dir.get()
        out_dir = self._batch_out_dir.get()

        if not in_dir or not out_dir:
            messagebox.showwarning("Atenção", "Selecione as pastas de entrada e saída.")
            return
        if not os.path.isdir(in_dir):
            messagebox.showerror("Erro", "Pasta de entrada não encontrada.")
            return

        self._log_batch("Iniciando extração em lote…")
        self.update_idletasks()

        def _work() -> None:
            created = media_tools.batch_extract_ptbr_subtitles(
                input_dir  = in_dir,
                output_dir = out_dir,
                log_fn     = lambda msg: self.after(0, lambda m=msg: self._log_batch(m)),
            )
            self.after(
                0,
                lambda: self._log_batch(
                    f"\n✔ Concluído — {len(created)} legenda(s) extraída(s)."
                ),
            )

        threading.Thread(target=_work, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────────
    # Logging helpers
    # ────────────────────────────────────────────────────────────────────────

    def _log(self, widget: tk.Text, message: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", message + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _log_sync(self, message: str) -> None:
        self._log(self._sync_log, message)

    def _log_batch(self, message: str) -> None:
        self._log(self._batch_log, message)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = MalibuSyncApp()
    app.mainloop()


if __name__ == "__main__":
    main()
