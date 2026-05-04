from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

from holodoppler.cli import preview, process

APP_NAME = "HoloDoppler"
SUPPORTED = {".holo", ".cine", ".txt"}
DEFAULT_CONFIG = Path("./src/holodoppler/default_parameters.json")

# ------------------------------------------------------------------
# DND availability check – root window will inherit from TkinterDnD.Tk if present
# ------------------------------------------------------------------
try:
    from tkinterdnd2 import TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    TkinterDnD = tk.Tk  # fallback (no DND)


class UI(TkinterDnD.Tk if DND_AVAILABLE else tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("800x720")
        self.minsize(700, 600)

        self.paths: list[Path] = []
        self.config_path = DEFAULT_CONFIG
        self.config_var = tk.StringVar(value=str(DEFAULT_CONFIG))
        self.status_var = tk.StringVar(value="Ready")

        self.q = queue.Queue()
        self.stop = threading.Event()
        self.worker = None

        self._setup_theme()
        self._build()
        self.after(50, self._poll)
        self._enable_dnd()

    # ------------------------------------------------------------------
    # THEME
    # ------------------------------------------------------------------

    def _setup_theme(self):
        style = ttk.Style(self)
        try:
            self.tk.call("source", "sun-valley.tcl")
            self.tk.call("set_theme", "dark")
        except Exception:
            style.theme_use("clam")
            bg = "#1e1e1e"
            fg = "#f0f0f0"
            style.configure(".", background=bg, foreground=fg)
            style.configure("TLabel", background=bg, foreground=fg)
            style.configure("TFrame", background=bg)
            style.configure("TLabelframe", background=bg, foreground=fg)
            style.configure("TLabelframe.Label", background=bg, foreground=fg)
            style.configure("TButton", background="#2d2d2d", foreground=fg, borderwidth=1)
            style.map("TButton", background=[("active", "#3c3c3c")])
            style.configure("Accent.TButton", background="#0a5c8e", foreground="white")
            style.map("Accent.TButton", background=[("active", "#0f6ba3")])

            # Custom style for readonly Entry (gray background, white text)
            style.configure("Readonly.TEntry",
                            fieldbackground="#3c3c3c",   # gray background
                            foreground="white",
                            borderwidth=1,
                            relief="solid")

        # Use a font that is guaranteed to exist, avoid tuple issues
        default_font = ("TkDefaultFont", 10)
        self.option_add("*Font", default_font)

    # ------------------------------------------------------------------
    # UI BUILD – clean grid layout, centered
    # ------------------------------------------------------------------

    def _build(self):
        main = ttk.Frame(self, padding="15 15 15 15")
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)   # middle column expands
        main.rowconfigure(2, weight=1)       # preview area expands

        # ----- ROW 0: Input files (label, scrollable list, button) -----
        ttk.Label(main, text="Input files", font="TkDefaultFont 10 bold").grid(
            row=0, column=0, sticky="nw", padx=(0, 10), pady=(5, 0)
        )
        files_frame = ttk.Frame(main)
        files_frame.grid(row=0, column=1, sticky="ew", pady=(5, 5))
        files_frame.columnconfigure(0, weight=1)
        self.filelist_text = ScrolledText(
            files_frame, height=5, wrap="word", relief="flat", borderwidth=0,
            bg="#252526", fg="#e0e0e0"
        )
        self.filelist_text.grid(row=0, column=0, sticky="ew")
        self.filelist_text.config(state="disabled")

        ttk.Button(main, text="📂 Open files", command=self.open_files).grid(
            row=0, column=2, padx=(10, 0), pady=(5, 0), sticky="n"
        )

        # ----- ROW 1: Config path (label, readonly entry, button) -----
        ttk.Label(main, text="Config JSON", font="TkDefaultFont 10 bold").grid(
            row=1, column=0, sticky="nw", padx=(0, 10), pady=(10, 0)
        )
        config_frame = ttk.Frame(main)
        config_frame.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        config_frame.columnconfigure(0, weight=1)

        # Use a custom style to give the readonly entry a gray background
        config_entry = ttk.Entry(config_frame, textvariable=self.config_var,
                                 state="readonly", style="Readonly.TEntry")
        config_entry.grid(row=0, column=0, sticky="ew")

        ttk.Button(main, text="⚙️ Change", command=self.choose_config).grid(
            row=1, column=2, padx=(10, 0), pady=(10, 0), sticky="n"
        )

        # ----- ROW 2: Preview area (centered, expands) -----
        preview_frame = ttk.LabelFrame(main, text="Preview", padding=8)
        preview_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=15)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame, text="Drop files or select input", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # ----- ROW 3: Action buttons (centered) -----
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="▶ RUN", command=self.run, state="disabled", style="Accent.TButton")
        self.run_btn.pack(side="left", padx=8)
        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self.stop_work, state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        # ----- ROW 4: Progress & Status -----
        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(5, 5))

        status_frame = ttk.Frame(main)
        status_frame.grid(row=5, column=0, columnspan=3, sticky="ew")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="center", font="TkDefaultFont 9 italic")
        self.status_label.pack(fill="x")

    # ------------------------------------------------------------------
    # INPUT HANDLING
    # ------------------------------------------------------------------

    def open_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("HoloDoppler inputs", "*.holo *.cine *.txt"), ("All files", "*.*")]
        )
        if paths:
            self._set_paths([Path(p) for p in paths])

    def _set_paths(self, paths):
        out = []
        for p in paths:
            if p.suffix.lower() == ".txt":
                out += self._read_txt(p)
            elif p.suffix.lower() in {".holo", ".cine"}:
                out.append(p)

        self.paths = [p.resolve() for p in out if p.exists()]
        self._update_filelist_display()
        self.run_btn["state"] = "normal" if self.paths else "disabled"

        if self.paths:
            self.status_var.set(f"Loaded {len(self.paths)} file(s)")
            self._start_preview(self.paths[0])
        else:
            self.status_var.set("No valid input")
            self.preview_label.config(text="No preview", image="")

    def _update_filelist_display(self):
        self.filelist_text.config(state="normal")
        self.filelist_text.delete(1.0, tk.END)
        if self.paths:
            for p in self.paths:
                self.filelist_text.insert(tk.END, f"{p}\n")
        else:
            self.filelist_text.insert(tk.END, "No files selected")
        self.filelist_text.config(state="disabled")

    def _read_txt(self, path: Path):
        base = path.parent
        res = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().strip('"')
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            res.append(p if p.is_absolute() else base / p)
        return res

    # ------------------------------------------------------------------
    # PREVIEW
    # ------------------------------------------------------------------

    def _start_preview(self, path):
        if self._busy():
            return
        self.stop.clear()
        self._set_busy(True, "Loading preview...")
        self.worker = threading.Thread(target=self._preview_worker, args=(path,), daemon=True)
        self.worker.start()

    def _preview_worker(self, path):
        try:
            params = self._load_params()
            img = preview(str(path), params)
            self.q.put(("img", img))
        except Exception as e:
            self.q.put(("err", str(e)))
        finally:
            self.q.put(("done", None))

    def _show_preview(self, arr):
        if arr is None or Image is None:
            self.preview_label.config(text="Preview not available", image="")
            return

        import numpy as np

        arr = np.asarray(arr).squeeze()

        if arr.dtype == np.uint8:
            img_arr = arr
        else:
            arr = arr.astype("float32")
            mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
            img_arr = ((arr - mn) * 255.0 / (mx - mn)).clip(0, 255).astype("uint8") if mx > mn else np.zeros_like(arr, dtype="uint8")

        img = Image.fromarray(img_arr).convert("RGB")
        max_w, max_h = 680, 320
        w, h = img.size
        scale = min(max_w / w, max_h / h)
        img = img.resize((int(w * scale), int(h * scale)))
        self._img = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._img, text="")

    # ------------------------------------------------------------------
    # PROCESSING
    # ------------------------------------------------------------------

    def run(self):
        if self._busy():
            return
        self.stop.clear()
        self._set_busy(True, "Running...")
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def _run_worker(self):
        try:
            params = self._load_params()
            for i, p in enumerate(self.paths, 1):
                if self.stop.is_set():
                    self.q.put(("status", "Stopped"))
                    return
                self.q.put(("status", f"Processing {i}/{len(self.paths)}"))
                process(str(p), params)
            self.q.put(("status", "Done ヽ(^o^)丿"))
        except Exception as e:
            self.q.put(("err", str(e)))
        finally:
            self.q.put(("done", None))

    def stop_work(self):
        self.stop.set()
        self.status_var.set("Stopping after current file...")

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def choose_config(self):
        p = filedialog.askopenfilename(filetypes=[("JSON config", "*.json"), ("All files", "*.*")])
        if p:
            self.config_path = Path(p)
            self.config_var.set(str(self.config_path))

    def _load_params(self):
        return json.loads(self.config_path.read_text(encoding="utf-8")) if self.config_path.exists() else {}

    # ------------------------------------------------------------------
    # DRAG & DROP (full window, no separate label)
    # ------------------------------------------------------------------

    def _enable_dnd(self):
        if not DND_AVAILABLE:
            self.status_var.set("Drag & drop not available (install tkinterdnd2)")
            return

        try:
            # Register the whole window as a drop target
            self.drop_target_register("DND_Files")
            self.dnd_bind("<<Drop>>", self._on_drop)
            self.status_var.set("Ready – drop .holo/.cine/.txt anywhere")
        except Exception as e:
            print(f"DND registration error: {e}")
            self.status_var.set("Drag & drop failed to initialize")

    def _on_drop(self, e):
        # e.data may contain space-separated paths, possibly with curly braces for spaces.
        # splitlist handles that correctly.
        files = self.tk.splitlist(e.data)
        if files:
            self._set_paths([Path(p) for p in files])

    # ------------------------------------------------------------------
    # QUEUE POLLING & BUSY STATE
    # ------------------------------------------------------------------

    def _poll(self):
        try:
            while True:
                k, v = self.q.get_nowait()
                if k == "img":
                    self._show_preview(v)
                elif k == "err":
                    self.status_var.set("Error")
                    self.preview_label.config(text=v, image="")
                elif k == "status":
                    self.status_var.set(v)
                elif k == "done":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _set_busy(self, busy, status="Ready"):
        if busy:
            self.status_var.set(status)
            self.progress.start(10)
            self.run_btn["state"] = "disabled"
            self.stop_btn["state"] = "normal"
        else:
            self.progress.stop()
            self.run_btn["state"] = "normal" if self.paths else "disabled"
            self.stop_btn["state"] = "disabled"

    def _busy(self):
        return self.worker and self.worker.is_alive()


if __name__ == "__main__":
    UI().mainloop()