"""
CLAHE YOLO Dataset Transformer
================================
Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to images
in a YOLO-format dataset. Labels are preserved untouched.

Usage:
    python clahe_yolo_transform.py
    
    A UI will appear to let you choose the dataset folder and tune parameters.
"""

import os
import sys
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install opencv-python-headless numpy")
    import cv2
    import numpy as np


# ─────────────────────────────────────────────
#  CLAHE Core Logic
# ─────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def apply_clahe(img_bgr: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """Apply CLAHE to a BGR image (works in LAB color space)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def find_image_dirs(dataset_root: Path) -> list[Path]:
    """Return all subdirectories (or root) that contain images."""
    dirs = set()
    for ext in SUPPORTED_EXTS:
        for p in dataset_root.rglob(f"*{ext}"):
            dirs.add(p.parent)
    return sorted(dirs)


def process_dataset(
    dataset_root: Path,
    output_root: Path,
    clip_limit: float,
    tile_size: int,
    copy_labels: bool,
    progress_cb,
    log_cb,
    done_cb,
):
    """Run CLAHE on all images; copy labels unchanged."""
    image_paths = []
    for ext in SUPPORTED_EXTS:
        image_paths.extend(dataset_root.rglob(f"*{ext}"))
    image_paths = sorted(image_paths)

    total = len(image_paths)
    if total == 0:
        log_cb("⚠  No images found in the selected directory.")
        done_cb(False)
        return

    log_cb(f"Found {total} image(s). Starting CLAHE transform…\n")

    errors = 0
    for idx, img_path in enumerate(image_paths, 1):
        rel = img_path.relative_to(dataset_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Could not read image (unsupported or corrupt).")
            enhanced = apply_clahe(img, clip_limit, tile_size)
            cv2.imwrite(str(out_path), enhanced)
        except Exception as e:
            log_cb(f"  ✗ {rel}  →  {e}")
            errors += 1

        # Copy matching label file if requested
        if copy_labels:
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                out_label = out_path.with_suffix(".txt")
                shutil.copy2(label_path, out_label)

        progress_cb(idx / total * 100)

    # Copy data.yaml / dataset.yaml if present
    for yaml_name in ("data.yaml", "dataset.yaml"):
        yaml_src = dataset_root / yaml_name
        if yaml_src.exists():
            shutil.copy2(yaml_src, output_root / yaml_name)
            log_cb(f"  ✔  Copied {yaml_name}")

    status = "completed" if errors == 0 else f"completed with {errors} error(s)"
    log_cb(f"\n✅  Processing {status}.")
    log_cb(f"    Output saved to: {output_root}")
    done_cb(True)


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CLAHE YOLO Dataset Transformer")
        self.resizable(False, False)
        self._build_ui()
        self._center()

    # ── layout ──────────────────────────────

    def _build_ui(self):
        BG       = "#0f1117"
        CARD     = "#1a1d27"
        ACCENT   = "#00e5ff"
        ACCENT2  = "#7c3aed"
        FG       = "#e2e8f0"
        FG_DIM   = "#64748b"
        ENTRY_BG = "#252836"

        self.configure(bg=BG)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",        background=BG)
        style.configure("Card.TFrame",   background=CARD)
        style.configure("TLabel",        background=BG,   foreground=FG,      font=("Courier New", 10))
        style.configure("Card.TLabel",   background=CARD, foreground=FG,      font=("Courier New", 10))
        style.configure("Dim.TLabel",    background=CARD, foreground=FG_DIM,  font=("Courier New", 9))
        style.configure("Title.TLabel",  background=BG,   foreground=ACCENT,  font=("Courier New", 15, "bold"))
        style.configure("Sub.TLabel",    background=BG,   foreground=FG_DIM,  font=("Courier New", 9))
        style.configure("Accent.TButton",
            background=ACCENT2, foreground=FG, font=("Courier New", 10, "bold"),
            borderwidth=0, relief="flat", padding=(14, 8))
        style.map("Accent.TButton",
            background=[("active", "#6d28d9"), ("disabled", "#2d2f3e")],
            foreground=[("disabled", FG_DIM)])
        style.configure("Ghost.TButton",
            background=ENTRY_BG, foreground=ACCENT, font=("Courier New", 10),
            borderwidth=0, relief="flat", padding=(10, 6))
        style.map("Ghost.TButton",
            background=[("active", "#2e3147")])
        style.configure("TCheckbutton",
            background=CARD, foreground=FG, font=("Courier New", 10))
        style.configure("TProgressbar",
            troughcolor=ENTRY_BG, background=ACCENT, thickness=6, borderwidth=0)

        # ── header ──────────────────────────
        hdr = ttk.Frame(self, padding=(24, 20, 24, 0))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="◈  CLAHE TRANSFORMER", style="Title.TLabel").pack(anchor="w")
        ttk.Label(hdr, text="Contrast-Limited Adaptive Histogram Equalization for YOLO datasets",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        sep = tk.Frame(self, bg=ACCENT2, height=1)
        sep.pack(fill="x", padx=24, pady=(12, 0))

        # ── main card ───────────────────────
        card = ttk.Frame(self, style="Card.TFrame", padding=20)
        card.pack(padx=24, pady=16, fill="x")

        # Dataset path
        self._build_path_row(card, "Dataset root:", "dataset",  ENTRY_BG, ACCENT, FG, FG_DIM)
        ttk.Frame(card, style="Card.TFrame", height=8).pack()
        self._build_path_row(card, "Output folder:", "output",   ENTRY_BG, ACCENT, FG, FG_DIM,
                             hint="Leave blank → auto (<dataset>_clahe)")

        sep2 = tk.Frame(card, bg="#252836", height=1)
        sep2.pack(fill="x", pady=12)

        # Parameters
        params = ttk.Frame(card, style="Card.TFrame")
        params.pack(fill="x")

        # Clip limit
        cl_row = ttk.Frame(params, style="Card.TFrame")
        cl_row.pack(fill="x", pady=4)
        ttk.Label(cl_row, text="Clip limit", style="Card.TLabel", width=18).pack(side="left")
        self.clip_var = tk.DoubleVar(value=2.0)
        self.clip_lbl = ttk.Label(cl_row, text="2.0", style="Card.TLabel", width=5)
        self.clip_lbl.pack(side="right")
        clip_sl = tk.Scale(cl_row, from_=0.5, to=10.0, resolution=0.5,
                           orient="horizontal", variable=self.clip_var,
                           bg=CARD, fg=FG, troughcolor=ENTRY_BG,
                           highlightthickness=0, bd=0, sliderrelief="flat",
                           command=lambda v: self.clip_lbl.config(text=f"{float(v):.1f}"),
                           showvalue=False, length=260)
        clip_sl.pack(side="left", expand=True, fill="x", padx=8)

        # Tile size
        ts_row = ttk.Frame(params, style="Card.TFrame")
        ts_row.pack(fill="x", pady=4)
        ttk.Label(ts_row, text="Tile grid size", style="Card.TLabel", width=18).pack(side="left")
        self.tile_var = tk.IntVar(value=8)
        self.tile_lbl = ttk.Label(ts_row, text="8×8", style="Card.TLabel", width=5)
        self.tile_lbl.pack(side="right")
        tile_sl = tk.Scale(ts_row, from_=2, to=32, resolution=2,
                           orient="horizontal", variable=self.tile_var,
                           bg=CARD, fg=FG, troughcolor=ENTRY_BG,
                           highlightthickness=0, bd=0, sliderrelief="flat",
                           command=lambda v: self.tile_lbl.config(text=f"{int(v)}×{int(v)}"),
                           showvalue=False, length=260)
        tile_sl.pack(side="left", expand=True, fill="x", padx=8)

        sep3 = tk.Frame(card, bg="#252836", height=1)
        sep3.pack(fill="x", pady=12)

        # Checkbox
        self.copy_labels_var = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(card, text="Copy label (.txt) files alongside images",
                              variable=self.copy_labels_var, style="TCheckbutton")
        chk.pack(anchor="w")

        # ── progress + log ──────────────────
        prog_frame = ttk.Frame(self, padding=(24, 0, 24, 0))
        prog_frame.pack(fill="x")
        self.progress = ttk.Progressbar(prog_frame, style="TProgressbar", length=500, maximum=100)
        self.progress.pack(fill="x", pady=(0, 6))

        log_frame = ttk.Frame(self, padding=(24, 0, 24, 0))
        log_frame.pack(fill="x")
        self.log = tk.Text(log_frame, height=9, bg=ENTRY_BG, fg=FG,
                           font=("Courier New", 9), relief="flat",
                           state="disabled", wrap="word",
                           insertbackground=ACCENT, selectbackground=ACCENT2)
        self.log.pack(fill="x")
        sb = tk.Scrollbar(log_frame, command=self.log.yview, bg=ENTRY_BG,
                          troughcolor=ENTRY_BG, relief="flat")
        self.log["yscrollcommand"] = sb.set

        # ── action button ───────────────────
        btn_frame = ttk.Frame(self, padding=(24, 12, 24, 20))
        btn_frame.pack(fill="x")
        self.run_btn = ttk.Button(btn_frame, text="▶  Run CLAHE Transform",
                                  style="Accent.TButton", command=self._start)
        self.run_btn.pack(fill="x")

    def _build_path_row(self, parent, label, key, bg, accent, fg, fg_dim, hint=""):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text=label, style="Card.TLabel", width=18).pack(side="left", anchor="n", pady=4)

        right = ttk.Frame(row, style="Card.TFrame")
        right.pack(side="left", fill="x", expand=True)

        var = tk.StringVar()
        setattr(self, f"{key}_var", var)

        entry_frame = tk.Frame(right, bg=bg, pady=0)
        entry_frame.pack(fill="x")
        entry = tk.Entry(entry_frame, textvariable=var, bg=bg, fg=fg,
                         font=("Courier New", 10), relief="flat",
                         insertbackground=accent, bd=6)
        entry.pack(side="left", fill="x", expand=True)

        browse = ttk.Button(entry_frame, text="Browse…", style="Ghost.TButton",
                            command=lambda k=key: self._browse(k))
        browse.pack(side="right")

        if hint:
            ttk.Label(right, text=hint, style="Dim.TLabel").pack(anchor="w", pady=(2, 0))

    # ── helpers ─────────────────────────────

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    def _browse(self, key):
        path = filedialog.askdirectory(title="Select folder")
        if path:
            getattr(self, f"{key}_var").set(path)

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_progress(self, pct: float):
        self.progress["value"] = pct

    # ── run ─────────────────────────────────

    def _start(self):
        dataset_str = self.dataset_var.get().strip()
        if not dataset_str:
            messagebox.showwarning("No dataset", "Please select a dataset root folder.")
            return

        dataset_root = Path(dataset_str)
        if not dataset_root.exists():
            messagebox.showerror("Not found", f"Path does not exist:\n{dataset_root}")
            return

        output_str = self.output_var.get().strip()
        output_root = Path(output_str) if output_str else dataset_root.parent / (dataset_root.name + "_clahe")

        clip  = float(self.clip_var.get())
        tile  = int(self.tile_var.get())
        copy  = self.copy_labels_var.get()

        self.run_btn.configure(state="disabled")
        self._set_progress(0)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

        self._log(f"Dataset : {dataset_root}")
        self._log(f"Output  : {output_root}")
        self._log(f"CLAHE   : clip={clip}, tile={tile}×{tile}")
        self._log("-" * 52)

        def progress_cb(pct):
            self.after(0, self._set_progress, pct)

        def log_cb(msg):
            self.after(0, self._log, msg)

        def done_cb(ok):
            self.after(0, self._on_done, ok)

        threading.Thread(
            target=process_dataset,
            args=(dataset_root, output_root, clip, tile, copy,
                  progress_cb, log_cb, done_cb),
            daemon=True,
        ).start()

    def _on_done(self, ok: bool):
        self.run_btn.configure(state="normal")
        self._set_progress(100 if ok else 0)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()