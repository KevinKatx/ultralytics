"""
Retinex YOLO Dataset Transformer
==================================
Applies Retinex-based illumination normalization to images in a YOLO-format
dataset. Labels are preserved untouched.

Three variants are supported:
  SSR  – Single-Scale Retinex          (fast, one Gaussian blur pass)
  MSR  – Multi-Scale Retinex           (balanced, three scales averaged)
  MSRCR– MSR with Color Restoration    (best colour fidelity, recommended)

Usage:
    python retinex_yolo_transform.py

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
#  Retinex Core Logic
# ─────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

EPS = 1e-6  # avoid log(0)


def _single_scale_retinex(img_float: np.ndarray, sigma: float) -> np.ndarray:
    """SSR on a single float channel [0,255]. Returns log-domain output."""
    blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
    return np.log1p(img_float) - np.log1p(blur + EPS)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Stretch each channel independently to [0, 255] uint8."""
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        lo, hi = ch.min(), ch.max()
        if hi - lo < EPS:
            out[:, :, c] = 0
        else:
            out[:, :, c] = (ch - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_ssr(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    """Single-Scale Retinex."""
    img_f = img_bgr.astype(np.float32) + 1.0
    result = np.zeros_like(img_f)
    for c in range(3):
        result[:, :, c] = _single_scale_retinex(img_f[:, :, c], sigma)
    return _normalize_to_uint8(result)


def apply_msr(img_bgr: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Multi-Scale Retinex (equal-weight average over scales)."""
    img_f = img_bgr.astype(np.float32) + 1.0
    result = np.zeros_like(img_f)
    weight = 1.0 / len(sigmas)
    for sigma in sigmas:
        for c in range(3):
            result[:, :, c] += weight * _single_scale_retinex(img_f[:, :, c], sigma)
    return _normalize_to_uint8(result)


def apply_msrcr(
    img_bgr: np.ndarray,
    sigmas: list[float],
    restoration_factor: float,
    color_gain: float,
    color_offset: float,
) -> np.ndarray:
    """Multi-Scale Retinex with Color Restoration."""
    img_f = img_bgr.astype(np.float32) + 1.0

    # MSR component
    msr = np.zeros_like(img_f)
    weight = 1.0 / len(sigmas)
    for sigma in sigmas:
        for c in range(3):
            msr[:, :, c] += weight * _single_scale_retinex(img_f[:, :, c], sigma)

    # Color restoration function  C_i = f * log(α * I_i / Σ I_j)
    total = np.sum(img_f, axis=2, keepdims=True) + EPS
    color_restore = restoration_factor * (
        np.log1p(color_gain * img_f) - np.log1p(color_offset + total)
    )

    result = color_restore * msr
    return _normalize_to_uint8(result)


def apply_retinex(
    img_bgr: np.ndarray,
    variant: str,
    sigma_low: float,
    sigma_mid: float,
    sigma_high: float,
    restoration_factor: float,
    color_gain: float,
    color_offset: float,
) -> np.ndarray:
    """Dispatch to the chosen Retinex variant."""
    sigmas = [sigma_low, sigma_mid, sigma_high]
    if variant == "SSR":
        return apply_ssr(img_bgr, sigma_mid)
    elif variant == "MSR":
        return apply_msr(img_bgr, sigmas)
    else:  # MSRCR
        return apply_msrcr(img_bgr, sigmas, restoration_factor, color_gain, color_offset)


def process_dataset(
    dataset_root: Path,
    output_root: Path,
    variant: str,
    sigma_low: float,
    sigma_mid: float,
    sigma_high: float,
    restoration_factor: float,
    color_gain: float,
    color_offset: float,
    copy_labels: bool,
    progress_cb,
    log_cb,
    done_cb,
):
    """Run Retinex on all images; copy labels unchanged."""
    image_paths = []
    for ext in SUPPORTED_EXTS:
        image_paths.extend(dataset_root.rglob(f"*{ext}"))
    image_paths = sorted(image_paths)

    total = len(image_paths)
    if total == 0:
        log_cb("⚠  No images found in the selected directory.")
        done_cb(False)
        return

    log_cb(f"Found {total} image(s). Starting {variant} transform…\n")

    errors = 0
    for idx, img_path in enumerate(image_paths, 1):
        rel = img_path.relative_to(dataset_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Could not read image (unsupported or corrupt).")
            enhanced = apply_retinex(
                img, variant,
                sigma_low, sigma_mid, sigma_high,
                restoration_factor, color_gain, color_offset,
            )
            cv2.imwrite(str(out_path), enhanced)
        except Exception as e:
            log_cb(f"  ✗ {rel}  →  {e}")
            errors += 1

        if copy_labels:
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                shutil.copy2(label_path, out_path.with_suffix(".txt"))

        progress_cb(idx / total * 100)

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
        self.title("Retinex YOLO Dataset Transformer")
        self.resizable(False, False)
        self._build_ui()
        self._center()

    # ── colours ─────────────────────────────

    BG       = "#0d1117"
    CARD     = "#161b22"
    ACCENT   = "#f0a500"      # warm amber — Retinex = light, warmth
    ACCENT2  = "#d4590a"
    FG       = "#e6edf3"
    FG_DIM   = "#6e7681"
    ENTRY_BG = "#21262d"

    # ── layout ──────────────────────────────

    def _build_ui(self):
        BG, CARD, ACCENT, ACCENT2 = self.BG, self.CARD, self.ACCENT, self.ACCENT2
        FG, FG_DIM, ENTRY_BG = self.FG, self.FG_DIM, self.ENTRY_BG

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
            background=[("active", "#b84d09"), ("disabled", "#2d2f3e")],
            foreground=[("disabled", FG_DIM)])
        style.configure("Ghost.TButton",
            background=ENTRY_BG, foreground=ACCENT, font=("Courier New", 10),
            borderwidth=0, relief="flat", padding=(10, 6))
        style.map("Ghost.TButton",
            background=[("active", "#2a2f38")])
        style.configure("TCheckbutton",
            background=CARD, foreground=FG, font=("Courier New", 10))
        style.configure("TProgressbar",
            troughcolor=ENTRY_BG, background=ACCENT, thickness=6, borderwidth=0)
        style.configure("TRadiobutton",
            background=CARD, foreground=FG, font=("Courier New", 10))
        style.map("TRadiobutton",
            background=[("active", CARD)])

        # ── header ──────────────────────────
        hdr = ttk.Frame(self, padding=(24, 20, 24, 0))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="◈  RETINEX TRANSFORMER", style="Title.TLabel").pack(anchor="w")
        ttk.Label(hdr, text="Illumination normalization via Retinex theory for YOLO datasets",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        tk.Frame(self, bg=ACCENT2, height=1).pack(fill="x", padx=24, pady=(12, 0))

        # ── main card ───────────────────────
        card = ttk.Frame(self, style="Card.TFrame", padding=20)
        card.pack(padx=24, pady=16, fill="x")

        # Paths
        self._build_path_row(card, "Dataset root:", "dataset")
        ttk.Frame(card, style="Card.TFrame", height=8).pack()
        self._build_path_row(card, "Output folder:", "output",
                             hint="Leave blank → auto (<dataset>_retinex)")

        tk.Frame(card, bg=ENTRY_BG, height=1).pack(fill="x", pady=12)

        # ── Variant selector ────────────────
        var_frame = ttk.Frame(card, style="Card.TFrame")
        var_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(var_frame, text="Variant", style="Card.TLabel", width=18).pack(side="left")

        self.variant_var = tk.StringVar(value="MSRCR")
        for v in ("SSR", "MSR", "MSRCR"):
            rb = ttk.Radiobutton(var_frame, text=v, value=v,
                                 variable=self.variant_var, style="TRadiobutton",
                                 command=self._on_variant_change)
            rb.pack(side="left", padx=(0, 14))

        self.variant_hint = ttk.Label(card,
            text="MSR + colour restoration — best colour fidelity  ★ recommended",
            style="Dim.TLabel")
        self.variant_hint.pack(anchor="w", padx=(0, 0), pady=(0, 6))

        tk.Frame(card, bg=ENTRY_BG, height=1).pack(fill="x", pady=(4, 12))

        # ── Sigma sliders ───────────────────
        params = ttk.Frame(card, style="Card.TFrame")
        params.pack(fill="x")

        self._sliders = {}

        self._build_slider(params, "sigma_low",  "Sigma low",   15,  5, 100,  5,
                           fmt=lambda v: f"{int(v)}")
        self._build_slider(params, "sigma_mid",  "Sigma mid",   80,  5, 200,  5,
                           fmt=lambda v: f"{int(v)}")
        self._build_slider(params, "sigma_high", "Sigma high", 250, 50, 500, 10,
                           fmt=lambda v: f"{int(v)}")

        tk.Frame(card, bg=ENTRY_BG, height=1).pack(fill="x", pady=10)

        # ── MSRCR-only sliders ──────────────
        self._cr_frame = ttk.Frame(card, style="Card.TFrame")
        self._cr_frame.pack(fill="x")

        self._build_slider(self._cr_frame, "restoration", "Restoration",  125,  10, 300, 5,
                           fmt=lambda v: f"{int(v)}")
        self._build_slider(self._cr_frame, "color_gain",  "Colour gain",  186,  10, 400, 2,
                           fmt=lambda v: f"{int(v)}")
        self._build_slider(self._cr_frame, "color_offset","Colour offset",  125,  1, 300, 5,
                           fmt=lambda v: f"{int(v)}")

        tk.Frame(card, bg=ENTRY_BG, height=1).pack(fill="x", pady=10)

        # Checkbox
        self.copy_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(card, text="Copy label (.txt) files alongside images",
                        variable=self.copy_labels_var, style="TCheckbutton").pack(anchor="w")

        # ── progress + log ──────────────────
        prog_frame = ttk.Frame(self, padding=(24, 0, 24, 0))
        prog_frame.pack(fill="x")
        self.progress = ttk.Progressbar(prog_frame, style="TProgressbar", length=500, maximum=100)
        self.progress.pack(fill="x", pady=(0, 6))

        log_frame = ttk.Frame(self, padding=(24, 0, 24, 0))
        log_frame.pack(fill="x")
        self.log_widget = tk.Text(log_frame, height=9, bg=ENTRY_BG, fg=FG,
                                  font=("Courier New", 9), relief="flat",
                                  state="disabled", wrap="word",
                                  insertbackground=ACCENT, selectbackground=ACCENT2)
        self.log_widget.pack(fill="x")
        sb = tk.Scrollbar(log_frame, command=self.log_widget.yview,
                          bg=ENTRY_BG, troughcolor=ENTRY_BG, relief="flat")
        self.log_widget["yscrollcommand"] = sb.set

        # ── run button ──────────────────────
        btn_frame = ttk.Frame(self, padding=(24, 12, 24, 20))
        btn_frame.pack(fill="x")
        self.run_btn = ttk.Button(btn_frame, text="▶  Run Retinex Transform",
                                  style="Accent.TButton", command=self._start)
        self.run_btn.pack(fill="x")

    # ── helpers ─────────────────────────────

    _VARIANT_HINTS = {
        "SSR":   "Single-Scale Retinex — fast, uses Sigma mid only",
        "MSR":   "Multi-Scale Retinex — averaged over three sigma scales",
        "MSRCR": "MSR + colour restoration — best colour fidelity  ★ recommended",
    }

    def _on_variant_change(self):
        v = self.variant_var.get()
        self.variant_hint.config(text=self._VARIANT_HINTS[v])
        # Show/hide colour-restoration sliders
        if v == "MSRCR":
            self._cr_frame.pack(fill="x")
        else:
            self._cr_frame.pack_forget()

    def _build_path_row(self, parent, label, key, hint=""):
        BG, ACCENT, FG, FG_DIM, ENTRY_BG = (
            self.BG, self.ACCENT, self.FG, self.FG_DIM, self.ENTRY_BG)
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text=label, style="Card.TLabel", width=18).pack(side="left", anchor="n", pady=4)

        right = ttk.Frame(row, style="Card.TFrame")
        right.pack(side="left", fill="x", expand=True)

        var = tk.StringVar()
        setattr(self, f"{key}_var", var)

        ef = tk.Frame(right, bg=ENTRY_BG)
        ef.pack(fill="x")
        tk.Entry(ef, textvariable=var, bg=ENTRY_BG, fg=FG,
                 font=("Courier New", 10), relief="flat",
                 insertbackground=ACCENT, bd=6).pack(side="left", fill="x", expand=True)
        ttk.Button(ef, text="Browse…", style="Ghost.TButton",
                   command=lambda k=key: self._browse(k)).pack(side="right")

        if hint:
            ttk.Label(right, text=hint, style="Dim.TLabel").pack(anchor="w", pady=(2, 0))

    def _build_slider(self, parent, key, label, default, lo, hi, step, fmt):
        CARD, FG, ENTRY_BG = self.CARD, self.FG, self.ENTRY_BG
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="Card.TLabel", width=18).pack(side="left")
        var = tk.DoubleVar(value=default)
        lbl = ttk.Label(row, text=fmt(default), style="Card.TLabel", width=6)
        lbl.pack(side="right")
        sl = tk.Scale(row, from_=lo, to=hi, resolution=step,
                      orient="horizontal", variable=var,
                      bg=CARD, fg=FG, troughcolor=ENTRY_BG,
                      highlightthickness=0, bd=0, sliderrelief="flat",
                      command=lambda v, l=lbl, f=fmt: l.config(text=f(float(v))),
                      showvalue=False, length=250)
        sl.pack(side="left", expand=True, fill="x", padx=8)
        self._sliders[key] = var

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
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", msg + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

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
        suffix = f"_retinex_{self.variant_var.get().lower()}"
        output_root = (Path(output_str) if output_str
                       else dataset_root.parent / (dataset_root.name + suffix))

        variant       = self.variant_var.get()
        sigma_low     = float(self._sliders["sigma_low"].get())
        sigma_mid     = float(self._sliders["sigma_mid"].get())
        sigma_high    = float(self._sliders["sigma_high"].get())
        restoration   = float(self._sliders["restoration"].get())
        color_gain    = float(self._sliders["color_gain"].get())
        color_offset  = float(self._sliders["color_offset"].get())
        copy          = self.copy_labels_var.get()

        self.run_btn.configure(state="disabled")
        self._set_progress(0)
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")

        self._log(f"Dataset : {dataset_root}")
        self._log(f"Output  : {output_root}")
        self._log(f"Variant : {variant}")
        if variant == "SSR":
            self._log(f"Sigma   : {sigma_mid}")
        else:
            self._log(f"Sigmas  : {sigma_low} / {sigma_mid} / {sigma_high}")
        if variant == "MSRCR":
            self._log(f"Restore : {restoration}  gain={color_gain}  offset={color_offset}")
        self._log("-" * 52)

        def progress_cb(pct):  self.after(0, self._set_progress, pct)
        def log_cb(msg):       self.after(0, self._log, msg)
        def done_cb(ok):       self.after(0, self._on_done, ok)

        threading.Thread(
            target=process_dataset,
            args=(dataset_root, output_root, variant,
                  sigma_low, sigma_mid, sigma_high,
                  restoration, color_gain, color_offset,
                  copy, progress_cb, log_cb, done_cb),
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