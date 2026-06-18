"""
YOLO Turbidity Detection Demo
==============================

A small desktop app (Tkinter UI) that demonstrates running YOLO detection
models on a video at different turbidity levels.

Features
--------
- Choose a turbidity level: 3.5 NTU, 17 NTU, 25 NTU, 34 NTU
  (this selects which subfolder of trained models is used)
- Choose a video file from disk
- Choose how many models run at once: 1, 2 (side-by-side), or 4 (2x2 grid)
- Pick the specific model configuration(s) to run from whichever .pt files
  are found for the selected turbidity level

Expected folder layout (models/ lives next to this script)
------------------------------------------------------------
    app.py
    models/
        3.5_NTU/
            <config_1>.pt
            <config_2>.pt
            ... up to 8 configs ...
        17_NTU/
            ... same 8 configs, trained for this turbidity ...
        25_NTU/
            ...
        34_NTU/
            ...

You don't need to rename your files to anything special -- the app scans
each turbidity folder at runtime and lists whatever .pt files it finds.
If your folder names differ from the ones above, just edit
TURBIDITY_FOLDERS below.

Requirements
------------
    pip install ultralytics opencv-python pillow numpy
"""

import os
import time
import threading

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Turbidity label shown in the UI -> subfolder name under models/.
# Edit this if your folders are named differently.
TURBIDITY_FOLDERS = {
    "3.5 NTU": "3.5_NTU",
    "17 NTU": "17_NTU",
    "25 NTU": "25_NTU",
    "34 NTU": "34_NTU",
}

ALLOWED_MODEL_COUNTS = (1, 2, 4)

DISPLAY_W, DISPLAY_H = 960, 540  # size of the video canvas in the UI

# Multiplier for the on-screen model-name label. 1.0 is a good default for
# all three grid modes since the label is now sized relative to each cell's
# own height (not the original video resolution) -- bump this up/down if
# you want the label bigger/smaller across the board.
LABEL_SCALE = 1.0
REFERENCE_CELL_AREA = DISPLAY_W * DISPLAY_H  # baseline = a full single-video cell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_models_for_turbidity(turbidity_label):
    """Return the sorted list of .pt/.pth files found for a turbidity level."""
    folder = os.path.join(MODELS_DIR, TURBIDITY_FOLDERS.get(turbidity_label, ""))
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".pt", ".pth"))]
    return sorted(files)


def grid_dims_for_count(n):
    """rows, cols layout for a given number of simultaneous models."""
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n == 4:
        return 2, 2
    raise ValueError(f"Unsupported model count: {n}")


def make_composite(frames, rows, cols, cell_w, cell_h):
    """Resize frames to (cell_w, cell_h) and tile them into a grid image."""
    cells = [cv2.resize(f, (cell_w, cell_h)) for f in frames]
    while len(cells) < rows * cols:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    row_imgs = []
    for r in range(rows):
        row_cells = cells[r * cols:(r + 1) * cols]
        row_imgs.append(cv2.hconcat(row_cells))
    return cv2.vconcat(row_imgs)


def label_frame(frame, text, scale=1.0):
    """Burn a text label into the top-left corner of a frame.

    Call this AFTER resizing the frame down to its final on-screen cell
    size, not before -- the label size is derived from the frame's own
    current dimensions (relative to a full single-video cell), so it stays
    legible at any grid size instead of shrinking along with the cell.
    `scale` is an extra multiplier (see LABEL_SCALE) if you want every
    label bigger/smaller across the board.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    area_ratio = (w * h) / REFERENCE_CELL_AREA
    size_factor = max(0.4, min(1.0, area_ratio ** 0.5)) * scale

    font_scale = 0.55 * size_factor
    thickness = max(1, round(2 * size_factor))
    pad_x = max(4, round(6 * size_factor))
    bar_h = max(16, round(26 * size_factor))
    text_y = round(bar_h * 0.72)

    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bar_w = min(w, text_w + pad_x * 2)

    cv2.rectangle(out, (0, 0), (bar_w, bar_h), (0, 0, 0), -1)
    cv2.putText(out, text, (pad_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class DetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Turbidity Detection Demo")
        self.geometry("640x420")
        self.resizable(False, False)

        self.video_path = None
        self.cap = None
        self.playback_thread = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()   # set = paused
        self.loaded_models = {}  # path -> loaded YOLO model (cached)
        self._tk_img = None      # keep a reference so Tk doesn't GC it

        # Frame history for prev/next navigation while paused
        self._frame_cache = []        # list of BGR composites already rendered
        self._frame_cache_max = 300   # keep at most N frames in memory
        self._current_frame_idx = -1  # index into _frame_cache being displayed

        self.video_window = None  # popup window shown while running
        self.canvas = None
        self.canvas_image_id = None

        # Playback-control buttons (created in _open_video_window)
        self.pause_btn = None
        self.prev_btn = None
        self.next_btn = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Turbidity:").grid(row=0, column=0, sticky="w")
        self.turbidity_var = tk.StringVar(value=list(TURBIDITY_FOLDERS.keys())[0])
        turbidity_combo = ttk.Combobox(
            top, textvariable=self.turbidity_var,
            values=list(TURBIDITY_FOLDERS.keys()), state="readonly", width=12
        )
        turbidity_combo.grid(row=0, column=1, sticky="w")
        turbidity_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_models())

        ttk.Label(top, text="Models running:").grid(row=0, column=2, sticky="w", padx=(24, 0))
        self.count_var = tk.IntVar(value=1)
        for i, c in enumerate(ALLOWED_MODEL_COUNTS):
            label = "1 (single)" if c == 1 else ("2 (side-by-side)" if c == 2 else "4 (2x2 grid)")
            ttk.Radiobutton(
                top, text=label, variable=self.count_var, value=c,
                command=self._on_count_change
            ).grid(row=0, column=3 + i, sticky="w", padx=(0, 6))

        video_frame = ttk.Frame(self)
        video_frame.pack(fill="x", **pad)
        ttk.Button(video_frame, text="Choose video...", command=self._choose_video).pack(side="left")
        self.video_label_var = tk.StringVar(value="No video selected")
        ttk.Label(video_frame, textvariable=self.video_label_var).pack(side="left", padx=8)

        models_frame = ttk.LabelFrame(self, text="Select model configuration(s)")
        models_frame.pack(fill="x", **pad)
        self.hint_var = tk.StringVar(value="")
        ttk.Label(models_frame, textvariable=self.hint_var, foreground="#555").pack(
            anchor="w", padx=6, pady=(4, 0)
        )
        self.models_listbox = tk.Listbox(
            models_frame, selectmode=tk.MULTIPLE, height=8, exportselection=False
        )
        self.models_listbox.pack(fill="x", padx=6, pady=6)

        controls = ttk.Frame(self)
        controls.pack(fill="x", **pad)
        self.start_btn = ttk.Button(controls, text="Start", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=20)

        self._refresh_models()
        self._on_count_change()

    # ------------------------------------------------------------------
    # UI behaviour
    # ------------------------------------------------------------------
    def _refresh_models(self):
        self.models_listbox.delete(0, tk.END)
        files = list_models_for_turbidity(self.turbidity_var.get())
        if not files:
            folder = TURBIDITY_FOLDERS[self.turbidity_var.get()]
            self.models_listbox.insert(tk.END, f"(no .pt files found in models/{folder})")
        for f in files:
            self.models_listbox.insert(tk.END, f)

    def _on_count_change(self):
        n = self.count_var.get()
        self.hint_var.set(f"Select exactly {n} model file{'s' if n != 1 else ''} below.")

    def _choose_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            self.video_label_var.set(os.path.basename(path))

    def _selected_model_files(self):
        idxs = self.models_listbox.curselection()
        return [self.models_listbox.get(i) for i in idxs]

    # ------------------------------------------------------------------
    # Popup video window
    # ------------------------------------------------------------------
    def _open_video_window(self, title_text):
        self._close_video_window()

        win = tk.Toplevel(self)
        win.title(title_text)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", self._stop)

        canvas = tk.Canvas(win, width=DISPLAY_W, height=DISPLAY_H, bg="black")
        canvas.pack()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=6)

        prev_btn = ttk.Button(btn_frame, text="⏮ Prev", command=self._prev_frame, state="disabled")
        prev_btn.pack(side="left", padx=4)

        pause_btn = ttk.Button(btn_frame, text="⏸ Pause", command=self._toggle_pause)
        pause_btn.pack(side="left", padx=4)

        next_btn = ttk.Button(btn_frame, text="Next ⏭", command=self._next_frame, state="disabled")
        next_btn.pack(side="left", padx=4)

        ttk.Button(btn_frame, text="⏹ Stop", command=self._stop).pack(side="left", padx=4)

        # Place the popup just to the right of the main window.
        self.update_idletasks()
        x = self.winfo_x() + self.winfo_width() + 16
        y = self.winfo_y()
        win.geometry(f"+{x}+{y}")

        self.video_window = win
        self.canvas = canvas
        self.canvas_image_id = canvas.create_image(0, 0, anchor="nw")
        self.pause_btn = pause_btn
        self.prev_btn = prev_btn
        self.next_btn = next_btn

    def _close_video_window(self):
        if self.video_window is not None:
            try:
                self.video_window.destroy()
            except tk.TclError:
                pass
            self.video_window = None
            self.canvas = None
            self.canvas_image_id = None

    # ------------------------------------------------------------------
    # Start / stop / playback
    # ------------------------------------------------------------------
    def _start(self):
        if YOLO is None:
            messagebox.showerror(
                "Missing dependency",
                "The 'ultralytics' package isn't installed.\n\nRun:\n"
                "    pip install ultralytics opencv-python pillow numpy"
            )
            return
        if not self.video_path:
            messagebox.showwarning("No video", "Please choose a video first.")
            return

        needed = self.count_var.get()
        selected = self._selected_model_files()
        if len(selected) != needed:
            messagebox.showwarning(
                "Model selection",
                f"Please select exactly {needed} model file(s) "
                f"({len(selected)} currently selected)."
            )
            return

        folder = os.path.join(MODELS_DIR, TURBIDITY_FOLDERS[self.turbidity_var.get()])
        model_paths = [os.path.join(folder, f) for f in selected]

        self.status_var.set("Loading model(s)...")
        self.update_idletasks()
        try:
            models = []
            for p in model_paths:
                if p not in self.loaded_models:
                    self.loaded_models[p] = YOLO(p)
                models.append(self.loaded_models[p])
        except Exception as e:
            messagebox.showerror("Model load error", str(e))
            self.status_var.set("Ready")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Video error", "Could not open the selected video.")
            return
        self.cap = cap

        self.stop_flag.clear()
        self.pause_flag.clear()
        self._frame_cache.clear()
        self._current_frame_idx = -1
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Running...")

        labels = [os.path.splitext(f)[0] for f in selected]
        popup_title = f"{self.turbidity_var.get()} — {', '.join(labels)}"
        self._open_video_window(popup_title)

        self.playback_thread = threading.Thread(
            target=self._playback_loop, args=(models, labels), daemon=True
        )
        self.playback_thread.start()

    def _stop(self):
        self.stop_flag.set()
        self.pause_flag.clear()  # unblock spin-wait so thread can exit
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped")
        self._close_video_window()

    def _toggle_pause(self):
        if self.pause_flag.is_set():
            # Resume
            self.pause_flag.clear()
            if self.pause_btn:
                self.pause_btn.config(text="⏸ Pause")
            if self.next_btn:
                self.next_btn.config(state="disabled")
            self.status_var.set("Running...")
        else:
            # Pause
            self.pause_flag.set()
            if self.pause_btn:
                self.pause_btn.config(text="▶ Resume")
            self._update_nav_buttons()
            self.status_var.set("Paused")

    def _update_nav_buttons(self):
        """Enable/disable Prev and Next based on current position in cache."""
        if not self.pause_flag.is_set():
            return
        can_prev = self._current_frame_idx > 0
        can_next = self._current_frame_idx < len(self._frame_cache) - 1
        if self.prev_btn:
            self.prev_btn.config(state="normal" if can_prev else "disabled")
        if self.next_btn:
            self.next_btn.config(state="normal" if can_next else "disabled")

    def _prev_frame(self):
        if self._current_frame_idx > 0:
            self._current_frame_idx -= 1
            self._show_frame(self._frame_cache[self._current_frame_idx])
            self._update_nav_buttons()

    def _next_frame(self):
        if self._current_frame_idx < len(self._frame_cache) - 1:
            self._current_frame_idx += 1
            self._show_frame(self._frame_cache[self._current_frame_idx])
            self._update_nav_buttons()

    def _playback_loop(self, models, labels):
        rows, cols = grid_dims_for_count(len(models))
        cell_w, cell_h = DISPLAY_W // cols, DISPLAY_H // rows
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps

        while not self.stop_flag.is_set():
            # Spin-wait while paused (sleep briefly to avoid burning CPU)
            if self.pause_flag.is_set():
                time.sleep(0.05)
                continue

            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                break

            annotated_frames = []
            for model, label in zip(models, labels):
                try:
                    results = model(frame, verbose=False)
                    annotated = results[0].plot()
                except Exception:
                    annotated = frame.copy()
                resized = cv2.resize(annotated, (cell_w, cell_h))
                annotated_frames.append(label_frame(resized, label, scale=LABEL_SCALE))

            composite = make_composite(annotated_frames, rows, cols, cell_w, cell_h)

            # Cache the composite for prev/next navigation
            self._frame_cache.append(composite)
            if len(self._frame_cache) > self._frame_cache_max:
                self._frame_cache.pop(0)
            self._current_frame_idx = len(self._frame_cache) - 1

            self._show_frame(composite)

            elapsed = time.time() - t0
            if elapsed < delay:
                time.sleep(delay - elapsed)

        if self.cap:
            self.cap.release()
        self.after(0, self._on_playback_end)

    def _on_playback_end(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if self.status_var.get() == "Running...":
            self.status_var.set("Finished")

    def _show_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(img)

        def update():
            if self.canvas is None or self.canvas_image_id is None:
                return  # popup was closed mid-frame
            self._tk_img = tk_img  # keep reference, avoid garbage collection
            self.canvas.itemconfig(self.canvas_image_id, image=self._tk_img)

        self.after(0, update)

    def on_close(self):
        self.stop_flag.set()
        self.pause_flag.clear()
        if self.cap:
            self.cap.release()
        self._close_video_window()
        self.destroy()


if __name__ == "__main__":
    app = DetectionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
