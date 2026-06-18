# YOLO Turbidity Detection Demo

A small Tkinter desktop app for demoing YOLO models across different
water turbidity levels.

## What it does

1. Pick a turbidity level: **3.5 NTU, 17 NTU, 25 NTU, 34 NTU**. This selects
   which folder of trained models the app looks in.
2. Pick how many models run at once: **1** (single video), **2**
   (side-by-side), or **4** (2x2 grid).
3. Pick the specific model file(s) to use from whatever `.pt` files the app
   finds for that turbidity level (you said you have 8 configs per level —
   they'll all show up in the list).
4. Pick a video file.
5. Hit **Start**. A popup window opens next to the main window showing the
   live grid of annotated video — each selected model runs on the same
   video, frame by frame. The popup has its own Stop button, and closing
   it (or hitting Stop in the main window) stops playback.

## Setup

```bash
pip install -r requirements.txt
```

(or individually: `pip install ultralytics opencv-python pillow numpy`)

## Folder structure it expects

`app.py` expects a `models/` folder sitting right next to it:

```
your_project/
├── app.py
├── requirements.txt
└── models/
    ├── 3.5_NTU/
    │   ├── config1.pt
    │   ├── config2.pt
    │   └── ... (your 8 model configs trained at 3.5 NTU)
    ├── 17_NTU/
    │   └── ... (same 8 configs, trained at 17 NTU)
    ├── 25_NTU/
    │   └── ...
    └── 34_NTU/
        └── ...
```

The app doesn't care what you name the individual `.pt` files — it scans
whichever folder matches the selected turbidity and lists every `.pt`/`.pth`
file it finds, so just drop your 32 trained weights into the matching
folders.

If your folders use different names (e.g. `NTU_3.5` instead of `3.5_NTU`),
just edit the `TURBIDITY_FOLDERS` dictionary near the top of `app.py`:

```python
TURBIDITY_FOLDERS = {
    "3.5 NTU": "3.5_NTU",
    "17 NTU": "17_NTU",
    "25 NTU": "25_NTU",
    "34 NTU": "34_NTU",
}
```

## Running it

```bash
python app.py
```

## Notes / things you may want to tweak

- **"4 models" = 2x2 grid**, not a literal 4x4 (16-cell) grid — that's the
  layout that matches "4 models side by side" most naturally. If you
  actually want a 16-cell grid for some other reason, let me know and I can
  adjust `grid_dims_for_count`.
- Each selected model is cached after first load, so switching turbidity or
  video without changing the model selection won't reload weights from
  disk every time.
- Playback runs in a background thread so the UI doesn't freeze; running 4
  models at once on a long video will be slower frame-for-frame than
  running 1, since each frame is run through every selected model in turn.
- This is a demo/inspection tool, not an optimized production pipeline —
  there's no batching across models or GPU/CPU device selection exposed in
  the UI. If you want a device picker (CPU/GPU) or a confidence-threshold
  slider added, that's a quick addition.
