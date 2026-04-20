from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import liveDetectionVideoStacking

def browse_file(*args):
    file_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )
    liveDetectionVideoStacking.run_comparison_demo(file_path)


root = Tk()
root.title("Attention Enhance YOLO Model Demo")

mainframe = ttk.Frame(root, padding=(3,3,12,12))
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

ttk.Label(mainframe, text="Select a video file to run the demo:").grid(column=1, row=1, sticky=W)
ttk.Button(mainframe, text="Browse", command=browse_file).grid(column=2, row=1, sticky=W)

root.mainloop()