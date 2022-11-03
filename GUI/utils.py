from tkinter import *
from tkinter import ttk

import sys, pathlib, os

# using pathlib to help with mac and linux compatibility.
current_dir = pathlib.Path(__file__).parent.resolve()
maindir = pathlib.Path(current_dir).parent.resolve()
datasets_folder = pathlib.Path(f"{maindir}\\datasets").resolve()
agents_folder = pathlib.Path(f"{maindir}\\model\\data\\agents").resolve()

class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#33393b", relief=SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"), fg="white")
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class ExperimentBanner(object):
    def __init__(self, frame, bannertext, experiment_name :str, 
    laps : int, dataset : str, embeddings : list[str] ):
        parent = ttk.Labelframe(frame, text=bannertext)
        namelabel = ttk.Label(parent, text=f'name: {experiment_name}')
        datalabel = ttk.Label(parent, text=f'dataset: {dataset}')
        laplabel = ttk.Label(parent, text=f'laps: {laps}')
        embeddingslabel = ttk.Label(parent, text=f'embeddings:\n {embeddings}')
        namelabel.grid(row=0, column=0)
        datalabel.grid(row=1, column=0)
        laplabel.grid(row=2, column=0)
        embeddingslabel.grid(row=3, column=0)
        self.parent = parent

    def getbanner(self):
        return self.parent


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

def GetConfig(is_experiments):
    sys.path.insert(0, f"{maindir}\\model")
    from config import get_config
    sys.path.pop(0)
    return get_config(is_experiments)

def GetDatasets():
    names, paths = [],[]
    for name in os.listdir(datasets_folder):
        dirpath = pathlib.Path(f"{datasets_folder}/{name}").resolve()
        if os.path.isdir(dirpath):
            names.append(name)
            paths.append(pathlib.Path(f"{datasets_folder}/{name}/graph.txt").resolve())

    return dict(zip(names, paths))
    
# asd = GetDatasets()
# print(asd)