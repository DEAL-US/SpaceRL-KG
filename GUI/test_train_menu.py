from tkinter import *
from tkinter import ttk

from utils import ExperimentBanner, GetDatasets

class menu():
    def __init__(self, root):
        self.root = Toplevel(root)
        self.add_elements()

    def add_elements(self):
        self.n = ttk.Notebook(self.root)

        # pages, we grid elements inside them
        self.trainframe = ttk.Frame(self.n)
        self.testframe = ttk.Frame(self.n)

        self.n.add(self.trainframe, text='Train')
        self.n.add(self.testframe, text='Test')

        self.sep_te = ttk.Separator(self.testframe, orient='horizontal')

        # Trainframe
        self.sep_tr = ttk.Separator(self.trainframe, orient='horizontal')
        self.namelabel = ttk.Label(self.trainframe, text=f'name:')
        self.lapslabel = ttk.Label(self.trainframe, text=f'laps:')
        self.datasetlabel = ttk.Label(self.trainframe, text=f'Dataset')
        self.embeddingslabel = ttk.Label(self.trainframe, text=f'Embedding')

        embeddings = ["TransE_l2", "DistMult", "ComplEx", "TransR"]
        choices_emb = StringVar(value=embeddings)
        self.embedlistbox = Listbox(self.trainframe, listvariable=choices_emb, height=4, selectmode='multiple')

        datasets = GetDatasets()
        choices_datasets = StringVar(value=list(datasets.keys()))
        self.datasetlistbox = Listbox(self.trainframe, listvariable=choices_datasets, height=4)

        self.grid_elements()

    def grid_elements(self):
        self.n.grid()

        self.grid_trainframe()
        self.grid_testframe()
    
    def grid_trainframe(self):
        pass

    def grid_testframe(self):
        pass
    
        

        