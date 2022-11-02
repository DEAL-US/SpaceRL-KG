from tkinter import *
from tkinter import ttk

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

        self.sep_tr = ttk.Separator(self.trainframe, orient='horizontal')
        self.sep_te = ttk.Separator(self.testframe, orient='horizontal')

        self.grid_elements()

    def grid_elements(self):
        self.n.grid()