from tkinter import *
from tkinter import ttk

class menu():
    def __init__(self, root):
        self.root = Toplevel(root)
        self.add_elements()

    def add_elements(self):
        print("I override the other method.")

    def grid_elements(self):
        print("I override the other method.")