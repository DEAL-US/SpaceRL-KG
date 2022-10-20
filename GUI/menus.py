from tkinter import *
from tkinter import ttk

class menuwindow(object):
    def __init__(self, root):
        self.root = root

    def instantiate(self):
        self.t = Toplevel(self.root)
        # self.t.geometry('500x400-5+40')

    def close(self):
        self.t.destroy()

    def add_elements(self):
        pass
    

    def grid_elements(self):
        pass


class configmenu(menuwindow):
    def add_elements(self):
        print("I override the other method.")

    def grid_elements(self):
        print("I override the other method.")

class experiment_test_menu(menuwindow):
    pass
