from tkinter import *
from tkinter import ttk

class menuwindow(object):
    def _init_(self, root):
        self.root = Toplevel(root)
        self.add_elements()

    def close(self):
        self.root.destroy()

    def add_elements(self):
        # add all elments to the window.
        self.grid_elements()
    
    def grid_elements(self):
        # set all elements in their grid positions.
        pass

class configmenu(menuwindow):
    def _init_(self, root):
        super()._init_(root)

    def add_elements(self):
        print("I override the other method.")

    def grid_elements(self):
        print("I override the other method.")

class experiment_test_menu(menuwindow):
    def _init_(self, root):
        super()._init_(root)

    def add_elements(self):
        

        self.n = ttk.Notebook(self.root)
        # pages, we grid elements inside them
        f1 = ttk.Frame(self.n)
        f2 = ttk.Frame(self.n)
        f3 = ttk.Frame(self.n)

        self.n.add(f1, text='One')
        self.n.add(f2, text='Two')
        self.n.add(f3, text='Three')

    def grid_elements(self):
        print("im here.")

        self.n.grid()