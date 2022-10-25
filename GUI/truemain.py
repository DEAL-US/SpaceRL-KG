from tkinter import ttk
from tkinter import *
from PIL import ImageTk, Image

import random
import os
import pathlib

current_dir = pathlib.Path(_file_).parent.resolve()

import menus

class mainmenu(object):
    def __init__(self):
        # parameters:
        self.is_running = False
        self.OSNAME = root.tk.call('tk', 'windowingsystem')

        self.root = Tk()
        self.root.title("Model Generator")
        self.root.resizable(FALSE, FALSE)
        self.root.option_add('*tearOff', FALSE)

        # this intercepts the closing button before doing so and can, for example
        # prompt the user to save its changes before proceeding.
        root.protocol("WM_DELETE_WINDOW", self.intercept_close)


        self.mainframe = ttk.Frame(root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.add_elements()

    def add_elements(self):
        pass

        config_lf = ttk.Labelframe(self.mainframe, text='Configuration')
        run_lf = ttk.Labelframe(self.mainframe, text='Runner')

        config_button = ttk.Button(config, text="Config", 
        command=lambda: open_menu("config"))
        

        self.grid_elements()

    def grid_elements(self):
        pass
    

    def add_styles():
        s = ttk.Style()
        # print(s.theme_names())
        # s.theme_use('clam') # tkinter alt theme.
        root.tk.call('lappend', 'auto_path', f"{current_dir}/awdark/")
        root.tk.call('package', 'require', 'awdark')
        s.theme_use('awdark') # imported theme


    # MISC OPERATIONS

    def open_menu(menutype):
        if(menutype == "config"):
            config = menus.configmenu(root)
            config.instantiate()
            config.add_elements()
            config.grid_elements()
        elif(menutype == "experiments"):
            config = menus.configmenu(root)
            config.instantiate()
            config.add_elements()
            config.grid_elements()
        elif(menutype == "testing"):
            config = menus.configmenu(root)
            config.instantiate()
            config.add_elements()
            config.grid_elements()

    # root.after(5000, lambda: config.close())

    def intercept_close():
        if self.is_running:
            messagebox.askyesno(
            message='There is something running, do you want to close anyway?',
            icon='warning', title='Interruption alert.')


mainmenu()