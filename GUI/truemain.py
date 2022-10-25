from tkinter import ttk
from tkinter import *
from PIL import ImageTk, Image

import random
import os
import pathlib

current_dir = pathlib.Path(__file__).parent.resolve()

import menus

class mainmenu(object):
    def __init__(self):
        # parameters:
        self.is_running = False
        
        self.root = Tk()
        self.root.title("Model Generator")
        self.root.resizable(FALSE, FALSE)
        self.root.option_add('*tearOff', FALSE)

        # this intercepts the closing button before doing so and can, for example
        # prompt the user to save its changes before proceeding.
        self.root.protocol("WM_DELETE_WINDOW", self.intercept_close)
        self.OSNAME = self.root.tk.call('tk', 'windowingsystem')


        self.mainframe = ttk.Frame(self.root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.add_styles()
        self.add_elements()

        self.root.mainloop()

    def add_elements(self):
        # first block
        self.config_lf = ttk.Labelframe(self.mainframe, text='Configuration')
        self.config_button = ttk.Button(self.config_lf, text="Config", 
        command = lambda: self.open_menu("config"))

        self.setup_button = ttk.Button(self.config_lf, text="Setup", 
        command = lambda: self.open_menu("setup"))

        #second block
        self.run_lf = ttk.Labelframe(self.mainframe, text='Runner')
        self.train_button = ttk.Button(self.run_lf, text="Train", 
        command = lambda: self.run_experimentation())

        visualize = BooleanVar(value=False)
        self.visualize_check = ttk.Checkbutton(self.run_lf, text='show\nvisuals',
        variable=visualize)

        self.test_button = ttk.Button(self.run_lf, text="Test", 
        command = lambda: self.run_tests(visualize.get()))
        
        self.grid_elements()

    def grid_elements(self):
        #row0
        self.config_lf.grid(column=0, row=0)
        self.config_button.grid(column=0, row=0)
        self.setup_button.grid(column=0, row=1)

        self.run_lf.grid(row=0, column=1)
        self.train_button.grid(row=0, column=0, columnspan=2)
        self.test_button.grid(row=1, column=0)
        self.visualize_check.grid(row=1, column=1)



    def add_styles(self):
        s = ttk.Style()
        self.root.tk.call('lappend', 'auto_path', f"{current_dir}/awdark/")
        self.root.tk.call('package', 'require', 'awdark')
        s.theme_use('awdark')


    # MISC OPERATIONS

    def open_menu(self, menutype):
        if(menutype == "config"):
            config = menus.configmenu(root)
            config.instantiate()
            
        elif(menutype == "setup"):
            config = menus.experiment_test_menu(root)
            config.instantiate()

    # root.after(5000, lambda: config.close())

    def intercept_close(self):
        if self.is_running:
            messagebox.askyesno(
            message='There is something running, do you want to close anyway?',
            icon='warning', title='Interruption alert.')

        self.root.destroy()

    def run_experimentation(self):
        print("running experiments...")

    def run_tests(self, show_visuals):
        print("running testing...")

mainmenu()