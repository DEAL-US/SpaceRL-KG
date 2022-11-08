from tkinter import ttk, filedialog, messagebox
from tkinter import *
from utils import CreateToolTip, GetConfig

import random, os, sys, pathlib, subprocess, config_menu, test_train_menu

current_dir = pathlib.Path(__file__).parent.resolve()
maindir = pathlib.Path(current_dir).parent.resolve()
datasets_folder = f"{maindir}\\datasets"
agents_folder = f"{maindir}\\model\\data\\agents"

config = GetConfig(True)

class mainmenu(object):
    def __init__(self):
        # functionality
        self.experiments, self.experiment_banners = [], []
        self.tests, self.test_banners = [], []

        # parameters:
        self.is_running = False
        
        self.root = Tk()
        self.root.title("Model Generator")
        self.root.resizable(FALSE, FALSE)
        self.root.option_add('*tearOff', FALSE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # this intercepts the closing button before doing so and can, for example
        # prompt the user to save its changes before proceeding.
        self.root.protocol("WM_DELETE_WINDOW", self.intercept_close)
        # returns x11, win32 or aqua
        self.OSNAME = self.root.tk.call('tk', 'windowingsystem')

        self.mainframe = ttk.Frame(self.root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0)

        self.add_styles()
        self.add_elements()

        self.root.mainloop()

    def add_elements(self):
        # infotext
        self.infotext = ttk.Label(self.mainframe, text="0 Experiment(s) Loaded, 0 Test(s) Loaded")

        # config block
        self.config_lf = ttk.Labelframe(self.mainframe, text='Configuration')
        self.config_button = ttk.Button(self.config_lf, text="Config", 
        command = lambda: self.open_menu("config"))

        self.setup_button = ttk.Button(self.config_lf, text="Setup", 
        command = lambda: self.open_menu("setup"))

        # runner block
        self.run_lf = ttk.Labelframe(self.mainframe, text='Runner')
        self.train_button = ttk.Button(self.run_lf, text="Train", 
        command = lambda: self.reset_progbar())

        visualize = BooleanVar(value=False)
        self.visualize_check = ttk.Checkbutton(self.run_lf, text='show\nvisuals', variable=visualize)
        CreateToolTip(self.visualize_check, text="selects a path\nobtained by the\nselected agent\nand displays it.")

        self.test_button = ttk.Button(self.run_lf, text="Test", 
        command = lambda: self.run_tests(visualize.get()))

        # progressbar + text
        self.progress_text = ttk.Label(self.mainframe, text="")
        self.progress_bar = ttk.Progressbar(self.mainframe, orient= HORIZONTAL, 
        mode='determinate', length=280, maximum=100)

        # folder buttons

        self.folder_lf = ttk.Labelframe(self.mainframe, text='Folders')
        
        self.datasets_button = ttk.Button(self.folder_lf, text='Datasets', 
        command= lambda: self.open_folder("datasets"))

        self.agents_button = ttk.Button(self.folder_lf, text='Agents', 
        command= lambda: self.open_folder("agents"))

        #error text.
        self.error_text = Label(self.mainframe, text="", fg='red', bg="#33393b")

        self.grid_elements()

    def grid_elements(self):
        #row0
        self.infotext.grid(row=0, column=0, columnspan=2)

        #row1
        self.config_lf.grid(row=1, column=0)
        self.config_button.grid(row=0, column=0)
        self.setup_button.grid(row=1, column=0)
        
        for child in self.config_lf.winfo_children():
            child.grid_configure(padx=15, pady=5)

        self.run_lf.grid(row=1, column=1)
        self.train_button.grid(row=0, column=0)
        self.test_button.grid(row=1, column=0)
        self.visualize_check.grid(row=1, column=1)

        for child in self.run_lf.winfo_children():
            child.grid_configure(padx=(15,0), pady=3)

        #row2
        self.progress_text.grid(row=2, column=0, columnspan=2, pady=3)
        
        #row3
        self.progress_bar.grid(row=3, column=0, columnspan=2)

        #row4
        self.folder_lf.grid(row=4, column=0, columnspan=2)
        self.datasets_button.grid(row=0, column=0, padx=(75,0), pady=(0,5))
        self.agents_button.grid(row=0, column=1, padx=(0,75), pady=(0,5))

        #row5
        self.error_text.grid(row=5, column=0, columnspan=2, pady=3)

    def add_styles(self):
        s = ttk.Style()
        self.root.tk.call('lappend', 'auto_path', f"{current_dir}/awdark/")
        self.root.tk.call('package', 'require', 'awdark')
        s.theme_use('awdark')


    # MISC OPERATIONS

    def open_menu(self, menutype):
        if(menutype == "config"):
            config = config_menu.menu(self.root)
            
        elif(menutype == "setup"):
            setup = test_train_menu.menu(self.root, self.experiments, self.tests)
            setup.root.wm_protocol("WM_DELETE_WINDOW", lambda: self.extract_info_on_close(setup))

    def extract_info_on_close(self, setup_window):
        print(setup_window.experiments)
        print(setup_window.tests)

        self.experiments, self.tests = setup_window.experiments, setup_window.tests
        
        self.infotext["text"] = f"{len(self.experiments)} Experiment(s) Loaded, {len(self.tests)} Test(s) Loaded"

        setup_window.root.destroy()


    def open_folder(self, folder:str):
        folder_to_open = datasets_folder if folder == "datasets" else agents_folder

        print(folder_to_open)
        # x11, win32 or aqua
        if(self.OSNAME == 'x11'): #linux
            subprocess.run(['xdg-open', os.path.realpath(folder_to_open)])

        elif(self.OSNAME == 'win32'): #windows
            subprocess.run(['explorer', os.path.realpath(folder_to_open)])

        elif(self.OSNAME == 'aqua'): #mac
            subprocess.run(['open', os.path.realpath(folder_to_open)])

        else:
            self.change_error_text("This operation is not supported for this Operatig System.")
        

    def intercept_close(self):
        if self.is_running:
            print("asking the user for permission to close.")
            canclose = messagebox.askyesno(
            message='There is something running, do you want to close anyway?',
            icon='warning', title='Interruption alert.')

            if(canclose):
                self.root.destroy()
        else:
            self.root.destroy()

    def run_experimentation(self):
        try:
            self.is_running = True
            print("running experiments...")
        except:
            self.is_running = False

    def run_tests(self, show_visuals):
        try:
            self.is_running = True
            print("running testing...")
        except:
            self.is_running = False

    # texts
    def change_progtext(self, newtext:str):
        self.progress_text['text'] = newtext
    
    def change_error_text(self, newtext:str):
        self.error_text['text'] = newtext

    # progress bar
    def reset_progbar(self):
        self.progress_bar['value'] = 0

    def increase_progbar(self, qtty:int):
        self.progress_bar.step(qtty)

    def change_progbar_max(self, qtty:int):
        print(f"new progress bar maximum is {qtty}")
        self.progress_bar['maximum'] = qtty

mainmenu()