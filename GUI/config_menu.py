from tkinter import *
from tkinter import ttk

from utils import CreateToolTip
from copy import deepcopy

class menu():
    def __init__(self, root, config):
        self.config = deepcopy(config)

        self.root = Toplevel(root)
        self.root.resizable(FALSE, FALSE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.mainframe = ttk.Frame(self.root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0)

        self.vcmd = (self.root.register(self.validation), '%P')
        self.ivcmd = (self.root.register(self.invalid),)
        
        self.add_elements()

    def add_elements(self):
        # LabelFrames:
        self.general_lf = ttk.Labelframe(self.mainframe, text='General')
        self.training_tf = ttk.Labelframe(self.mainframe, text='Training')
        self.shared_tf = ttk.Labelframe(self.mainframe, text='Shared')

        self.errors = Label(self.mainframe, text='', fg='red', bg="#33393b")

        # GENERAL
        self.coreslabel = ttk.Label(self.general_lf, text='CPU cores:')

        coresvar = IntVar()
        self.cores_entry = ttk.Entry(self.general_lf, textvariable=coresvar, text="cores",
        validate='key', validatecommand=self.vcmd, invalidcommand=self.ivcmd)
        CreateToolTip(self.cores_entry, text="number of cores to use.")

        use_gpu = BooleanVar(value=self.config["gpu_acceleration"])
        self.gpu_check = ttk.Checkbutton(self.general_lf, text='use gpu?', variable=use_gpu)
        CreateToolTip(self.gpu_check, text="allows for gpu\nto be used\nwhen running models.")
        
        verbose = BooleanVar(value=self.config["verbose"])
        self.verb_check = ttk.Checkbutton(self.general_lf, text='vebatim?', variable=verbose)
        CreateToolTip(self.verb_check, text="prints to the terminal\nthe progress of\neach episode")
        
        logs = BooleanVar(value=self.config["log_results"])
        self.logs_check = ttk.Checkbutton(self.general_lf, text='create logs?', variable=logs)
        CreateToolTip(self.logs_check, text="generate logs for the training")
        
        debug = BooleanVar(value=self.config["debug"])
        self.debug_check = ttk.Checkbutton(self.general_lf, text='run debug?', variable=debug)
        CreateToolTip(self.debug_check, text="if the program crashes\nit runs postmortem debug.")
        
        self.grid_elements()

    def grid_elements(self):
        #row0:
        self.errors.grid(row=0, column=0)
        
        ########
        # row1 #
        ########
        self.general_lf.grid(row=1, column=0)

        #subrow0:
        self.coreslabel.grid(row=0, column=0)
        self.cores_entry.grid(row=0, column=1)
        self.gpu_check.grid(row=0, column=2)

        #subrow1:
        self.verb_check.grid(row=1, column=0)
        self.logs_check.grid(row=1, column=1)
        self.debug_check.grid(row=1, column=2)

    def validation(self, value):
        pass

    def invalid(self):
        pass