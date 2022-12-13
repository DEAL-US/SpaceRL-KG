from tkinter import *
from tkinter import ttk

import multiprocessing
import sys

class menu():
    def __init__(self, root):
        self.root = Toplevel(root)
        self.root.title('View Paths')
        self.root.resizable(FALSE, FALSE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.mainframe = ttk.Frame(self.root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0)
        
        self.add_elements()

    def add_elements(self):
        self.guided_rew_label = ttk.Label(self.mainframe, text='Select test')
        possible_rewards = []

        
        rewards = StringVar(value=possible_rewards)
        self.rewards_listbox = Listbox(self.selectors_frame, listvariable=rewards, height=3, selectmode='multiple', exportselection=False)
        for r in self.config['guided_to_compute']:
            self.rewards_listbox.select_set(possible_rewards.index(r))
        
        self.grid_elements()

    def grid_elements(self):
       pass


    def validation(self, value: str, origin):
        int_origins, float_origins = ["path", "cpu", "seed"], ["alpha", "gamma", "lr"]
        all_origins = [*int_origins,*float_origins]
        ranges = [(3,10),(1,multiprocessing.cpu_count()),(1,sys.maxsize), (0.9,0.99),(0.8,0.99),(1e-3, 1e-5)]
        
        if(value.isnumeric()):
            if(origin in int_origins):
                v = int(value)
            elif(origin in float_origins):
                v = float(value)
            else:
                print(f"bad origin {origin}")
                return False
        else:
            self.errors["text"] = f"{origin} must be a number"
            return False
            
        o_index = all_origins.index(origin)
        a, b = ranges[o_index][0], ranges[o_index][1]

        if(v < a or v > b):
            self.errors["text"] = f"{origin} must in range {a}-{b}"
            return True
        else:
                return False
           

    def invalid(self, origin):
        print(origin)
        if(origin == "seed"):            
            self.seed_entry.insert(0, str(self.config["seed"]))

        elif(origin == "path"):
            self.path_entry.insert(0, str(self.config["path_length"]))
        
        elif(origin == "alpha"):
            self.alpha_entry.insert(0, str(self.config["alpha"]))

        elif(origin == "gamma"):
            self.gamma_entry.insert(0, str(self.config["gamma"]))
        
        elif(origin == "lr"):
            self.lr_entry.insert(0, str(self.config["learning_rate"]))

        elif(origin == "cpu"):
            self.cores_entry.insert(0, str(self.config["available_cores"]))

        else:
            self.errors["text"] = "an unexpected error ocurred"
        

    def save_config(self, close):
        print("saved")
        self.modfications_saved = True

    def watch_variables(self, *vars):
        print(vars)
        for var in vars:
            var.trace_add("write", self.modification_happened)

    def modification_happened(self, *vars):
        print("prompter for change save")
        self.modfications_saved = False
