from tkinter import *
from tkinter import ttk

from guiutils import CreateToolTip
from copy import deepcopy
import multiprocessing
import sys

class menu():
    def __init__(self, root, config):
        self.config = deepcopy(config)

        self.root = Toplevel(root)
        self.root.title('Configuration')
        self.root.resizable(FALSE, FALSE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.mainframe = ttk.Frame(self.root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0)
        
        self.add_elements()

    def add_elements(self):
        # LabelFrames:
        self.general_lf = ttk.Labelframe(self.mainframe, text='General')
        self.training_tf = ttk.Labelframe(self.mainframe, text='Training')
        self.shared_tf = ttk.Labelframe(self.mainframe, text='Shared')

        self.errors = Label(self.mainframe, text='', fg='red', bg="#FFFFFF")

        # GENERAL
        self.coreslabel = ttk.Label(self.general_lf, text='CPU cores:')
        vcmd = (self.root.register(lambda value: self.validation(value, "cpu")), '%P')
        ivcmd = (self.root.register(lambda: self.invalid("cpu")),)
        coresvar = IntVar()
        self.cores_entry = ttk.Entry(self.general_lf, textvariable=coresvar, text="cores",
        validate='focusout', validatecommand=vcmd, invalidcommand=ivcmd)
        CreateToolTip(self.cores_entry, text="number of cores to use.")
        self.cores_entry.delete(0, 'end')
        self.cores_entry.insert(0, self.config["available_cores"])

        use_gpu = BooleanVar(value=self.config["gpu_acceleration"])
        self.gpu_check = ttk.Checkbutton(self.general_lf, text='use gpu?', variable=use_gpu)
        CreateToolTip(self.gpu_check, text="allows for gpu to be \nused when running models.")
        
        verbose = BooleanVar(value=self.config["verbose"])
        self.verb_check = ttk.Checkbutton(self.general_lf, text='vebatim?', variable=verbose)
        CreateToolTip(self.verb_check, text="prints to the terminal the\n progress of each episode")
        
        logs = BooleanVar(value=self.config["log_results"])
        self.logs_check = ttk.Checkbutton(self.general_lf, text='create logs?', variable=logs)
        CreateToolTip(self.logs_check, text="generate logs for the training")
        
        debug = BooleanVar(value=self.config["debug"])
        self.debug_check = ttk.Checkbutton(self.general_lf, text='run debug?', variable=debug)
        CreateToolTip(self.debug_check, text="if the program crashes\nit runs postmortem debug.")

        # TRAINING

        # RADIOBUTTONS
        self.radioframe = ttk.Frame(self.training_tf)

        self.radio1frame = ttk.Frame(self.radioframe)
        CreateToolTip(self.radio1frame, text="The algorithm to train the network on:\n\
        PPO-> Proximal Policy Optimization\n\
        BASE-> Baseline RL algorithm")

        self.algo_var = StringVar(None, self.config["algorithm"])
        self.algo_label = ttk.Label(self.radio1frame, text='Algorithm')
        self.rad_ppo = ttk.Radiobutton(self.radio1frame, text='PPO', variable=self.algo_var, value='PPO') 
        self.rad_base = ttk.Radiobutton(self.radio1frame, text='BASE', variable=self.algo_var, value='BASE') 
        
        self.radio2frame = ttk.Frame(self.radioframe)
        CreateToolTip(self.radio2frame, text="Which reward type to use when passing it to the network.\n\
        Simple-> Feeds the calculated rewards directly to the agent\n\
        BackProp-> backpropagates the rewards enphasizing the latest actions taken")

        self.rew_type_var = StringVar(None, self.config["reward_type"])
        self.reward_type_label = ttk.Label(self.radio2frame, text='Reward type')
        self.rad_simp = ttk.Radiobutton(self.radio2frame, text='Simple', variable=self.rew_type_var, value='simple') 
        self.rad_retro = ttk.Radiobutton(self.radio2frame, text='BackProp', variable=self.rew_type_var, value='retropropagation') 
        
        self.radio3frame = ttk.Frame(self.radioframe)
        CreateToolTip(self.radio3frame, text="How to compute the rewards with the netowork output.\n\
            Max% -> \% of performance compared to the top action in the step\n\
            One Hot Max-> 1 for best performers, 0 otherwise.\n\
            Straight -> Feed the output of the network directly as output is [0-1]")

        self.rew_comp_var = StringVar(None, self.config["reward_computation"])
        self.rew_comp_label = ttk.Label(self.radio3frame, text='Reward computing')
        self.rad_max_p = ttk.Radiobutton(self.radio3frame, text='Max %', variable=self.rew_comp_var, value='max_percent') 
        self.rad_ohm = ttk.Radiobutton(self.radio3frame, text='One Hot Max', variable=self.rew_comp_var, value='one_hot_max') 
        self.rad_str = ttk.Radiobutton(self.radio3frame, text='Straight', variable=self.rew_comp_var, value='straight') 

        self.sep1 = ttk.Separator(self.training_tf, orient='horizontal')

        # SELECTORS
        self.selectors_frame = ttk.Frame(self.training_tf)

        self.activation_label = ttk.Label(self.selectors_frame, text='Activation function')
        possible_activations = ["relu", "prelu", "leaky_relu", "elu", "tanh"]
        activation = StringVar(value=possible_activations)
        self.activation_listbox = Listbox(self.selectors_frame, listvariable=activation, height=5, exportselection=False)
        self.activation_listbox.select_set(possible_activations.index(self.config['activation']))
        CreateToolTip(self.activation_label, text="which activation function to use for intermediate layers.")


        self.regularizers_label = ttk.Label(self.selectors_frame, text='Regularization')
        possible_regularizers = ["kernel", "bias", "activity"]
        regularizers = StringVar(value=possible_regularizers)
        self.regularizer_listbox = Listbox(self.selectors_frame, listvariable=regularizers, height=3, selectmode='multiple', exportselection=False)
        for r in self.config['regularizers']:
            self.regularizer_listbox.select_set(possible_regularizers.index(r))
        CreateToolTip(self.regularizers_label, text="which regularization to apply:\n\
            Kernel-> \n\
            Bias-> \n\
            Activity-> \n")


        self.guided_rew_label = ttk.Label(self.selectors_frame, text='Active Rewards')
        possible_rewards = ["distance","terminal","embedding"]
        rewards = StringVar(value=possible_rewards)
        self.rewards_listbox = Listbox(self.selectors_frame, listvariable=rewards, height=3, selectmode='multiple', exportselection=False)
        for r in self.config['guided_to_compute']:
            self.rewards_listbox.select_set(possible_rewards.index(r))
        CreateToolTip(self.guided_rew_label, text="which rewards to use:\n\
            Distance-> \n\
            Embedding-> \n\
            Terminal-> \n")

        self.sep2 = ttk.Separator(self.training_tf, orient='horizontal')

        # CHECKBOXES
        self.checkboxes_frame = ttk.Frame(self.training_tf)

        guided_rew_var = BooleanVar(value=self.config["guided_reward"])
        self.guided_rew_check = ttk.Checkbutton(self.checkboxes_frame, text='guided \nrewards', variable=guided_rew_var)
        CreateToolTip(self.guided_rew_check, text="")

        regen_embs_var = BooleanVar(value=self.config["regenerate_embeddings"])
        self.regen_embs_check = ttk.Checkbutton(self.checkboxes_frame, text='regenerate \nembeddings', variable=regen_embs_var)
        CreateToolTip(self.regen_embs_check, text="")

        normal_embs_var = BooleanVar(value=self.config["normalize_embeddings"])
        self.normal_embs_check = ttk.Checkbutton(self.checkboxes_frame, text='normalize \nembeddings', variable=normal_embs_var)
        CreateToolTip(self.normal_embs_check, text="")

        use_LSTM_var = BooleanVar(value=self.config["use_LSTM"])
        self.use_LSTM_check = ttk.Checkbutton(self.checkboxes_frame, text='use LSMT \nlayers', variable=use_LSTM_var)
        CreateToolTip(self.use_LSTM_check, text="")

        self.sep3 = ttk.Separator(self.training_tf, orient='horizontal')

        #ENTRIES
        self.entries_frame = ttk.Frame(self.training_tf)

        self.alpha_label = ttk.Label(self.entries_frame, text='Alpha')
        vcmd1 = (self.root.register(lambda value: self.validation(value, "alpha")), '%P')
        ivcmd1 = (self.root.register(lambda: self.invalid("alpha")),)
        alpha_var = IntVar()
        self.alpha_entry = ttk.Entry(self.entries_frame, textvariable=alpha_var, text="alpha", validate='focusout', 
        validatecommand = vcmd1, invalidcommand = ivcmd1)
        CreateToolTip(self.alpha_entry, text="(0.8-0.99) previous step network learnin rate (PPO only).")
        self.alpha_entry.delete(0, 'end')
        self.alpha_entry.insert(0, self.config["alpha"])

        self.gamma_label = ttk.Label(self.entries_frame, text='Gamma')
        vcmd2 = (self.root.register(lambda value: self.validation(value, "gamma")), '%P')
        ivcmd2 = (self.root.register(lambda: self.invalid("gamma")),)
        gamma_var = IntVar()
        self.gamma_entry = ttk.Entry(self.entries_frame, textvariable=gamma_var, text="gamma", validate='focusout', 
        validatecommand = vcmd2, invalidcommand = ivcmd2)
        CreateToolTip(self.gamma_entry, text="(0.9-0.99) decay rate of past observations in backprop reward.")
        self.gamma_entry.delete(0, 'end')
        self.gamma_entry.insert(0, self.config["gamma"])

        self.lr_label = ttk.Label(self.entries_frame, text='Learning Rate')
        vcmd3 = (self.root.register(lambda value: self.validation(value, "lr")), '%P')
        ivcmd3 = (self.root.register(lambda: self.invalid("lr")),)
        lr_var = IntVar()
        self.lr_entry = ttk.Entry(self.entries_frame, textvariable=lr_var, text="lr", validate='focusout', 
        validatecommand = vcmd3, invalidcommand = ivcmd3)
        CreateToolTip(self.lr_entry, text="(1e-3 - 1e-5) neural network learning rate.")
        self.lr_entry.delete(0, 'end')
        self.lr_entry.insert(0, self.config["learning_rate"])


        #SHARED

        self.path_length = ttk.Label(self.shared_tf, text='Path Length')
        vcmd4 = (self.root.register(lambda value: self.validation(value, "path")), '%P')
        ivcmd4 = (self.root.register(lambda: self.invalid("path")),)
        path_var = IntVar()
        self.path_entry = ttk.Entry(self.shared_tf, textvariable=path_var, text="path", validate='focusout', 
        validatecommand = vcmd4, invalidcommand = ivcmd4)
        CreateToolTip(self.path_entry, text="(3-10) discovered path length.")
        self.path_entry.delete(0, 'end')
        self.path_entry.insert(0, self.config["path_length"])

        random_seed_var = BooleanVar(value=self.config["random_seed"])
        self.random_seed_check = ttk.Checkbutton(self.shared_tf, text='random seed?', variable=random_seed_var)
        CreateToolTip(self.random_seed_check, text="if set, uses a random seed")


        self.seed_label = ttk.Label(self.shared_tf, text='Set Seed')
        vcmd5 = (self.root.register(lambda value: self.validation(value, "seed")), '%P')
        ivcmd5 = (self.root.register(lambda: self.invalid("seed")),)
        set_seed_var = IntVar()
        self.seed_entry = ttk.Entry(self.shared_tf, textvariable=set_seed_var, text="seed", validate='focusout', 
        validatecommand = vcmd5, invalidcommand = ivcmd5)
        CreateToolTip(self.seed_entry, text="(int) if random seed is not set, uses the specified value.")
        self.seed_entry.delete(0, 'end')
        self.seed_entry.insert(0, self.config["seed"])

        self.save_button = ttk.Button(self.mainframe, text="Save", 
        command = self.save_config)

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

        ########
        # row2 #
        ########
        self.training_tf.grid(row=2, column=0)

        # subrow0
        self.checkboxes_frame.grid(row=0, column=0)
        
        self.guided_rew_check.grid(row=1, column=0)
        self.regen_embs_check.grid(row=1, column=1)
        self.normal_embs_check.grid(row=1, column=2)
        self.use_LSTM_check.grid(row=1, column=3)

        # subrow1
        self.sep2.grid(row=1, column=0, columnspan=5, sticky="we")

        # subrow2
        self.radioframe.grid(row=2, column=0, columnspan=2)

        self.radio1frame.grid(row=0, column=0)
        self.radio2frame.grid(row=0, column=1)
        self.radio3frame.grid(row=0, column=2)

        #col0
        self.algo_label.grid(row=0, column=0)
        self.rad_ppo.grid(row=1, column=0)
        self.rad_base.grid(row=2, column=0)

        #col1
        self.reward_type_label.grid(row=0, column=0)
        self.rad_simp.grid(row=1, column=0)
        self.rad_retro.grid(row=2, column=0)

        #col2
        self.rew_comp_label.grid(row=0, column=0)
        self.rad_max_p.grid(row=1, column=0)
        self.rad_ohm.grid(row=2, column=0)
        self.rad_str.grid(row=3, column=0)

        # subrow3
        self.sep1.grid(row=3, column=0, columnspan=5, sticky="we")

        # subrow4
        self.selectors_frame.grid(row=4, column=0)

        self.activation_label.grid(row=0, column=0)
        self.activation_listbox.grid(row=1, column=0)

        self.regularizers_label.grid(row=0, column=1)
        self.regularizer_listbox.grid(row=1, column=1)

        self.guided_rew_label.grid(row=0, column=2)
        self.rewards_listbox.grid(row=1, column=2)

        # subrow5
        self.sep3.grid(row=5, column=0, sticky="we")

        # subrow6

        self.entries_frame.grid(row=6, column=0)

        #col0
        self.alpha_label.grid(row=0, column=0)
        self.alpha_entry.grid(row=1, column=0)

        #col1
        self.gamma_label.grid(row=0, column=1)
        self.gamma_entry.grid(row=1, column=1)

        #col1
        self.lr_label.grid(row=0, column=2)
        self.lr_entry.grid(row=1, column=2)

        ########
        # row3 #
        ########
        self.shared_tf.grid(row=3, column=0)

        #col0
        self.path_length.grid(row=0, column=0)
        self.path_entry.grid(row=1, column=0)

        #col1
        self.random_seed_check.grid(row=0, column=1, rowspan=2)

        #col2
        self.seed_label.grid(row=0, column=2)
        self.seed_entry.grid(row=1, column=2)

        # row4
        self.save_button.grid(row=4, column=0, sticky="ns", pady=15)

        #Extra padding

        for child in self.general_lf.winfo_children():
            child.grid_configure(padx=3, pady=3)

        for child in self.training_tf.winfo_children():
            child.grid_configure(padx=5, pady=2)
        
        for child in self.selectors_frame.winfo_children():
            child.grid_configure(padx=5, pady=2)

        for child in self.radioframe.winfo_children():
            child.grid_configure(padx=25, pady=2)

        for child in self.checkboxes_frame.winfo_children():
            child.grid_configure(padx=5, pady=2)

        for child in self.entries_frame.winfo_children():
            child.grid_configure(padx=3, pady=2)

        for child in self.shared_tf.winfo_children():
            child.grid_configure(padx=9, pady=0)

    def validation(self, value: str, origin):
        int_origins, float_origins = ["path", "cpu", "seed"], ["alpha", "gamma", "lr"]
        all_origins = [*int_origins,*float_origins]
        ranges = [(3,10),(1,multiprocessing.cpu_count()),(1,sys.maxsize), (0.9,0.99),(0.8,0.99),(1e-5, 1e-3)]

        if(value.replace('.','',1).isdigit()): #check for any non-negative number
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
            self.errors["text"] = f"{origin} must in range [{a}-{b}]"
            return False
        else:
            return True

    def invalid(self, origin):
        print(f"the origin of the validation error is: {origin}")
        if(origin == "seed"):  
            self.seed_entry.delete(0,END)          
            self.seed_entry.insert(0, str(self.config["seed"]))

        elif(origin == "path"):
            self.path_entry.delete(0,END)          
            self.path_entry.insert(0, str(self.config["path_length"]))
        
        elif(origin == "alpha"):
            self.alpha_entry.delete(0,END)          
            self.alpha_entry.insert(0, str(self.config["alpha"]))

        elif(origin == "gamma"):
            self.gamma_entry.delete(0,END)          
            self.gamma_entry.insert(0, str(self.config["gamma"]))
        
        elif(origin == "lr"):
            self.lr_entry.delete(0,END)          
            self.lr_entry.insert(0, str(self.config["learning_rate"]))

        elif(origin == "cpu"):
            self.cores_entry.delete(0,END)          
            self.cores_entry.insert(0, str(self.config["available_cores"]))

        else:         
            self.errors["text"] = "an unexpected error ocurred"
        
    def save_config(self):
        print("saving config")
        self.config["available_cores"] = self.cores_entry.get()
        self.config["gpu_acceleration"] = self.gpu_check.state()[0] == 'selected'
        self.config["verbose"] = self.verb_check.state()[0] == 'selected'
        self.config["log_results"] = self.logs_check.state()[0] == 'selected'
        self.config["debug"] = self.debug_check.state()[0] == 'selected'
        self.config["print_layers"] = False

        self.config["restore_agent"] = False
        self.config["guided_reward"] = self.guided_rew_check.state()[0] == 'selected'
        self.config["guided_to_compute"] = [self.rewards_listbox.get(idx) for idx in self.rewards_listbox.curselection()]
        self.config["regenerate_embeddings"] = self.regen_embs_check.state()[0] == 'selected'
        self.config["normalize_embeddings"] = self.normal_embs_check.state()[0] == 'selected'
        self.config["use_LSTM"] = self.use_LSTM_check.state()[0] == 'selected'

        self.config["use_episodes"] = False
        self.config["episodes"] = 0

        self.config["alpha"] = self.alpha_entry.get()
        self.config["gamma"] = self.gamma_entry.get()
        self.config["learning_rate"] = self.lr_entry.get()

        self.config["activation"] = self.activation_listbox.get(ACTIVE)
        self.config["regularizers"] = [self.regularizer_listbox.get(idx) for idx in self.regularizer_listbox.curselection()]

        self.config["algorithm"] = self.algo_var.get()
        self.config["reward_type"] = self.rew_type_var.get()
        self.config["action_picking_policy"] = "probability"
        self.config["reward_computation"] = self.rew_comp_var.get()

        self.config["path_length"] = self.path_entry.get()
        self.config["random_seed"] = self.random_seed_check.state()[0] == 'selected'
        self.config["seed"] = self.seed_entry.get()

        self.errors["text"] = "data saved succesfully!"