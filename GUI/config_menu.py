from tkinter import *
from tkinter import ttk

from utils import CreateToolTip
from copy import deepcopy

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
        validate='focusout', validatecommand=self.vcmd, invalidcommand=self.ivcmd)
        CreateToolTip(self.cores_entry, text="number of cores to use.")

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

        guided_rew_var = BooleanVar(value=False)
        self.guided_rew_check = ttk.Checkbutton(self.checkboxes_frame, text='guided \nrewards', variable=guided_rew_var)
        CreateToolTip(self.guided_rew_check, text="")

        regen_embs_var = BooleanVar(value=False)
        self.regen_embs_check = ttk.Checkbutton(self.checkboxes_frame, text='regenerate \nembeddings', variable=regen_embs_var)
        CreateToolTip(self.regen_embs_check, text="")

        normal_embs_var = BooleanVar(value=False)
        self.normal_embs_check = ttk.Checkbutton(self.checkboxes_frame, text='normalize \nembeddings', variable=normal_embs_var)
        CreateToolTip(self.normal_embs_check, text="")

        use_LSTM_var = BooleanVar(value=False)
        self.use_LSTM_check = ttk.Checkbutton(self.checkboxes_frame, text='use LSMT \nlayers', variable=use_LSTM_var)
        CreateToolTip(self.use_LSTM_check, text="")

        #ENTRIES
        self.entries_frame = ttk.Frame(self.training_tf)

        self.alpha_label = ttk.Label(self.entries_frame, text='')
        vcmd1 = (self.root.register(lambda value: self.validation(value, "alpha")), '%P')

        alpha_var = IntVar()
        self.alpha_entry = ttk.Entry(self.entries_frame, textvariable=coresvar, text="", validate='focusout', 
        validatecommand = vcmd1, invalidcommand = self.ivcmd)
        CreateToolTip(self.cores_entry, text="")
        
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

        for child in self.general_lf.winfo_children():
            child.grid_configure(padx=3, pady=3)

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

        # subrow6

        self.alpha_entry.grid(row=4, column=0)


        for child in self.training_tf.winfo_children():
            child.grid_configure(padx=5, pady=2)
        
        for child in self.selectors_frame.winfo_children():
            child.grid_configure(padx=5, pady=2)

        for child in self.radioframe.winfo_children():
            child.grid_configure(padx=25, pady=2)

        for child in self.checkboxes_frame.winfo_children():
            child.grid_configure(padx=5, pady=2)


    def validation(self, value, origin):
        print(value, origin)
        return False

    def invalid(self):
        print("bad")