from tkinter import *
from tkinter import ttk

from guiutils import AgentInfo, ExperimentBanner, GetDatasets, GetAgents, CheckForRelationInDataset
from guiutils import CheckAgentNameColision, CheckTestCollision, GetExperimentInstance, GetTestInstance

class menu():
    def __init__(self, root, experiments, tests):
        self.root = Toplevel(root)
        self.root.title('Setup')
        self.root.resizable(FALSE, FALSE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.train_index, self.test_index = len(experiments), len(tests)
        self.experiments, self.tests = experiments, tests
        self.experiment_banners, self.test_banners = [], []

        self.vcmd = (self.root.register(self.ValidateRange), '%P')
        self.ivcmd = (self.root.register(self.InvalidInput),)

        self.agents = dict()
        for a in GetAgents():
            self.agents[a.name] = a

        self.add_elements()

    def add_elements(self):
        self.n = ttk.Notebook(self.root)

        # pages, we grid elements inside them
        self.trainframe = ttk.Frame(self.n)
        self.testframe = ttk.Frame(self.n)

        self.n.add(self.trainframe, text='Train')
        self.n.add(self.testframe, text='Test')

        self.n.grid()
        self.add_train_elements()
        self.add_test_elements()

    def add_train_elements(self):
        # Create a canvas object and a vertical scrollbar for scrolling it.
        self.experiments_frame_scrollbar = ttk.Scrollbar(self.trainframe)
        self.experiment_canvas = Canvas(self.trainframe, bd=2, 
        height=200, bg='#33393b', highlightthickness=0)
        self.experiments_frame_scrollbar.config(command = self.experiment_canvas.yview)

        # Reset the view
        self.experiment_canvas.xview_moveto(0)
        self.experiment_canvas.yview_moveto(0)
        
        # create frame and add to canvas
        self.experiments_frame = ttk.Frame(self.trainframe, padding= "12 0 0 0")
        self.experiment_canvas.create_window(0, 0, window=self.experiments_frame, anchor='nw')
        self.experiment_canvas.config(yscrollcommand = self.experiments_frame_scrollbar.set,
        scrollregion = (0, 0, 150, 150))

        # scrolling
        self.experiments_frame_scrollbar.lift(self.experiments_frame)
        self.experiments_frame.bind('<Configure>', lambda event: self._configure_window(event, self.experiment_canvas, self.experiments_frame))
        self.experiments_frame.bind('<Enter>', lambda event: self._bound_to_mousewheel(event, self.experiment_canvas))
        self.experiments_frame.bind('<Leave>', lambda event: self._unbound_to_mousewheel(event, self.experiment_canvas))

        self.t_sep = ttk.Separator(self.trainframe, orient='vertical')

        self.namelabel = ttk.Label(self.trainframe, text=f'name:')
        self.lapslabel = ttk.Label(self.trainframe, text=f'laps:')
        self.datasetlabel = ttk.Label(self.trainframe, text=f'Dataset')
        self.embeddingslabel = ttk.Label(self.trainframe, text=f'Embedding')
        self.relationnamelabel = ttk.Label(self.trainframe, text=f'relation name')

        namevar = StringVar()
        self.name_entry = ttk.Entry(self.trainframe, textvariable=namevar, text="name")

        lapsvar = IntVar()
        self.laps_entry = ttk.Entry(self.trainframe, textvariable=lapsvar, text="laps",
        validate='key', validatecommand= self.vcmd, invalidcommand=self.ivcmd)
        self.laps_entry.delete(0, 'end')
        self.laps_entry.insert(0, 10)

        embeddings = ["TransE_l2", "DistMult", "ComplEx", "TransR"]
        choices_emb = StringVar(value=embeddings)
        self.embedlistbox = Listbox(self.trainframe, listvariable=choices_emb, height=4, selectmode='multiple', exportselection=False)

        datasets = GetDatasets()
        choices_datasets = StringVar(value=["--------", *datasets])
        self.datasetlistbox = Listbox(self.trainframe, listvariable=choices_datasets, height=4, exportselection=False)
        self.datasets_scrollbar = ttk.Scrollbar(self.trainframe)
        self.datasetlistbox.config(yscrollcommand=self.datasets_scrollbar.set)
        self.datasets_scrollbar.config(command=self.datasetlistbox.yview)

        singlecheckvar = BooleanVar(value=False)
        self.single_check = ttk.Checkbutton(self.trainframe, text='is single\nrelation', variable=singlecheckvar)

        singletextvar = StringVar()
        self.single_entry = ttk.Entry(self.trainframe, textvariable=singletextvar, text="single")
        
        self.add_to_list_train = ttk.Button(self.trainframe, text="add", 
        command=lambda: self.add_to_list("train"))

        self.remove_last_train = ttk.Button(self.trainframe, text="remove last", 
        command=lambda: self.remove_from_list("train"))

        self.error_text_train = Label(self.trainframe, text="", fg='red', bg="#33393b")

        self.grid_trainframe()

    def grid_trainframe(self):
        #row0
        self.namelabel.grid(row=0, column=0)
        self.name_entry.grid(row=0, column=1)
        self.t_sep.grid(row=0, column=2, rowspan=30, sticky="ns")
        self.experiment_canvas.grid(row=0, column=3, rowspan=30, sticky='ne')
        self.experiments_frame_scrollbar.grid(row=0, column=4, rowspan=30, sticky="ns")

        #row1
        self.lapslabel.grid(row=1, column=0)
        self.laps_entry.grid(row=1, column=1)

        #row2
        self.datasetlabel.grid(row=2, column=0)
        self.embeddingslabel.grid(row=2, column=1)

        #row3
        self.datasetlistbox.grid(row=3, column=0, padx=(0,20))
        self.datasets_scrollbar.place(x = 165, y = 69, height=73)

        self.embedlistbox.grid(row=3, column=1,)
        # self.embedding_scrollbar.place(x = 268, y = 65)
        
        #row4
        self.single_check.grid(row=4, column=0, rowspan=2)
        self.relationnamelabel.grid(row=4, column=1)

        #row5
        self.single_entry.grid(row=5, column=1)

        #row6
        self.add_to_list_train.grid(row=6, column=0, padx=(75,0), pady=5)
        self.remove_last_train.grid(row=6, column=1, padx=(0,75), pady=5)

        #row7
        self.error_text_train.grid(row=7, column=0, columnspan=2)

        for i, e in enumerate(self.experiments):            
            e_banner = ExperimentBanner(self.experiments_frame, 
            f"experiment {i}", e.name, e.laps, e.dataset, e.embeddings,
            e.single_relation, e.relation_to_train)
            banner = e_banner.getbanner()
            banner.grid(row=i, column=0)
    
    def add_test_elements(self):
        # Create a canvas object and a vertical scrollbar for scrolling it.
        self.test_frame_scrollbar = ttk.Scrollbar(self.testframe)
        self.test_canvas = Canvas(self.testframe, bd=2, 
        height=200, bg='#33393b', highlightthickness=0)
        self.test_frame_scrollbar.config(command = self.test_canvas.yview)

        # Reset the view
        self.test_canvas.xview_moveto(0)
        self.test_canvas.yview_moveto(0)
        
        # create frame and add to canvas
        self.test_frame = ttk.Frame(self.testframe, padding= "12 0 0 0")
        self.test_canvas.create_window(0, 0, window=self.test_frame, anchor='nw')
        self.test_canvas.config(yscrollcommand = self.test_frame_scrollbar.set,
        scrollregion = (0, 0, 150, 150))

        # scrolling
        self.test_frame_scrollbar.lift(self.test_frame)
        self.test_frame.bind('<Configure>', lambda event: self._configure_window(event, self.test_canvas, self.test_frame))
        self.test_frame.bind('<Enter>', lambda event: self._bound_to_mousewheel(event, self.test_canvas))
        self.test_frame.bind('<Leave>', lambda event: self._unbound_to_mousewheel(event, self.test_canvas))

        self.test_sep = ttk.Separator(self.testframe, orient='vertical')

        # labels
        self.t_namelabel = ttk.Label(self.testframe, text=f'name:')
        self.t_lapslabel = ttk.Label(self.testframe, text=f'episodes:') 
        self.t_agentslabel = ttk.Label(self.testframe, text=f'Agents')
        self.t_embeddingslabel = ttk.Label(self.testframe, text=f'Embeddings')       

        namevar = StringVar()
        self.t_name_entry = ttk.Entry(self.testframe, textvariable=namevar, text="name")

        runsvar = IntVar()
        self.t_runs_entry = ttk.Entry(self.testframe, textvariable=runsvar, text="runs",
        validate='key', validatecommand= self.vcmd, invalidcommand=self.ivcmd)
        self.t_runs_entry.delete(0, 'end')
        self.t_runs_entry.insert(0, 100)

        #listboxes
        self.t_embeddings = []
        self.t_choices_emb = StringVar(value=self.t_embeddings)
        self.t_embedlistbox = Listbox(self.testframe, listvariable=self.t_choices_emb, height=4, selectmode='multiple', exportselection=False)

        # print(list(self.agents.keys()))
        choices_agents = StringVar(value=list(self.agents.keys()))
        self.agentlistbox = Listbox(self.testframe, listvariable=choices_agents, height=4, exportselection=False)
        self.agents_scrollbar = ttk.Scrollbar(self.testframe)
        self.agentlistbox.config(yscrollcommand=self.agents_scrollbar.set)
        self.agents_scrollbar.config(command=self.agentlistbox.yview)
        self.agentlistbox.bind("<<ListboxSelect>>", lambda event: self.populate_embedding_listbox_test(event))

        self.add_to_list_test = ttk.Button(self.testframe, text="add", 
        command=lambda: self.add_to_list("test"))

        self.remove_last_test = ttk.Button(self.testframe, text="remove last", 
        command=lambda: self.remove_from_list("test"))

        self.error_text_test = Label(self.testframe, text="", fg='red', bg="#33393b")

        self.grid_testframe()
    
    def grid_testframe(self):
        #row0
        self.test_sep.grid(row=0, column=2, rowspan=30, sticky="ns")
        self.test_canvas.grid(row=0, column=3, rowspan=30, sticky='ne')
        self.test_frame_scrollbar.grid(row=0, column=4, rowspan=30, sticky="ns")

        self.t_namelabel.grid(row=0, column=0)
        self.t_name_entry.grid(row=0, column=1)

        #row1
        self.t_lapslabel.grid(row=1, column=0)
        self.t_runs_entry.grid(row=1, column=1)

        #row2
        self.t_agentslabel.grid(row=2, column=0)
        self.t_embeddingslabel.grid(row=2, column=1)

        #row3
        self.agentlistbox.grid(row=3, column=0, padx=(0,20))
        self.agents_scrollbar.place(x = 165, y = 69, height=73)

        self.t_embedlistbox.grid(row=3, column=1)

        #row4
        self.add_to_list_test.grid(row=4, column=0, padx=(75,0), pady=5)
        self.remove_last_test.grid(row=4, column=1, padx=(0,75), pady=5)

        #row5
        self.error_text_test.grid(row=5, column=0, columnspan=2)

        for i, e in enumerate(self.tests):            
            e_banner = ExperimentBanner(self.test_frame, 
            f"test {i}", e.name, e.episodes, e.dataset, e.embeddings,
            e.single_relation, e.relation_to_train, lapstext = "runs")
            banner = e_banner.getbanner()
            banner.grid(row=i, column=0)
    
    # MISC button functions
    def add_to_list(self, from_frame:str):
        if(from_frame == "train"):
            error_text = ""

            name = self.name_entry.get()
            laps = int(self.laps_entry.get())
            dataset = self.datasetlistbox.get(ACTIVE)
            embeddings = [self.embedlistbox.get(idx) for idx in self.embedlistbox.curselection()]
            try:
                e = self.single_check.state()[0]
                if(e=="alternate"):
                    e = False
                elif(e=="selected"):
                    e=True
            except:
                e = False
            
            rel_name = self.single_entry.get()

            # Validation:
            if(name == ""):
                error_text += "name must not be empty\n"

            if(CheckAgentNameColision(name)):
                error_text += "name collides with existing experiment.\n"

            
            if(laps<10 or laps >999):
                error_text += "laps range is 10-999\n"

            if(dataset == "--------"):
                error_text += "no dataset selected\n"

            for ex in self.experiments:
                if(ex.name == name):
                    error_text += "choose a name which not in the experiment list.\n"

            if(len(embeddings) == 0):
                error_text += "no embeddings selected\n"

            if(e):
                if(not CheckForRelationInDataset(dataset, rel_name)):
                    error_text += f"the specified relation {rel_name}\
                    \n does not exist in dataset {dataset}\n"

            if(error_text != ""):
                self.error_text_train["text"] = error_text
            
            else:
                e_banner = ExperimentBanner(self.experiments_frame, 
                f"experiment {self.train_index}", name, laps, dataset, embeddings, e, rel_name)

                banner = e_banner.getbanner()
                banner.grid(row=self.train_index,column=0)

                self.experiment_banners.append(banner)

                exp = GetExperimentInstance(name, dataset, embeddings, laps, e, rel_name)
                self.experiments.append(exp)

                self.train_index +=1

        if(from_frame == "test"):
            error_text = ""

            name = self.t_name_entry.get()
            runs = int(self.t_runs_entry.get())

            i = self.agentlistbox.curselection()[0]
            active = self.agentlistbox.get(i)
            agent = self.agents[active].get()

            dataset = agent[2]
            embeddings = [self.t_embedlistbox.get(idx) for idx in self.t_embedlistbox.curselection()]
            e = agent[3]   
            rel_name = agent[4]

            # Validation:
            if(name == ""):
                error_text += "name must not be empty\n"

            if(CheckTestCollision(name)):
                error_text += "name collides with existing test.\n"

            if(runs<100 or runs >9999):
                error_text += "runs range is 100-9999\n"

            for ex in self.tests:
                if(ex.name == name):
                    error_text += "choose a name which not in the test list.\n"

            if(len(embeddings) == 0):
                error_text += "no embeddings selected\n"


            if(error_text != ""):
                self.error_text_test["text"] = error_text
            else:
                e_banner = ExperimentBanner(self.test_frame, 
                f"experiment {self.test_index}", name, runs, dataset,
                embeddings, e, rel_name, lapstext="runs")

                banner = e_banner.getbanner()
                banner.grid(row=self.test_index,column=0)

                self.experiment_banners.append(banner)

                test = GetTestInstance(agent[0], name, embeddings, runs)
                self.tests.append(test)

                self.test_index +=1

    def remove_from_list(self, from_frame:str):
        if(from_frame == "train"):
            latest_b = self.experiment_banners.pop()
            self.experiments.pop()
            latest_b.destroy()
            self.train_index -= 1
        else:
            pass
    

    def populate_embedding_listbox_test(self, event):
        i = self.agentlistbox.curselection()[0]
        active = self.agentlistbox.get(i)
        embeddings = self.agents[active].get()[1]
        self.t_embedlistbox.delete(0,END)
        for i, e in enumerate(embeddings):
            self.t_embedlistbox.insert(i, e)

    # MISC SCROLLABLES
    def _bound_to_mousewheel(self, event, canvas):
        canvas.bind_all("<MouseWheel>", lambda event: self._on_mousewheel(event, canvas))   

    def _unbound_to_mousewheel(self, event, canvas):
        canvas.unbind_all("<MouseWheel>") 

    def _on_mousewheel(self, event, canvas):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")  

    def _configure_window(self, event, canvas, frame):
        # update the scrollbars to match the size of the inner frame
        size = (frame.winfo_reqwidth(), frame.winfo_reqheight())
        canvas.config(scrollregion='0 0 %s %s' % size)

    
    #spinbox only numbers
    def ValidateRange(self, value):
        # disallow anything but numbers
        valid = value.isdigit() or value == ''
        if not valid:
            self.root.bell()

        return valid
    
    def InvalidInput(self):
        self.error_text_train["text"] = 'laps must be a number.'