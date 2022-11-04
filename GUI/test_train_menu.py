from tkinter import *
from tkinter import ttk

from utils import ExperimentBanner, GetDatasets, GetConfig

class menu():
    def __init__(self, root):
        self.root = Toplevel(root)
        self.root.resizable(FALSE, FALSE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.experiments, self.experiment_banners = [], []
        self.tests, self.test_banners = [], []

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

    def add_train_elements(self):
        # Create a canvas object and a vertical scrollbar for scrolling it.
        self.experiments_frame_scrollbar = ttk.Scrollbar(self.trainframe)
        self.experiment_canvas = Canvas(self.trainframe, bd=2, width=180, 
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
        self.laps_entry = ttk.Entry(self.trainframe, textvariable=lapsvar, text="laps")

        embeddings = ["TransE_l2", "DistMult", "ComplEx", "TransR"]
        choices_emb = StringVar(value=embeddings)
        self.embedlistbox = Listbox(self.trainframe, listvariable=choices_emb, height=4, selectmode='multiple')

        datasets = GetDatasets()
        choices_datasets = StringVar(value=list(datasets.keys()))
        self.datasetlistbox = Listbox(self.trainframe, listvariable=choices_datasets, height=4)
        self.datasets_scrollbar = ttk.Scrollbar(self.trainframe)
        self.datasetlistbox.config(yscrollcommand=self.datasets_scrollbar.set)
        self.datasets_scrollbar.config(command=self.datasetlistbox.yview)


        singlecheckvar = BooleanVar(value=False)
        self.single_check = ttk.Checkbutton(self.trainframe, text='is single\nrelation', variable=singlecheckvar)

        singletextvar = StringVar()
        self.single_entry = ttk.Entry(self.trainframe, textvariable=singletextvar, text="name")
        
        self.add_to_list_train = ttk.Button(self.trainframe, text="add", 
        command=lambda: self.add_to_list("train"))

        self.remove_last_train = ttk.Button(self.trainframe, text="remove last", 
        command=lambda: self.remove_from_list("train"))

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
        self.datasets_scrollbar.place(x = 125, y = 63, height=65)

        self.embedlistbox.grid(row=3, column=1,)
        # self.embedding_scrollbar.place(x = 268, y = 65)
        
        #row4
        self.single_check.grid(row=4, column=0, rowspan=2)
        self.relationnamelabel.grid(row=4, column=1)

        #row5
        self.single_entry.grid(row=5, column=1)

        #row6
        self.add_to_list_train.grid(row=6, column=0)
        self.remove_last_train.grid(row=6, column=1)

    
    def add_test_elements(self):
        pass
    
    def grid_testframe(self):
        pass
    

    # MISC button functions
    def add_to_list(self, from_frame:str):
        if(from_frame == "train"):
            a = self.name_entry.get()
            b = int(self.laps_entry.get())
            c = self.datasetlistbox.get(ACTIVE)
            d = [self.embedlistbox.get(idx) for idx in self.embedlistbox.curselection()]
            e = self.single_check.state()[0]
            f = self.single_entry.get()
            print(a,b,c,d,e,f)

            # testing only:
            for i in range(5):
                testbanner = ExperimentBanner(self.experiments_frame, "experiment1", 
                "experiment_name", 150, "dataset", ["embeddings"])
                testbanner.getbanner().grid(row=i,column=0)

        if(from_frame == "test"):
            pass

    
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

        