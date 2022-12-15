from tkinter import *
from tkinter import ttk

import sys, pygame, pathlib, os, random
import matplotlib.pyplot as plt
import networkx as nx
from guiutils import GetTestsPaths
from keras.models import load_model

from inspect import getsourcefile
import os.path as path, sys

# add the parent directory to path so you can import config into data manager. 
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from model.data.data_manager import DataManager
sys.path.pop(0)


class menu():
    def __init__(self, root):
        self.root = Toplevel(root)
        self.root.title('View Paths')
        self.root.resizable(FALSE, FALSE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.mainframe = ttk.Frame(self.root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0)

        self.tests = GetTestsPaths()
        self.testnames = [t['name'] for t in self.tests]
        
        self.add_elements()

        self.maindir = pathlib.Path(__file__).parent.parent.resolve()
        self.datasets_dir = pathlib.Path(f"{self.maindir}/datasets")
        
    def add_elements(self):
        self.testselect_label = ttk.Label(self.mainframe, text='Select test')
        testselect_strvar = StringVar(value=self.testnames)
        # testselect_strvar = StringVar(value=["a","a","a","a","a"])
        self.testselect_listbox = Listbox(self.mainframe, listvariable=testselect_strvar, height=4, exportselection=False)
        
        self.testselect_scrollbar = ttk.Scrollbar(self.mainframe)
        self.testselect_listbox.config(yscrollcommand=self.testselect_scrollbar.set)
        self.testselect_scrollbar.config(command=self.testselect_listbox.yview)

        self.start_display = ttk.Button(self.mainframe, text="View", command= lambda: self.launch_visualizer())

        self.grid_elements()

    def grid_elements(self):
        # row 0
        self.testselect_label.grid(row=0, column=0)

        # row 1
        self.testselect_listbox.grid(row=1, column=0)

        self.testselect_scrollbar.grid(row=1, column=0)
        self.testselect_listbox.update()
        l_width = self.testselect_listbox.winfo_width()
        l_height = self.testselect_listbox.winfo_height()
        posx = self.testselect_listbox.winfo_x()
        posy = self.testselect_listbox.winfo_y()
        self.testselect_scrollbar.place(x = posx + l_width - 15, y = posy -10, height=l_height-3)

        # row 2 
        self.start_display.grid(row=2, column=0)

    def launch_visualizer(self):
        i = self.testselect_listbox.curselection()
        if(i == ()):
            print("please select a test to visualize.")
            return

        active = self.testselect_listbox.get(i)
        print(f"loading graph information for {active}")

        pathdicts, dataset, agent_name = [(t["pathdicts"], t["dataset"], t["agent_name"]) for t in self.tests if t["name"] == active][0]

        # TODO: get the agent models here to get predictions and setup the triples.

        dm = DataManager(name=agent_name)
        subfolders = [f.name.rstrip(f"_{dataset}_0") for f in os.scandir(f"{self.datasets_dir}/{dataset}/embeddings") if f.is_dir()]
        embedding = random.choice(subfolders)
        
        print(dataset, agent_name, embedding, random.choice(pathdicts))

        triples, relations_emb, entities_emb, _ = dm.get_dataset(dataset, embedding)

        print(triples[0]) #, list(relations_emb.items())[0], list(entities_emb.items())[0])

        G = self.create_networkX_graph(triples)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

    def create_networkX_graph(self, triples):
        G = nx.Graph()
        print(len(triples))
        for t in triples:
            G.add_node(t[0])
            G.add_node(t[2])
            G.add_edge(t[0], t[2], object=t[1])

        print(f"nodes:{G.number_of_nodes()}, edges:{G.number_of_edges()}")
        return G

    def pygame_display(self):
        current_dir = pathlib.Path(__file__).parent.resolve()
        assests_dir = pathlib.Path(f"{current_dir}/assets").resolve()

        pygame.init()

        size = width, height = 720, 540
        speed = [1, 1]
        white = 255, 255, 255

        screen = pygame.display.set_mode(size)

        ball = pygame.image.load(f"{assests_dir}/intro_ball.gif")
        ballrect = ball.get_rect()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

            ballrect = ballrect.move(speed)

            if ballrect.left < 0 or ballrect.right > width:
                speed[0] = -speed[0]

            if ballrect.top < 0 or ballrect.bottom > height:
                speed[1] = -speed[1]

            screen.fill(white)
            screen.blit(ball, ballrect) 
            pygame.display.flip()

    def get_agent(self, name:str, dataset:str, embedding:str):
        """
        given the name of the agent, the dataset name and the embedding to use, gets the model for that agent.

        :param name: the name of the agent.
        :param dataset: the name of the dataset
        :param embedding: the name of the embedding to use.

        :return: the models for the requested agent.
        """
        agent_path = pathlib.Path(f"{self.maindir}/model/data/agents/{name}").resolve()
        constant_path = f"{agent_path}/{dataset}-{embedding}"

        ppo = constant_path
        base = ppo + ".h5"

        ppo_exist = os.path.isdir(ppo)
        base_exist = os.path.isfile(base)

        if(ppo_exist and base_exist):
            print(f"2 agents found for embedding {embedding} and dataset {dataset}, remove one.")
        else:
            if(ppo_exist):
                actor = load_model(f"{ppo}/actor.h5")
                critic = load_model(f"{ppo}/critic.h5")
                agent= [actor, critic]

            if(base_exist):
                policy_network = load_model(base)
                agent = [policy_network]

        return agent
