from tkinter import *
from tkinter import ttk

import sys, pathlib, os, random
import matplotlib.pyplot as plt
import networkx as nx
import pygame as pg
from guiutils import GetTestsPaths
from keras.models import load_model
from keras import Model

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

        subfolders = [f.name.rstrip(f"_{dataset}_0") for f in os.scandir(f"{self.datasets_dir}/{dataset}/embeddings") if f.is_dir()]
        embedding = random.choice(subfolders)

        dm = DataManager(name=agent_name)
        triples, relations_emb, entities_emb, _ = dm.get_dataset(dataset, embedding)

        agent = self.get_agent(agent_name, dataset, embedding)

        G = self.create_networkX_graph(triples)
        pos = nx.drawing.layout.kamada_kawai_layout(G)

        # THIS DRAWS THE COMLPETE GRAPH
        # nx.draw_networkx(G, pos, with_labels=True, font_weight='bold')
        # plt.show()
        
        # THIS LAUNCHES OUR VISUALIZER.
        self.pygame_display(agent, G, pos, pathdicts, relations_emb, entities_emb)

    def create_networkX_graph(self, triples:list):
        G = nx.Graph()
        print(len(triples))
        for t in triples:
            G.add_node(t[0])
            G.add_node(t[2])
            G.add_edge(t[0], t[2], object=t[1])

        print(f"nodes:{G.number_of_nodes()}, edges:{G.number_of_edges()}")
        return G

    def get_agent(self, name:str, dataset:str, embedding:str):
        """
        given the name of the agent, the dataset name and the embedding to use, gets the model for that agent.

        :param name: the name of the agent.
        :param dataset: the name of the dataset
        :param embedding: the name of the embedding to use.

        :return: the model for the requested agent.
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
                agent= actor

            if(base_exist):
                policy_network = load_model(base)
                agent= policy_network

        return agent

    def pygame_display(self, agent:Model, G: nx.Graph, pos:dict, pathdicts:list, relations_emb:dict, entities_emb:dict):
        current_dir = pathlib.Path(__file__).parent.resolve()
        assests_dir = pathlib.Path(f"{current_dir}/assets").resolve()



        # setup variables
        size = width, height = 1280, 720
        white = 255, 255, 255
        requested_exit = False

        pg.init()
        screen = pg.display.set_mode(size)
        pg.display.set_caption("Path Visualization")

        prev_button = Button(100, 360, f"{assests_dir}/leftarrow.png", lambda: print("prev"))
        next_button = Button(1180, 360, f"{assests_dir}/rightarrow.png", lambda: print("next"))
       
        prev_button.render(screen)
        next_button.render(screen)

        print(pos)
        
        while not requested_exit:

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    requested_exit = True

            screen.fill(white)
            prev_button.render(screen)
            next_button.render(screen)

            
        
        pg.quit()

    def pos_to_pygame(self, pos:tuple, w:int, h:int):
        pass

    

# PYGAME helper classes:
class Button:
    def __init__(self, x, y, img_path: str, command):
        self.img = pg.image.load(img_path).convert_alpha()
        self.rect = self.img.get_rect()
        self.rect.topleft = (x,y)
        self.command = command

    def render(self, screen):
        screen.blit(self.img, self.rect)

    def get_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pg.mouse.get_pos()):
                self.command()
    