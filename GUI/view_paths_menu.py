from tkinter import *
from tkinter import ttk

import sys, pathlib, os, random
import matplotlib.pyplot as plt
import networkx as nx
import pygame as pg
from guiutils import GetTestsPaths
from keras.models import load_model
from keras import Model

import numpy as np

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

        self.MAX_PATHS_TO_DISPLAY = 1000
        
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
        # dict with key->node names, values->node pos in format array([float, float]) 
        pos = nx.drawing.layout.kamada_kawai_layout(G)
        
        # LAUNCHES VISUALIZER.
        self.pygame_display(agent, G, pos, pathdicts, relations_emb, entities_emb)

        # DRAWS THE COMLPETE GRAPH with network X
        # nx.draw_networkx(G, pos, with_labels=True, font_weight='bold')
        # plt.show()

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

    def pygame_display(self, agent:Model, G: nx.Graph, pos:dict, 
    pathdicts:list, relations_embs:dict, entities_embs:dict):

        current_dir = pathlib.Path(__file__).parent.resolve()
        assests_dir = pathlib.Path(f"{current_dir}/assets").resolve()

        is_ppo = False
        for layer in agent.layers:
            if layer.name == "Advantage" or layer.name == "Old_Prediction":
                is_ppo = True
        
        print("=== Actor Network ===")
        print(agent.summary())

        # setup variables
        size = width, height = 1280, 720
        white = 255, 255, 255
        requested_exit = False

        node_positions = self.get_node_absolute_pos_pygame(pos, width, height)
        self.get_weighted_paths_with_representative_neighbors(G, agent, is_ppo, pathdicts, entities_embs, relations_embs)
        # node_neighbors = self.get_node_representative_neighbors(G, agent, is_ppo, list(node_positions.keys()), entities_embs, relations_embs)
        # self.get_node_pygame_positions(pos, pathdicts, width, height)

        pg.init()
        screen = pg.display.set_mode(size)
        pg.display.set_caption("Path Visualization")
        font = pg.font.SysFont("dejavuserif", 16)

        # Objects
        prev_button = Button(30, 360, f"{assests_dir}/leftarrow.png", 0.8, lambda: print("prev"))
        next_button = Button(1180, 360, f"{assests_dir}/rightarrow.png", 0.8, lambda: print("next"))

        node_colors = [(255,127,80), (240,128,128), (255,160,122), (238,232,170), (173,255,47), (144,238,144),
        (102,205,170), (0,255,255), (127,255,212), (135,206,235), (106,90,205), (186,85,211), (219,112,147),
        (255,228,196), (244,164,96), (176,196,222), (169,169,169)]

        nodes = []
        for name, position in node_positions.items():
            node = Node(font, random.choice(node_colors), name, position[0], position[1])
            nodes.append(node)
       
        while not requested_exit:
            screen.fill(white)
            prev_button.run(screen)
            next_button.run(screen)
            
            for n in nodes:
                n.run(screen)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    requested_exit = True
            
            pg.display.flip()

        pg.quit()

    def get_node_absolute_pos_pygame(self, pos:dict, w:int, h:int):
        res = dict()

        min_val_x, max_val_x, min_val_y, max_val_y = 1, -1, 1, -1

        for x,y in pos.values():
            if x < min_val_x:
                min_val_x = x

            if y < min_val_y:
                min_val_y = y

            if x > max_val_x:
                max_val_x = x

            if y > max_val_y:
                max_val_y = y
                
        for p in pos.items():
            x, y = p[1][0], p[1][1]
            
            x_abs, y_abs = 0, 0
            if(x <= 0):
                x_abs = int((1-abs(x))*(w/2))
            else:
                x_abs = int((abs(x))*(w/2)+w/2)

            if(y <= 0):
                y_abs = int((1-abs(y))*(h/2))
            else:
                y_abs = int((abs(y))*(h/2)+h/2)

            res[p[0]] = (x_abs, y_abs)

        return res
    
    def get_weighted_paths_with_representative_neighbors(self, G:nx.Graph, agent:Model, is_ppo: bool,
    pathdicts:list, entities_embs:dict, relations_embs:dict, maxnodes:int = 3):
        # print(pathdicts)
        # print(entities_embs)
        # print(relations_embs)

        # netork input
        # [(*e1,*r),*et] [*relation_embedding, *entity_embedding]
        
        res = dict()
        for t in pathdicts:
            path = t["path"]

            e_0 = path[0][0]
            e_final = t["target"]
            r = G.adj[e_0][e_final]

            for p in path:
                observation = [entities_embs(e_0)*, relations_embs(r)*]

            # TODO: FINISH THIS SHIT...
           
            # inputs_stacked = np.vstack(np.array(s))
            # if(is_ppo):

            #     self.policy_network([inputs_stacked, 0 , 0 ])
            # else:
            #     self.policy_network([inputs_stacked])
            # res[og]=[t[2]]

    def get_node_pygame_positions(self, pos:dict, pathdicts:list, w:int, h:int):
        if (len(pathdicts) > self.MAX_PATHS_TO_DISPLAY):
            pathdicts = pathdicts[0:self.MAX_PATHS_TO_DISPLAY]
        
        for p in pathdicts:
            abs_pos = dict()
            print(p)
            for t in p["path"]:
                abs_pos[t[0]] = pos[t[0]]
                abs_pos[t[2]] = pos[t[2]]

            print(abs_pos)

            min_val_x, max_val_x, min_val_y, max_val_y = 1, -1, 1, -1
            for x, y in abs_pos.values():

                if x < min_val_x:
                    min_val_x = x

                if y < min_val_y:
                    min_val_y = y

                if x > max_val_x:
                    max_val_x = x

                if y > max_val_y:
                    max_val_y = y
            

    

# PYGAME helper classes:
class Button:
    def __init__(self, x:int, y:int, img_path:str, scale:float, command):
        self.img = pg.image.load(img_path).convert_alpha()
        h = self.img.get_height()
        w = self.img.get_width()
        self.img = pg.transform.scale(self.img, (int(w*scale), int(h*scale)))
        self.rect = self.img.get_rect()
        self.rect.topleft = (x,y)
        self.command = command
        self.clicked = False

    def run(self, screen):
        if pg.mouse.get_pressed()[0] == 1:
            pos = pg.mouse.get_pos()
            if self.rect.collidepoint(pos) and not self.clicked:
                self.clicked = True
                print(f"clicked {self.rect}")
                self.command()
        
        if pg.mouse.get_pressed()[0] == 0:
            self.clicked = False

        screen.blit(self.img, self.rect)
    
class Node:
    def __init__(self, font: pg.font.Font, color:tuple, text:str, x:int , y:int):
        self.color = color
        self.text = text
        self.font = font
        self.x , self.y = x,y

    def run(self, screen):
        # render node circle.
        pg.draw.circle(screen, self.color, (self.x, self.y+40), 40)
        pg.draw.circle(screen, (0,0,0), (self.x, self.y+40), 40, 4)


        # render text
        text_img = self.font.render(self.text, True, (0,0,0))
        t_w = text_img.get_width()
        t_h = text_img.get_height()
        screen.blit(text_img, (self.x - int(t_w/2), self.y - int(t_h/2)+40))

