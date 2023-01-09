from tkinter import *
from tkinter import ttk

import sys, pathlib, os, random
import matplotlib.pyplot as plt
import networkx as nx
import pygame as pg
from guiutils import GetTestsPaths
from keras.models import load_model
from keras import Model
from tqdm import tqdm
from itertools import chain

import numpy as np
import tensorflow as tf

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

        if (len(pathdicts) > self.MAX_PATHS_TO_DISPLAY):
            pathdicts = pathdicts[0:self.MAX_PATHS_TO_DISPLAY]

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

    def create_networkX_graph(self, triples:list):
        G = nx.DiGraph()
        print(len(triples))
        for t in triples:
            G.add_node(t[0])
            G.add_node(t[2])
            G.add_edge(t[0], t[2], name=t[1])
            G.add_edge(t[2], t[0], name=f"¬{t[1]}")

        for n in G.nodes():
            G.add_edge(n, n, name="NO_OP")

        print(f"nodes:{G.number_of_nodes()}, edges:{G.number_of_edges()}")
        return G

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
        paths_with_neighbors = self.get_weighted_paths_with_neighbors(G, agent, is_ppo, pathdicts, entities_embs, relations_embs)
        processed_pathdicts = self.keep_valuable_nodes_and_recalculate_positions(node_positions, paths_with_neighbors, width, height)

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

        self.nodes, self.neighbor_edges, self.path_edges = [], [], []

        current_visualized_path_idx = 0
        path_to_viz = processed_pathdicts[current_visualized_path_idx]

        object_step_dicts = []

        for i,n in enumerate(path_to_viz['present_nodes']):
            position = path_to_viz['node_path_positions'][i]
            node = Node(font, random.choice(node_colors), n, position[0], position[1], width, height)
            self.nodes.append(node)

        for step in path_to_viz['path']:
            o_step_dict = dict()

            valid = step['valid']
            nodes_in_rel = [n for n in self.nodes if n.text == valid[0] or n.text == valid[2]]
            e = Edge(font, valid[1][0], valid[1][1], nodes_in_rel[0], nodes_in_rel[1])
            e.set_active_state(True)
            e.set_show_edge_info(True)
            self.path_edges.append(e)
            
            o_step_dict["curr_node"] = nodes_in_rel[0]
            #TODO: add the nodes to the object step dict.

            worst = step['worst']
            best = step['best']

            for i in range(len(worst)):
                nodes_in_rel = [n for n in self.nodes if n.text == worst[i][0] or n.text == worst[i][2]]
                i1, i2 = (0,0)  if len(nodes_in_rel) == 1 else (0,1)
                e1 = Edge(font, worst[i][1][0], worst[i][1][1], nodes_in_rel[i1], nodes_in_rel[i2])

                nodes_in_rel = [n for n in self.nodes if n.text == best[i][0] or n.text == best[i][2]]
                i1, i2 = (0,0)  if len(nodes_in_rel) == 1 else (0,1)
                e2 = Edge(font, best[i][1][0], best[i][1][1], nodes_in_rel[i1], nodes_in_rel[i2])

                self.neighbor_edges.extend((e1,e2))

        while not requested_exit:
            screen.fill(white)
            prev_button.run(screen)
            next_button.run(screen)
            
            for e in self.path_edges:
                e.run(screen)

            for e in self.neighbor_edges:
                e.run(screen)

            for n in self.nodes:
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
    
    def get_weighted_paths_with_neighbors(self, G:nx.Graph, agent:Model, is_ppo: bool,
    pathdicts:list, entities_embs:dict, relations_embs:dict):
        # print(pathdicts)
        # print(entities_embs.keys())
        # print(relations_embs.keys())

        # network input
        # [(*e1,*r),*et] [*relation_embedding, *entity_embedding]

        res = []

        for t in tqdm(pathdicts):
            path = t["path"]

            e_0 = path[0][0]
            e_final = t["target"]
            r = G.adj[e_0][e_final]["name"]

            inputs, neighbors = [], []

            for p in path:
                if[p[1] == "NO_OP"]:
                    re = list(np.zeros(len(entities_embs[e_0])))
                else:
                    re = relations_embs[p[1]]

                observation = [*entities_embs[e_0], *relations_embs[r],
                *entities_embs[p[0]], *re, *entities_embs[p[2]]]

                inputs.append(observation)

                adjacents = G.adj[p[0]].copy()
                del adjacents[p[2]]
                # print(f"adjacency in node {p[0]}->{p[2]} is:\n {adjacents}\n")

                current_node_neighbors = []
                for node, v in adjacents.items():
                    rel = v["name"]

                    if[rel == "NO_OP"]:
                        re = list(np.zeros(len(entities_embs[e_0])))
                    else:
                        re = relations_embs[rel]
                    
                    observation = [*entities_embs[e_0], *relations_embs[r],
                    *entities_embs[p[0]], *re, *entities_embs[node]]

                    inputs.append(observation)
                    current_node_neighbors.append((rel, node))
                    
                neighbors.append(current_node_neighbors)
           
            inputs_stacked = np.vstack(np.array(inputs))
            if(is_ppo):
                output = agent([inputs_stacked, 0 , 0 ])
            else:
                output = agent([inputs_stacked])

            calculated = tf.get_static_value(output)

            # print(f"output values:\n {calculated}\nneighbors:\n{neighbors}\npathdict:\n{t}")

            res_dict = dict()
            res_dict['target'] = t['target']
            cont = 0
            path_res = []
            for i, p in enumerate(path):
                step_dict = dict()
                step_dict["valid"] = (p[0], (p[1], calculated[cont][0]), p[2])
                cont += 1
                neighs = []
                for n in neighbors[i]:
                    neighs.append((p[0],(n[0],calculated[cont][0]), n[1]))
                    cont += 1

                step_dict['neighbors'] = neighs
                path_res.append(step_dict)
            
            res_dict['path'] = path_res
            res.append(res_dict)
            # print(cont, len(calculated))
    
        return res

    def keep_valuable_nodes_and_recalculate_positions(self, node_positions:dict, path_with_neighbors:list, w:int, h:int, maxnodes:int = 2):
        def get_weakest_idx(node_list:list, weak_type:str):
            if(weak_type == "min"):
                weak_val = 999999
            else:
                weak_val = -999999

            res = 0
            for i, n in enumerate(node_list):
                val = n[1][1]
                if(weak_type == "min" and val < weak_val):
                    res = i
                    weak_val = val

                if(weak_type == "max" and val > weak_val):
                    res = i
                    weak_val = val

            return res

        for p in path_with_neighbors:
            all_nodes_in_path = set()
            path_with_processed_neighbors = []

            for step in p['path']:
                step_dict = dict()
                vld = step["valid"]
                step_dict["valid"] = vld
                all_nodes_in_path.update([vld[0], vld[2]])

                # print(f"\n{step}\n")
                # Calculate the best and worst neighbors for the path step.
                worst, best = [], []
                worst_weakest_idx, best_weakest_idx = 0, 0
                for n in step['neighbors']:

                    if len(worst) < maxnodes or len(best) < maxnodes:
                        worst.append(n)
                        best.append(n)
                        if(len(worst) == maxnodes or len(best) == maxnodes):
                            worst_weakest_idx = get_weakest_idx(worst, 'max')
                            best_weakest_idx = get_weakest_idx(best, 'min')

                    else:
                        # print(f"\nevaluating node {n}\n before:\nworst:{worst}\nbest:\n{best}\n")
                        if(n[1][1] < worst[worst_weakest_idx][1][1]):
                            worst.pop(worst_weakest_idx)
                            worst.append(n)
                            worst_weakest_idx = get_weakest_idx(worst, 'max')
                        
                        if(n[1][1] > best[best_weakest_idx][1][1]):
                            best.pop(best_weakest_idx)
                            best.append(n)
                            best_weakest_idx = get_weakest_idx(best, 'min')

                step_dict["worst"] = worst
                step_dict["best"] = best

                all_n = set()
                for i in range(len(worst)):
                    all_n.add(worst[i][0])
                    all_n.add(worst[i][2])
                    all_n.add(best[i][0])
                    all_n.add(best[i][2])

                all_nodes_in_path.update(all_n)

                # print(step_dict)
                path_with_processed_neighbors.append(step_dict)

            p['path'] = path_with_processed_neighbors
            p['present_nodes'] = all_nodes_in_path

            minvx, maxvx, minvy, maxvy = w,0,h,0

            for node in all_nodes_in_path:
                x, y = node_positions[node]

                if x < minvx:
                    minvx = x

                if y < minvy:
                    minvy = y

                if x > maxvx:
                    maxvx = x

                if y > maxvy:
                    maxvy = y
            
            path_node_positions = []
            conversion_factor_x, conversion_factor_y = (w)/(maxvx-minvx), (h)/(maxvy-minvy)
            for node in all_nodes_in_path:
                x, y =  int((node_positions[node][0] - minvx) * conversion_factor_x), int((node_positions[node][1] - minvy) * conversion_factor_y)
                path_node_positions.append((x,y))
            
            p['node_path_positions'] = path_node_positions

        return path_with_neighbors
                

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

    def run(self, screen:pg.surface.Surface):
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
    def __init__(self, font: pg.font.Font, color:tuple, text:str, x:int , y:int, w:int, h:int):
        self.color, self.text, self.font = color, text, font

        mid_w = w/2
        if(x < mid_w):
            x += 160 * (1 - x/mid_w)
        else:
            x -= 160 * ((x/w)-0.5)*2

        mid_h = h/2
        if(y < mid_h):
            y += 90 * (1 - y/mid_h)
        else:
            y -= 90 * ((y/h)-0.5)*2

        self.x , self.y = int(x), int(y)

    def run(self, screen:pg.surface.Surface):
        
        # render node circle.
        pg.draw.circle(screen, self.color, (self.x, self.y+40), 40)
        pg.draw.circle(screen, (0,0,0), (self.x, self.y+40), 40, 4)

        # render text
        text_img = self.font.render(self.text, True, (0,0,0))
        t_w = text_img.get_width()
        t_h = text_img.get_height()
        screen.blit(text_img, (self.x - int(t_w/2), self.y - int(t_h/2)+40))

class Edge:
    def __init__(self, font:pg.font.Font, relation:str, value:float, a:Node, b:Node):
        self.is_active, self.show_edge_info = False, False
        self.active_color, self.base_color = (136, 8, 8), (136, 8, 8)

        self.font, self.rel, self.value, self.a, self.b = font, relation, value, a, b

    def run(self, screen:pg.surface.Surface):
        origin, dest = (self.a.x, self.a.y+20), (self.b.x, self.b.y+20)
        color = self.active_color if self.is_active else self.base_color

        if(origin == dest):
            self.draw_bezier(screen, origin, color)
        else:
            pg.draw.line(screen, color, origin, dest, 3)

        # render text
        if self.show_edge_info:
            text_img = self.font.render(f"{self.rel}-({self.value:.4f})", True, (0,0,0))
            screen.blit(text_img, (int((self.a.x + self.b.x)/2), int((self.a.y + self.b.y)/2)))

    def set_active_state(self, s:bool):
        self.is_active = s
    
    def set_show_edge_info(self, s:bool):
        self.show_edge_info = s

    def draw_bezier(self, screen, point, color):
        pass
