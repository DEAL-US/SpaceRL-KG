from tkinter import *
from tkinter import ttk

import sys, pathlib, os, random, time, math
import networkx as nx
import pygame as pg
from guiutils import GetTestsPaths, remove_prefix_suffix
from keras.models import load_model
from keras import Model
from tqdm import tqdm
from itertools import chain
from copy import deepcopy

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from inspect import getsourcefile
import os.path as path, sys

# add the parent directory to path so you can import config into data manager. 
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from model.data.data_manager import DataManager
sys.path.pop(0)


NODERADIUS = 15

class menu():
    #########################
    # TKINTER MENU OPTIONS: #
    #########################

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

        self.MAX_PATHS_TO_DISPLAY = 10
        
        self.add_elements()

        self.maindir = pathlib.Path(__file__).parent.parent.resolve()
        self.datasets_dir = pathlib.Path(f"{self.maindir}/datasets")
        self.agents_dir = pathlib.Path(f"{self.maindir}/model/data/agents/")

    def add_elements(self):

        self.maxnumpathsframe = ttk.Frame(self.mainframe)

        self.maxnumpaths_label = ttk.Label(self.maxnumpathsframe, text='Maximum number of paths to display:')
        
        vcmd = (self.root.register(lambda value: self.validation(value, "maxnumpath")), '%P')
        ivcmd = (self.root.register(lambda: self.invalid("maxnumpath")),)

        maxnumpathsvar = IntVar()
        self.maxnumpaths = ttk.Entry(self.maxnumpathsframe, textvariable=maxnumpathsvar, text="max # of paths",
        validate='key', validatecommand=vcmd, invalidcommand=ivcmd)

        self.testselect_label = ttk.Label(self.mainframe, text='Select test')
        testselect_strvar = StringVar(value=self.testnames)
        self.testselect_listbox = Listbox(self.mainframe, listvariable=testselect_strvar, height=8, width=50, exportselection=False)
        
        self.testselect_scrollbar = ttk.Scrollbar(self.mainframe)
        self.testselect_listbox.config(yscrollcommand=self.testselect_scrollbar.set)
        self.testselect_scrollbar.config(command=self.testselect_listbox.yview)

        self.start_display = ttk.Button(self.mainframe, text="View", state='disabled', command= lambda: self.launch_visualizer())

        self.errors = Label(self.mainframe, text='', fg='red', bg="#FFFFFF")

        self.grid_elements()

    def grid_elements(self):

        # row 0
        self.maxnumpathsframe.grid(row=0, column=0)
        self.maxnumpaths_label.grid(row=0, column=0)
        self.maxnumpaths.grid(row=0, column=1)

        # row 1
        self.testselect_label.grid(row=1, column=0)

        # row 2
        self.testselect_listbox.grid(row=2, column=0)

        self.testselect_scrollbar.grid(row=2, column=0)
        self.testselect_listbox.update()
        l_width = self.testselect_listbox.winfo_width()
        l_height = self.testselect_listbox.winfo_height()
        posx = self.testselect_listbox.winfo_x()
        posy = self.testselect_listbox.winfo_y()
        self.testselect_scrollbar.place(x = posx + l_width - 15, y = posy -10, height=l_height-3)

        # row 3
        self.start_display.grid(row=3, column=0)

        # row 4
        self.errors.grid(row=4, column=0)

    def validation(self, value:str, origin:str):
        int_origins = ["maxnumpath"]
        ranges = [(1,1000)]
        a, b = ranges[0][0], ranges[0][1]

        if(origin != int_origins[0]):
            print(f"bad origin {origin}")
            return False

        if(value == ""):
            self.change_button_status(False)
            return True

        if(value.isdigit()): #check for any non-negative number, non float number
            v = int(value)
        else:
            self.errors["text"] = f"{origin} must be a number"
            return False
        
        if(v < a or v > b):
            self.errors["text"] = f"{origin} must in range [{a}-{b}]"
            return False
        
        self.change_button_status(True)
        self.MAX_PATHS_TO_DISPLAY = v
        return True

    def invalid(self, origin:str):
        """
        If the validation of the field didn't pass, what actions to take.
        :param origin: the origin of the validation trigger it represents one of the text fields in the window. 
        """
        self.change_button_status(False)
        # self.errors["text"] = "an unexpected error ocurred"

    def change_button_status(self, on_off: bool):
        self.start_display['state'] = 'enabled' if on_off else 'disabled'

    ##############################
    # PYGAME GRAPH VISUALIZATION #
    ##############################

    def launch_visualizer(self):
        i = self.testselect_listbox.curselection()
        if(i == ()):
            self.errors["text"] = "please select a test to visualize."
            return

        active = self.testselect_listbox.get(i)
        print(f"loading graph information for {active}")

        pathdicts, dataset, agent_name = [(t["pathdicts"], t["dataset"], t["agent_name"]) for t in self.tests if t["name"] == active][0]

        if (len(pathdicts) > self.MAX_PATHS_TO_DISPLAY):
            print(f"paths exceed limit, displaying the first {self.MAX_PATHS_TO_DISPLAY}.")
            pathdicts = pathdicts[0:self.MAX_PATHS_TO_DISPLAY]

        subfolders = [remove_prefix_suffix(f.name, f"{dataset}-", ".h5") for f in os.scandir(f"{self.agents_dir}/{agent_name}")]
        subfolders.remove("config_used.txt")
        embedding = random.choice(subfolders)

        dm = DataManager(name=agent_name)
        triples, relations_emb, entities_emb, _ = dm.get_dataset(dataset, embedding)

        agent = self.get_agent(agent_name, dataset, embedding)

        G = self.create_networkX_graph(triples)
        # dict with key->node names, values->node pos in format array([float, float]) 
        
        # LAUNCHES VISUALIZER.
        self.pygame_display(agent, G, pathdicts, relations_emb, entities_emb)

        # DRAWS THE COMLPETE GRAPH with network X WARNING! SLOW!!
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
                agent = actor

            if(base_exist):
                policy_network = load_model(base)
                agent = policy_network

        return agent

    def create_networkX_graph(self, triples:list):
        G = nx.MultiDiGraph()
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

    def pygame_display(self, agent:Model, G: nx.Graph, 
    pathdicts:list, relations_embs:dict, entities_embs:dict):

        self.path_length = len(pathdicts[0]["path"])

        current_dir = pathlib.Path(__file__).parent.resolve()
        assests_dir = pathlib.Path(f"{current_dir}/assets").resolve()

        is_ppo = False
        for layer in agent.layers:
            if layer.name == "Advantage" or layer.name == "Old_Prediction":
                is_ppo = True
        
        print("=== Actor Network ===")
        print(agent.summary())

        # setup variables
        size = self.width, self.height = 1824, 1026
        white = 255, 255, 255
        requested_exit = False

        # Calculate informaton about all the paths that are going to be represented and the participant nodes.
        paths_with_neighbors = self.get_weighted_paths_with_neighbors(G, agent, is_ppo, pathdicts, entities_embs, relations_embs)
        self.processed_pathdicts = self.keep_valuable_nodes_and_recalculate_positions(paths_with_neighbors, self.width, self.height)

        pg.init()
        pg.display.set_caption("Path Visualization")
        self.font = pg.font.SysFont("dejavuserif", 16)

        screen = pg.display.set_mode(size)
        node_surface = pg.Surface(size, pg.SRCALPHA)

        # Objects
        prev_button = Button(30, int(self.height/2), f"{assests_dir}/leftarrow.png", 0.8, lambda: self.change_visualized_path(-1))
        next_button = Button(self.width-90, int(self.height/2), f"{assests_dir}/rightarrow.png", 0.8, lambda: self.change_visualized_path(1))

        # get initial path.
        self.current_visualized_path_idx, self.total_path_count = 0, len(self.processed_pathdicts)
        self.currently_visualized_path = self.processed_pathdicts[self.current_visualized_path_idx]

        # Initialize clock and paths
        self.clock = pg.time.Clock()
        self.init_visualized_path()
        self.cycle, self.ticks = -1, 0
        
        self.arrow_keys_text = SimpleText(f"Use the arrow keys ◀ ▶ to move through the nodes and the arrow buttons to either side of the screen to jump top the next path", self.width/2, self.height-20, (0,0,0))

        # THIS IS THE UPDATE METHOD, MEANING IT RUNS FOR EVERY FRAME.
        while not requested_exit:
            screen.fill(white)
            node_surface.fill(pg.Color(0,0,0,0)) 

            # run buttons
            prev_button.run(screen)
            next_button.run(screen)

            # run displayed texts
            self.numpath_displayed.run(screen)
            self.literal_path.run(screen)
            self.path_step_text.run(screen)
            self.arrow_keys_text.run(screen)

            # nodes must run first as adges rely on them.
            # but we need to redraw them later as they have to be on top
            for n in self.nodes:
                n.run(self.cycle, screen, node_surface)

            for e in self.path_edges:
                e.run(self.cycle, screen)

            for e in self.neighbor_edges:
                e.run(self.cycle, screen)

            for n in self.nodes:
                n.run(self.cycle, screen, node_surface)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    requested_exit = True
                
                # Arrow key scroll:

                if event.type == pg.KEYDOWN and self.ticks > 5:
                    if event.key == pg.K_LEFT:
                        self.cycle -= 1
                        if (self.cycle < -1):
                            self.cycle = self.path_length

                    if event.key == pg.K_RIGHT:
                        self.cycle += 1
                        if (self.cycle > self.path_length):
                            self.cycle = -1
                    
                    self.path_step_text = SimpleText(f"{self.cycle + 1}/{self.path_length+1}", self.width/2, self.height-40, (0,0,0))
                    self.ticks = 0
            
            screen.blit(node_surface, (0,0))

            pg.display.flip()

            self.clock.tick(120)
            self.ticks += 1

        pg.quit()

    def change_visualized_path(self, change:int):
        """
        change to a new path to visualize
        """
        if(change == -1 and self.current_visualized_path_idx == 0) or (change == 1 and self.current_visualized_path_idx == len(self.processed_pathdicts)-1):
            print("tried to go over range...")
            return 
        
        self.cycle = -1
        self.current_visualized_path_idx += change
        self.currently_visualized_path = self.processed_pathdicts[self.current_visualized_path_idx]
        self.init_visualized_path()

    def init_visualized_path(self):
        """
        creates all edge, node, and text objects that are going to be renderer by pygame
        """
        self.nodes, self.neighbor_edges, self.path_edges = [], [], []

        # Text objects init
        self.path_step_text = SimpleText(f"1/{self.path_length+1}", self.width/2, self.height-40, (0,0,0))
        self.numpath_displayed = SimpleText(f"{self.current_visualized_path_idx+1}/{self.total_path_count}", self.width/2, 40, (0,0,0))
        textual_path = ""
        
        # add nodes to node list.
        for i,n in enumerate(self.currently_visualized_path['present_nodes']):
            position = self.currently_visualized_path['node_path_positions'][i]
            node = Node(self.font, n, position[0], position[1], self.width, self.height)
            self.nodes.append(node)

        # add edges to edge list.
        is_first = True
        for i, step in enumerate(self.currently_visualized_path['path']):
            o_step_dict = dict()

            # HANDLING MAIN PATH...(VALID)
            valid = step['valid']
            if(is_first):
                textual_path += f" Inferred Path: {valid[0]} -> {valid[1][0]} -> {valid[2]} -> "
                is_first = False
            else:
                textual_path += f"{valid[1][0]} -> {valid[2]} -> "

            nodes_in_rel = [n for n in self.nodes if n.text == valid[0] or n.text == valid[2]]
            
            for n in nodes_in_rel:
                if(n.text == valid[0]):
                    n.main_in_step.add(i)
                else:
                    n.active_in_step.add(i)

                if(i+1 == self.path_length):
                    if(n.text == valid[2]):
                        n.main_in_step.add(i+1)
                        try:
                            n.active_in_step.remove(i)
                        except:pass

            try:
                a,b = (nodes_in_rel[0], nodes_in_rel[1]) if nodes_in_rel[0].text == valid[0] else (nodes_in_rel[1], nodes_in_rel[0])
                e = Edge(self.font, valid[1][0], valid[1][1], a, b)
            except:
                # node to itself.
                e = Edge(self.font, valid[1][0], valid[1][1], nodes_in_rel[0], nodes_in_rel[0])

            e.main_in_step.add(i)
            if(i+1 == self.path_length):
                e.main_in_step.add(i+1)

            self.path_edges.append(e)
            
            o_step_dict["curr_node"] = nodes_in_rel[0]
            worst = step['worst']
            # best = step['best']

            active = set()
            for x in range(len(worst)):
                w = [n for n in self.nodes if n.text == worst[x][0] or n.text == worst[x][2]]
                # b = [n for n in self.nodes if n.text == best[x][0] or n.text == best[x][2]]

                # for a in b:
                #     active.add(a)
                
                for a in w:
                    active.add(a)
            
                for a in active:
                    a.active_in_step.add(i)

            for j in range(len(worst)):
                # add worst nodes.
                nodes_in_rel = [n for n in self.nodes if n.text == worst[j][0] or n.text == worst[j][2]]
                if len(nodes_in_rel) != 1: #straight path
                    a,b = (nodes_in_rel[0], nodes_in_rel[1]) if nodes_in_rel[0].text == worst[j][0] else (nodes_in_rel[1], nodes_in_rel[0])
                else:
                    a,b = nodes_in_rel[0], nodes_in_rel[0]

                e1 = Edge(self.font, worst[j][1][0], worst[j][1][1], a, b)
                e1.active_in_step.add(i)
                
                # add best nodes.
                # nodes_in_rel = [n for n in self.nodes if n.text == best[j][0] or n.text == best[j][2]]
                # if len(nodes_in_rel) != 1: #straight path
                #     a,b = (nodes_in_rel[0], nodes_in_rel[1]) if nodes_in_rel[0].text == best[j][0] else (nodes_in_rel[1], nodes_in_rel[0])
                # else:
                #     a,b = nodes_in_rel[0], nodes_in_rel[0]

                # e2 = Edge(self.font, best[j][1][0], best[j][1][1], a, b)
                # e2.active_in_step.add(i)

                self.neighbor_edges.append(e1) # self.neighbor_edges.extend((e1,e2))
                

        self.literal_path = SimpleText(textual_path[:-3], self.width/2, 20, (0,0,0))

    def get_node_absolute_pos_pygame(self, pos:dict, w:int, h:int):
        res = dict()

        min_val_x, max_val_x, min_val_y, max_val_y = 9999, -9999, 9999, -9999

        for x,y in pos.values():
            if x < min_val_x:
                min_val_x = x

            if y < min_val_y:
                min_val_y = y

            if x > max_val_x:
                max_val_x = x

            if y > max_val_y:
                max_val_y = y
            
        # calculate cluster center.
        x_dist = max_val_x - min_val_x
        y_dist = max_val_y - min_val_y
        current_cluster_center = (max_val_x-(x_dist/2), max_val_y-(y_dist/2))

        max_val_x -= current_cluster_center[0]
        min_val_x -= current_cluster_center[0]
        max_val_y -= current_cluster_center[1]
        min_val_y -= current_cluster_center[1]

        # shift all elements in dict to 0,0
        for k, v in pos.items():
            x, y = v[0], v[1]

            x_pos = (x-current_cluster_center[0])/max_val_x
            y_pos = (y-current_cluster_center[1])/max_val_y
            
            pos[k] = (x_pos,y_pos)


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
        res = []

        for t in tqdm(pathdicts, "Recalculating paths..."):
            path = t["path"]

            e_0 = path[0][0]
            e_final = t["target"]
            r = G.adj[e_0][e_final][0]["name"]

            inputs, neighbors = [], []

            initial_q_entity = entities_embs[e_0]
            initial_q_relation = relations_embs[r]

            for p in path:
                if str(p[1]) == "NO_OP":
                    re = list(np.zeros(len(entities_embs[e_0]))) # Zeros if NO_OP
                else:
                    re = relations_embs[p[1]] # values for relation if exists.
                                
                current_status_node = entities_embs[p[0]]
                connective_rel = re
                destination_node = entities_embs[p[2]]

                observation = deepcopy([*initial_q_entity, *initial_q_relation, *current_status_node, *connective_rel, *destination_node])

                inputs.append(observation)

                # get adjacent to current node and remove main path entity.
                adjacents = G.adj[p[0]].copy()
                del adjacents[p[2]]
                
                # get current node neighbors observations
                current_node_neighbors = []
                for node, v in adjacents.items():
                    for relpair in v.values():
                        rel = relpair["name"]

                        if rel == "NO_OP":
                            re = list(np.zeros(len(entities_embs[e_0])))
                        else:
                            re = relations_embs[rel]
                        
                        current_status_node = entities_embs[p[0]]
                        connective_rel = re
                        destination_node = entities_embs[node]

                        observation = deepcopy([*initial_q_entity, *initial_q_relation, *current_status_node, *connective_rel, *destination_node])

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

    def keep_valuable_nodes_and_recalculate_positions(self, path_with_neighbors:list, w:int, h:int, maxnodes:int = 2):
        
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

        for p in tqdm(path_with_neighbors, "Recalculating paths..."):
            all_nodes_in_path = set()
            path_with_processed_neighbors = []

            for step in p['path']:
                step_dict = dict()
                vld = step["valid"]
                step_dict["valid"] = vld
                all_nodes_in_path.update([vld[0], vld[2]])

                # print(f"\n{step}\n")
                # Calculate the best and worst neighbors for the path step.
                worst = []
                # best = []
                worst_weakest_idx = 0
                # best_weakest_idx = 0

                for n in step['neighbors']:
                    if len(worst) < maxnodes: # or len(best) < maxnodes:
                        worst.append(n)
                        # best.append(n)
                        if len(worst) == maxnodes: #or len(best) == maxnodes):
                            worst_weakest_idx = get_weakest_idx(worst, 'max')
                            # best_weakest_idx = get_weakest_idx(best, 'min')

                    else:
                        # print(f"\nevaluating node {n}\n before:\nworst:{worst}\nbest:\n{best}\n")
                        if(n[1][1] < worst[worst_weakest_idx][1][1]):
                            worst.pop(worst_weakest_idx)
                            worst.append(n)
                            worst_weakest_idx = get_weakest_idx(worst, 'max')
                        
                        # if(n[1][1] > best[best_weakest_idx][1][1]):
                        #     best.pop(best_weakest_idx)
                        #     best.append(n)
                        #     best_weakest_idx = get_weakest_idx(best, 'min')

                step_dict["worst"] = worst
                # step_dict["best"] = best

                all_n = set()
                for i in range(len(worst)):
                    all_n.add(worst[i][0])
                    all_n.add(worst[i][2])
                    # all_n.add(best[i][0])
                    # all_n.add(best[i][2])

                all_nodes_in_path.update(all_n)

                # print(step_dict)
                path_with_processed_neighbors.append(step_dict)

            p['path'] = path_with_processed_neighbors
            p['present_nodes'] = all_nodes_in_path

            localG = nx.MultiDiGraph()
            
            localG.add_nodes_from(p["present_nodes"])
            for aux in p["path"]:
                v_t = aux['valid']
                worst = aux['worst']
                # best = aux['best']

                localG.add_edge(v_t[0], v_t[2], name = v_t[1][0])

                for wrst in worst:
                    localG.add_edge(wrst[0], wrst[2], name = wrst[1][0])

                # for bst in best:
                #     localG.add_edge(bst[0], bst[2], name = bst[1][0])


            # NetworkX direct drawing config, hardcoded values.
            # pos = nx.drawing.layout.spring_layout(localG) 

            # Graphviz as intermediary.
            pos = nx.drawing.nx_agraph.graphviz_layout(localG, prog="neato", args=" -s=125")

            res = self.get_node_absolute_pos_pygame(pos, w, h)

            path_node_positions = []
            for p_node in p["present_nodes"]:
                path_node_positions.append(res[p_node])

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
                # print(f"clicked {self.rect}")
                self.command()
        
        if pg.mouse.get_pressed()[0] == 0:
            self.clicked = False

        screen.blit(self.img, self.rect)
    
class Node:
    def __init__(self, font: pg.font.Font, text:str, x:int , y:int, w:int, h:int):
        self.main_in_step, self.active_in_step = set(), set()
        self.text, self.font = text, font
        self.is_active, self.is_main = False, False

        mid_w, mid_h = w/2, h/2

        # Hardcoded for aspect ratio 16/9
        if(x < mid_w):
            x += 160 * (1 - x/mid_w)
        else:
            x -= 160 * ((x/w)-0.5)*2

        if(y < mid_h):
            y += 90 * (1 - y/mid_h)
        else:
            y -= 90 * ((y/h)-0.5)*2

        self.x , self.y = int(x), int(y)
        self.center = (self.x, self.y)
        self.circle_rect = pg.Rect(self.center, (0, 0)).inflate((NODERADIUS*2, NODERADIUS*2))

    def get_state(self, cycle):
        borderopacity = 1
        write_text = False

        if(len(self.main_in_step) != 0): # part of main path
            if(cycle in self.main_in_step): # current node
                maincolor = (255, 0, 0, 255) # red
                borderopacity = 3

            else: # not current node.
                maincolor = (100, 149, 237, 255) # blue

        else:
            if(cycle in self.active_in_step):
                maincolor = (62, 195, 39, 190) # green
                write_text = True
            else:
                maincolor = (62, 195, 39, 75) # veeery clear green
        
        if cycle in self.main_in_step:
            write_text = True

        return maincolor, borderopacity, write_text

    def write_text(self, screen: pg.surface.Surface, cycle):
        w = screen.get_width()
        h = screen.get_height()
        text_img = self.font.render(self.text, True, (0,0,0))
        loc = self.move_text_by_cuadrant(text_img, w, h, 30)
        screen.blit(text_img, loc)

    def move_text_by_cuadrant(self, text_surface, w, h, gap):
        t_x = self.x - int(text_surface.get_width()/2)
        t_y = self.y - int(text_surface.get_height()/2)

        if(self.x < w/2 and self.y > h/2): # bottom left
            loc = (t_x -gap , t_y +gap)

        elif(self.x > w/2 and self.y > h/2): # bottom right
            loc = (t_x +gap, t_y +gap)

        elif(self.x > w/2 and self.y < h/2): # top right
            loc = (t_x +gap, t_y -gap)
            
        elif(self.x < w/2 and self.y < h/2): # top left
            loc = (t_x -gap, t_y -gap)
           
        else: # centered
            loc = (t_x , t_y -gap)
        
        return loc

    def run(self, cycle: int, screen:pg.surface.Surface, node_surface: pg.surface.Surface):
        # render node circle.
        maincolor, borderopacity, write_b = self.get_state(cycle)
        bordercolor = (0,0,0,255)

        pg.draw.circle(node_surface, maincolor, self.center, NODERADIUS)
        pg.draw.circle(node_surface, bordercolor, self.center, NODERADIUS, borderopacity)

        if write_b:
            self.write_text(screen, cycle)

        if(len(self.main_in_step) != 0 and cycle == -1):
            num_path = self.font.render(str(self.main_in_step), True, (0,0,0))
            loc = self.move_text_by_cuadrant(num_path, screen.get_width(), screen.get_height(), 30)
            screen.blit(num_path, loc)

class Edge:
    def __init__(self, font:pg.font.Font, relation:str, value:float, a:Node, b:Node):
        self.active_in_step, self.main_in_step = set(), set()
        self.is_active, self.is_main = False, False
        self.active_color, self.base_color = (136, 8, 8), (0, 0, 0) # red for active edges and black for the rest of the edges

        self.font, self.rel, self.value, = font, relation, value

        self.a, self.b = a, b

        self.ax, self.ay = a.circle_rect.centerx, a.circle_rect.centery
        self.bx, self.by = b.circle_rect.centerx, b.circle_rect.centery

    def run(self, cycle:int , screen:pg.surface.Surface):
        origin, dest = (self.ax, self.ay), (self.bx, self.by)
        
        linewidth, color = self.get_state(cycle)

        if(origin == dest):
            self.draw_self(screen, color, linewidth) # draws lines to itself for the NO_OP action.
        else:
           self.draw_straight(screen, color, linewidth, origin, dest) # draws a straight line to another edge.

    def get_state(self, cycle: int):
        self.is_main, self.is_active = False, False

        if(len(self.main_in_step) != 0): # is in main path:
            self.is_main = True
            return 4, self.active_color

        else: # not in main path
            if(cycle in self.active_in_step):
                self.is_active = True
                return 2, self.base_color
            else:
                return 1, self.base_color

    def draw_straight(self, screen:pg.surface.Surface, color, linewidth, origin, dest):
        dx = dest[0] - origin[0]
        dy = dest[1] - origin[1]
        dl = math.sqrt(dx**2 + dy**2)

        o, d = self.calculate_external_node_point(origin, dest, dx, dy, dl)       
        line = pg.draw.line(screen, color, o, d, linewidth)
        x, y, z = self.calculate_triangle(20, 20, o, d)
        direction_tip = pg.draw.polygon(screen, color, (x,y,z))

        if(self.is_main):
            self.render_text_rotated(screen, line, origin, True, dest, dy, dx, dl)
        elif(self.is_active):
            self.render_text_rotated(screen, line, origin, False, dest, dy, dx, dl)
    
    def render_text_rotated(self, screen, line, origin, fullinfo, dest, dy, dx, dl):
        alpha = -math.degrees(math.asin(dy/dl))
        if dest[0] < origin[0]:
            alpha = -alpha

        if(fullinfo):
            txt = f"{self.rel}-({self.value:.2f})"
        else:
            txt = f"({self.value:.2f})"

        text_img = self.font.render(txt, True, (0,0,0))
        text_img = pg.transform.rotate(text_img, alpha)

        x, y = line.center
        x = x - text_img.get_size()[0]/2
        y = y - text_img.get_size()[1]/2

        # normal_vec_normalized = -dy/dl, dx/dl
        gap = 12
        if(origin[0] < dest [0]):
            x += (-dy/dl)*gap
            y += (dx/dl)*gap
        else:
            x -= (-dy/dl)*gap
            y -= (dx/dl)*gap

        self.text_rect = screen.blit(text_img, (x,y))

    def calculate_external_node_point(self, origin, dest, dx, dy, dl):
        """
        Finds the point in the node circumpherence where the line collides for the origin and dest nodes.
        """
        # NO TOCAR.
        x_offset = int(dx/dl * NODERADIUS)
        y_offset = int(dy/dl * NODERADIUS)

        return (origin[0]+x_offset, origin[1]+y_offset), (dest[0]-x_offset, dest[1]-y_offset)

    def draw_self(self, screen:pg.surface.Surface, color, linewidth):
        w = screen.get_width()
        h = screen.get_height()

        x, y = self.ax, self.ay
        r = self.a.circle_rect.copy()

        if(x < w/2 and y > h/2): # bottom left
            r.move_ip((-15,15))
            start_angle, end_angle = 90, 0

        elif(x > w/2 and y > h/2): # bottom right
            r.move_ip((15,15))
            start_angle, end_angle = 180, 90

        elif(x > w/2 and y < h/2): # top right
            r.move_ip((15,-15))
            start_angle, end_angle = 270, 180
            
        elif(x < w/2 and y < h/2): # top left
            r.move_ip((-15,-15))
            start_angle, end_angle = 0, 270

        else: # centered 
            r.move_ip((-0, -15))
            start_angle, end_angle = 300, 240

        pg.draw.arc(screen, color, r ,math.radians(start_angle), math.radians(end_angle), linewidth)

        if self.is_active or self.is_main:
            text_img = self.font.render(f"{self.rel}-({self.value:.2f})", True, (0,0,0))
            x, y = text_img.get_size()
            a, b = r.center

            x = a-x/2
            y = b-y/2

            screen.blit(text_img, (x, y))

    def calculate_triangle(self, height, base, line_origin, line_dest):
        #calculate inverse vector:
        reverse_line_vector = np.array(line_origin) - np.array(line_dest)
        unit_line_vector = reverse_line_vector/np.linalg.norm(reverse_line_vector)


        # add to the origin the unit vector times the arrow tip distance:
        triangle_base_midpoint = line_dest + unit_line_vector*height
        unit_perp_vector = np.empty_like(unit_line_vector)
        unit_perp_vector[0] = -unit_line_vector[1]
        unit_perp_vector[1] = unit_line_vector[0]

        triangle_base_top = triangle_base_midpoint + unit_perp_vector*(base/2)
        triangle_base_bottom = triangle_base_midpoint - unit_perp_vector*(base/2)

        return line_dest, triangle_base_top.tolist(), triangle_base_bottom.tolist()

class SimpleText:
    def __init__(self, text:str, x:int, y:int, color: tuple):
        # Tophead text.
        self.x, self.y = x, y
        self.text, self.local_font = text, pg.font.SysFont("dejavuserif", 16)
        self.color = color
        
    def run(self, screen:pg.surface.Surface):
        txt_obj = self.local_font.render(self.text, True, self.color)
        w, h = txt_obj.get_width(), txt_obj.get_height()
        screen.blit(txt_obj, (self.x-(w/2), self.y-(h/2)) )