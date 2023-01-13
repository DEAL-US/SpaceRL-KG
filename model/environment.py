import numpy as np
import networkx as nx
from numpy.linalg import norm
import logging
import random
import copy
import gym

from multiprocess import Process
from data.generator.generate_trans_embeddings import generate_embedding
from data.kg_structure import KnowledgeGraph
from gym import error, spaces
from data.data_manager import DataManager

from tqdm import tqdm
from itertools import chain

from utils import Utils

logger = logging.getLogger()

class KGEnv(gym.Env):
    """
    Defines the environment and keeps track of the episodes that have been elapsed during training.
    We use the Gym package (https://github.com/openai/gym) as it offers a standard model to generate 
    environments and tie them up with keras agents.

    ### Description
    The environment is the currently selected knowledge graph built with inverse relationships

    The agent is positioned in the initial node that we want to connect via a certain relationship and then
    we act over the graph moving the agents location trying to find the end entity for the triple.

    ### Action space
    The actions vary depending on the current node we are in, we can take any edge connected 
    to this node as an action moving the agent to the next node, actions are pairs of ("relation","node")

    ### Observation space
    Since the complete state of the graph cannot be observed by the agent due to the size limitations
    we make the agent aware of its current location (et) and  the query triple(e1q, rq, e?), but not the answer (e2q)

    :param data_manager: The data manager instance asociated 
    :param dataset: The name of the folder containing the dataset file (graph.txt)
    :param single_relation_pair: (is_single_rel(bool), single_rel_name(str)), the tuple representing if we are training for a single relation.
    :param embedding: the embedding representation to load.
    :param is_distance: if the reward calculation contains distance.
    :param seed: the seed to operate on
    :param threads: number of cores to use on calculations.
    :param path_length: the length of path exploration.
    :param regenerate_embeddings: wether to regenerate embeddings or use the saved ones.
    :param normalize_embeddings: if recalculation is active, wether to normalize them.
    :param gpu_accel: wether to use the gpu for calculations.
    :param use_episodes: wether to use a set number of episodes or not
    :param laps: the loops to perform over the dataset.
    :param verbose: wether to print detailed information every episode.

    """
    def __init__(self, data_manager: DataManager, dataset:str, single_relation_pair:tuple, embedding:str, is_distance:bool, seed:int, threads:int,
    path_length:int, regenerate_embeddings:bool, normalize_embeddings:bool, gpu_accel:bool, use_episodes:bool, laps:int, verbose:bool):
        self.dataset_name = dataset 
        
        # generation of new dataset embedding
        self.selected_embedding_name = embedding
        self.path_length = path_length

        self.utils = Utils(verbose, False)

        self.threads = threads
        self.remaining_laps = laps
        self.use_episodes = use_episodes
        self.dm = data_manager

        # for repetitions sake.
        random.seed(seed)
        np.random.seed(seed)
        self.queries = []

        # if embedding does not exist we generate it.
        generate_embedding(dataset, models = [self.selected_embedding_name],
        add_inverse_path=True, regenerate_existing=regenerate_embeddings, 
        normalize = normalize_embeddings, use_gpu = gpu_accel)

        # get the selected dataset triples
        self.triples, self.relation_emb, self.entity_emb, self.embedding_len = self.dm.get_dataset(dataset, self.selected_embedding_name)
        self.min_ent_emb, self.min_rel_emb, self.max_ent_emb, self.max_rel_emb = self.calculate_embedding_min_max()
        self.cache_init(dataset, is_distance)

        # instantiate the corresponding KG() from input_dir
        self.kg = KnowledgeGraph(self.triples, directed=True, inverse_triples=True)
        self.netX_KG = self.create_networkX_graph()

        self.single_relation, self.relation_name = single_relation_pair
    
        if(self.single_relation):
            self.relation_name = self.relation_name.replace("\'", "")
            print(f"pruning triples where relation is not {self.relation_name}")          
            self.triples = [t for t in self.triples if self.relation_name in t[1]] 

            if(len(self.triples) == 0):
                print(f"there are no triples with relation {self.relation_name}, please try again...")
                quit()

        self.reset()

    #######################
    #    GYM FUCNTIONS    #
    #######################

    def step(self, action:list): # required by openAI.gym
        """
        Performs one environment step, determined by the recieved action.

        :param action: the action in its triple format

        :returns:
        state -> the current state after the step is performed. \n
        done -> if the episode is done. \n
        info -> empty dict (conforming with gym, but we provide no extra info.) \n

        """
        # add the chosen action to the history.
        self.path_history.append((self.current_node, *action))

        self.current_node = action[1] # assign the selected action node as the new current node.
        self.state = self.get_encoded_state() # recalculate the state of the environment.
        self.update_actions() # update the actions in the new state.
        
        # Episode termination conditions
        self.episode_length -= 1
        if(self.episode_length == 0):
            self.done = True
        else:
            self.done = False

        # extra information relevant to the state.
        info = {}

        return self.state, self.done, info

    def reset(self):
        """
        resets the environment for the next episode.

        :returns: the state after the reset.
        """
        valid = False
        # updates the target_triple variable.
        while(not valid):
            valid = self.select_target()

        # The embedding representation of the state.
        self.state = self.get_encoded_state()

        # updates the actions list based on the current state.
        self.update_actions()

        # The number of steps it should take for the agent to find the desired path, 
        # if it fails to do so in this period we reward the agent negatively.
        self.episode_length = self.path_length
        # The actions that have been picked by the agent, is a list of triples so we can traceback.
        self.path_history = []
        self.done = False

        return self.state 

    @property
    def action_space(self): # required by openAI.gym, must return a Spaces type from gym.spaces
        """
        Returns the action space of the environment, which is the minimum and maximum values of the embeddings.
        These do not all represent possible actions, but the scope in which actions can be contained.

        :returns: A boxed state (gym.spaces) of all possible actions in the current environment.
        
        """
        return spaces.Box(low=np.array([*self.min_rel_emb, *self.min_ent_emb]), high = np.array([*self.max_rel_emb, *self.max_ent_emb]))
    
    @property
    def observation_space(self): # required by openAI.gym
        """
        Returns the observation space of the environment, which is the minimum and maximum values of the embeddings.
        These do not all represent possible observation, but the scope in which observations can be contained.

        :returns: A boxed space(gym.spaces) of all possible observations in the current environment.
        """
        return spaces.Box(low=np.array([*self.min_ent_emb, *self.min_rel_emb, *self.min_ent_emb]),
        high=np.array([*self.max_ent_emb, *self.max_rel_emb, *self.max_ent_emb]))

    #########################
    #    CACHE FUNCTIONS    #
    #########################

    def cache_init(self, dataset:str, is_distance:bool):
        """
        initializes the cache for distance rewards for the selected dataset.

        :param dataset: the name of the dataset
        :param is_distance: if true tries to load cache, sets cache to None otherwise.
        
        """
        if(not is_distance):
            self.distance_cache = None
        else:
            try:
                self.utils.verb_print("Loading cached distance for dataset, this may take a while...")
                self.distance_cache = self.dm.get_cache_for_dataset(dataset)
                if(self.distance_cache is None):
                    self.distance_cache = pairdict()
            except:
                self.distance_cache = pairdict()

        print(f"distance cache after init: {self.distance_cache}" )
        
    def save_current_cache(self, dataset:str):
        """
        saves the dataset cache.

        :param dataset: the dataset name
        """
        self.dm.save_cache_for_dataset(dataset, self.distance_cache)

    #######################################
    #    STATES, OBSERVATIONS & ACTIONS   #
    #######################################

    def create_networkX_graph(self):
        G = nx.MultiDiGraph()
        for t in self.triples:
            G.add_node(t[0])
            G.add_node(t[2])

            G.add_edge(t[0], t[2], key = t[1])
            G.add_edge(t[2], t[0], key = f"¬{t[1]}")

        for n in G.nodes():
            G.add_edge(n, n, key="NO_OP")

        print(f"triples: {len(self.triples)}, nodes:{G.number_of_nodes()}, edges:{G.number_of_edges()}")
        return G

    def get_current_state(self):
        """
        returns the current state of the environment.

        :returns: 
        self.target_triple[0] -> e1 \n
        self.target_triple[1] -> r \n
        self.target_triple[2] -> e2 \n
        self.current_node -> et \n
        """
        return self.target_triple[0], self.target_triple[1], self.target_triple[2], self.current_node

    def get_encoded_state(self):
        """
        The encoding of the state is just the addition of the embedding representation of the triple + the node position.

        :returns: [*e1,*r,*e2,*et]
        """
        e1 = self.entity_emb[self.target_triple[0]]
        r = self.relation_emb[self.target_triple[1]]
        e2 = self.entity_emb[self.target_triple[2]]
        et = self.entity_emb[self.current_node]

        return [*e1,*r,*e2,*et]

    def get_encoded_observations(self):
        """
        The encoding of the observations is just the addition of the embedding representation of the query + the node position.

        :returns: [(*e1,*r),*et]
        """
        e1 = self.entity_emb[self.target_triple[0]]
        r = self.relation_emb[self.target_triple[1]]
        et = self.entity_emb[self.current_node]

        return [*e1,*r,*et]
    
    def select_target(self):
        """
        choose a new target triple to find. 

        :returns: true if valid triple is found, false otherwise.
        """

        if(self.use_episodes):
            self.target_triple = random.choice(self.triples)
        else:
            if len(self.queries) == 0: 
                self.reset_queries()

            self.target_triple = self.queries.pop(0)

        neighbors = self.kg.get_neighbors(self.target_triple[0])
        neighbors = set(chain(*neighbors.values()))
        if(len(neighbors) == 1):
            self.utils.verb_print(f"chosen non valid target entity: {self.target_triple}, no valid neighbors: {neighbors}")
            return False
        
        self.current_node = self.target_triple[0]
        # print(f"Episode triple is: {self.target_triple}")

        return True
        
    def reset_queries(self):
        """
        resets the queries and randomizes them
        """
        self.queries = copy.deepcopy(self.triples)
        random.shuffle(self.queries)

    def update_actions(self):
        """
        Updates the actions for the new step
        """
        neighbors = copy.deepcopy(self.kg.get_neighbors(self.current_node))
        if(self.current_node == self.target_triple[0]):
            # if we are training and we are in the initial node we remove the direct relation
            # to the end tripleas it would be the best option everytime.
            neighbors[self.target_triple[1]].remove(self.target_triple[2])

            # if the result of deleting the entity is an empty set delete the relation too
            if(len(neighbors[self.target_triple[1]]) == 0):
                del neighbors[self.target_triple[1]]

        self.actions = []
        self.actions.append(("NO_OP", self.current_node))  # Add the no_op to stay in the current node.
        for pair in neighbors.items():
            relation = pair[0]
            for dest_entity in list(pair[1]):
                action = (relation, dest_entity)
                self.actions.append(action)
    
    #############################
    #    REWARDS & EMBEDDINGS   #
    #############################
    def get_distance_net_x(self, origin_node:str, dest_node:str, excluded_rel:str = None):
        """
        computes the minimum distance from the origin node to the end node.
        A relation can be exculded from the search.
        Checks the cache for the requested distance and adds it if its not there.

        :param origin_node: the origin node to calculate.
        :param dest_node: the destination node to reach.
        :param exculded_rel: the relation to ignore when calculating the distance if it exists between the nodes.
        
        :returns: the distance from the origin node to the destination node.
        """
        # check cache for distance.
        if (origin_node, dest_node) in self.distance_cache:
            return self.distance_cache[(origin_node, dest_node)]

        # Remove conections from excluded.
        if(excluded_rel is not None):
            self.netX_KG.remove_edge(origin_node, dest_node, key=excluded_rel)
            self.netX_KG.remove_edge(dest_node, origin_node, key=f"¬{excluded_rel}")

        # calculate info.
        lengths = nx.single_source_shortest_path_length(self.netX_KG, origin_node, cutoff=self.path_length)
        l_keys = lengths.keys()
        l_items = lengths.items()
        
        # reiuntroduce deleted edges.
        if(excluded_rel is not None):
            self.netX_KG.add_edge(origin_node, dest_node, key=excluded_rel)
            self.netX_KG.add_edge(dest_node, origin_node, key=f"¬{excluded_rel}")

        # if not in cache, add all calcualted nodes to cache.
        for k, v in l_items:
            self.distance_cache[origin_node, k] = v
        
        # return distance if it exists, None if non connected.
        if dest_node in l_keys:
            return lengths[dest_node]
        else: 
            return None

    def get_distance(self, current_node:str, end_node:str):
        """
        given the current node calculate the minimum distance to the end node.

        :param current_node: the current node in the environment
        :param end_node: the target node.

        :returns: the distance to the end node from the current node.
        """
        if(current_node == end_node):
            return 0

        d = 1
        to_evaluate , visited = set(), set()
        done_flag = [False]
        to_evaluate.add(current_node)

        if (current_node, end_node) in self.distance_cache:
            d = self.distance_cache[(current_node, end_node)]
            return d

        else:
        
            while not done_flag[0]:
                to_evaluate_next_step, thread_list = set(), []
                sublist_size = (len(to_evaluate)//self.threads)

                if len(to_evaluate) <= self.threads:
                    self.dist_func(0, len(to_evaluate), to_evaluate, d, done_flag,
                    to_evaluate_next_step, visited, current_node, end_node)
                else:
                    for i in range(self.threads):
                        init_index = (sublist_size*i)
                        last_index = len(to_evaluate) if self.threads == i+1 else sublist_size*(i+1)-1
                        # print(f"({init_index},{last_index})")

                        x = Process(target=self.dist_func, 
                        args=(init_index, last_index, to_evaluate, d, done_flag,
                        to_evaluate_next_step, visited, current_node, end_node))

                        thread_list.append(x)
                        x.start()
                    
                for t in thread_list:
                    t.join()       

                if(not done_flag[0]):
                    to_evaluate = to_evaluate_next_step-visited
                    d += 1
                    if(d >= self.path_length):
                        self.utils.verb_print("path, to end node is too large...")
                        return None

                if(len(to_evaluate)==0):
                    self.utils.verb_print("no paths connecting to the end node")
                    return None

            return d

    def dist_func(self, init_index:int, last_index:int, to_evaluate:list,
    d:int, done_flag: bool, to_evaluate_next_step: list, visited:list, current_node:str, end_node:str):
        """
        helper function to calculate the distance to the end node.
        
        :param init_index: starting point in list of nodes to evaluate.
        :param last_index: end point in list of nodes to evaluate
        :param to_evaluate: the current list of nodes to get the neighbors to.
        :param d: current distance from starting node.
        :param done_flag: true if we reached the last node.
        :param to_evaluate_next_step: the next iteration list of nodes to evaluate.
        :param visited: the list of visited nodes.
        :param current_node: the current node of exploration
        :param end_node: the destination node.

        """
        for node in list(to_evaluate)[init_index:last_index]:
            visited.add(node)
            neighbors = copy.deepcopy(self.kg.get_neighbors(node))

            if(node == self.target_triple[0]):
                # remove the direct relation to the end node.
                neighbors[self.target_triple[1]].remove(self.target_triple[2])
                # if the relation is now disconnected eliminate it completely.
                if(len(neighbors[self.target_triple[1]]) == 0):
                    del neighbors[self.target_triple[1]]

                self.distance_cache[(current_node, self.target_triple[2])] = 1

            neighbors = set(chain(*neighbors.values()))

            for n in neighbors: 
                self.distance_cache[(node, n)] = 1
                self.distance_cache[(current_node, n)] = d

            to_evaluate_next_step.update(neighbors)

            if(end_node in neighbors):
                done_flag[0] = True
                break
        
    def calculate_embedding_min_max(self):
        """
        Iterates over the embedding representations of the entities and relations and computes the minimum and maximum values to be used in a gym.Spaces.Box()

        :returns: mins_rel, mins_ent, maxs_rel, maxs_ent (the minimum and maximum values for entities and relations.)
        """
        mins_rel = list(np.zeros(self.embedding_len,))
        maxs_rel = list(np.zeros(self.embedding_len,))
        mins_ent = list(np.zeros(self.embedding_len,))
        maxs_ent = list(np.zeros(self.embedding_len,))

        print("calculating embedding minimums and maximums...")
        rels = self.relation_emb.items()
        ents = self.entity_emb.items()

        for pair in tqdm(rels):
            try:
                for i, d in enumerate(pair[1]):
                    if d > maxs_rel[i]:
                        maxs_rel[i] = d
                    
                    if d < mins_rel[i]:
                        mins_rel[i] = d
            except Exception as e:
                print(f" broke execution at: {pair}\n on index {i}, max_rel len is {len(maxs_rel)}, rel_emb len is {len(pair[1])}")
                raise(e)

        for pair in tqdm(ents):
            for i, d in enumerate(pair[1]):
                if d > maxs_ent[i]:
                    maxs_ent[i] = d
                
                if d < mins_ent[i]:
                    mins_ent[i] = d
        

        return mins_ent, mins_rel, maxs_ent, maxs_rel

    def get_embedding_info(self, evaluated_node:str, end_node:str):
        """
        Calculate the embedding rewards for the current node.

        :param evaluated_node: the current node of exploration.
        :param end_node: the destination node.

        :returns: this is a description of what is returned
        """
        a = np.array(self.entity_emb[evaluated_node])
        b = np.array(self.entity_emb[end_node])
        
        #Dot product: max = 655
        dot = np.dot(a, b)

        #Cosine similarity: range [0-1] 1 is best.
        cos_sim = dot/(norm(a)*norm(b))

        #Euclidean distance: min = 0
        euc_dist = norm(a-b)

        return(dot, euc_dist, cos_sim)
   
class pairdict(dict):
    "Extends the basic python dict to only accept pairs as keys"
    def __init__(self, *args):
        super().__init__(self, *args)

    def __getitem__(self, key):
        self.tuple_check(key)
        a, b = key
        if b < a:
            key = (b, a)
        return super().__getitem__(key)

    def __setitem__(self, key, val):
        self.tuple_check(key)
        a, b = key
        if b < a:
            key = (b, a)
        super().__setitem__(key, val)

    def tuple_check(self, key):
        if(type(key) is not tuple or len(key) != 2):
            raise ValueError("given key is not a tuple")
