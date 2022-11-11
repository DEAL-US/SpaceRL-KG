import numpy as np
from numpy.linalg import norm
import logging
import random
import copy

from multiprocess import Process
from data.generator.generate_trans_embeddings import generate_embedding
from data.kg_structure import KnowledgeGraph
import gym
from gym import error, spaces
from data.data_manager import DataManager

from tqdm import tqdm
from itertools import chain

from utils import Utils

logger = logging.getLogger()

class KGEnv(gym.Env):
    '''
    Defines the environment and keeps track of the episodes that have been elapsed during training.
    We use the Gym package (https://github.com/openai/gym) as it offers a standard API to generate 
    environments and tie them up with keras agents

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

    ### Rewards
    - terminal rewards: +1 if agent reaches tail entity in determined time, -1 otherwise.

    - length based reward: 1-log(len(path)), discourages long paths, we punish the agent if it finds paths longer than 10 edges.

    - path diversity: we want to avoid redundant knowledge from being generated so we use the 
    cosine similarity comparing the sum of the path taken with the connected paths to the entity 
    and reward it based on how different is is from the known paths
    spatial.distance.cosine(average(known_paths), taken_path)

    ### Arguments

    Params (dict): All of the necessary variables for the network performance (unknow which ones yet.)
    Data (dict): which dataset and dataset embeeddings to use in experimentation. 
    

    '''
    def __init__(self, data_manager: DataManager, dataset, single_relation_pair, embedding, is_distance, seed, threads,
    path_length, regenerate_embeddings, normalize_embeddings, gpu_accel, use_episodes, laps, verbose):
        '''
        Initialize the environment with the desired batch size, the length of the path and the location of the dataset. \n
        ### Parameters:\n
            dataset (str) => The name of the folder containing the dataset file (graph.txt) \n
            embedding_index (int) => 0: TransE_l2, 1: DistMult, 2: ComplEx, 3: RotatE, 4:TransR, 5: RESCAL \n
            path_length (int) => the number of steps the agent will be allowed to do before stopping the episode \n
            queries (list) * => a list of triples to train, if not passed a random triple will be selected from the provided seed\n
        '''
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
        
        self.single_relation, self.relation_name = single_relation_pair
    
        if(self.single_relation):
            self.triples = [t for t in self.triples if t[1] == self.relation_name]

        self.reset()

    def step(self, action): # required by openAI.gym
        '''
        Recieves the chosen action in the format (chosen_relation, chosen_entity)
        then updates the state of the environment accordingly.
        '''
        self.current_node = action[1] # assign the selected action node as the new current node.
        self.state = self.get_encoded_state() # recalculate the state of the environment.
        self.update_actions() # update the actions in the new state.
        
        # Episode termination conditions
        self.episode_length -= 1
        if(self.episode_length == 0):
            self.done = True
        else:
            self.done = False
        
        # add the chosen action to the history.
        self.path_history.append((self.current_node, *action))

        # extra information relevant to the state.
        info = {}

        return self.state, self.done, info

    @property
    def action_space(self): # required by openAI.gym, must return a Spaces type from gym.spaces
        '''
        returns a Box space with dimensions relative 
        to the current embedding representations length

        since the actions are made up of the relation and the
        destination node we pass their mins and max as a Box space.
        '''
        return spaces.Box(low=np.array([*self.min_rel_emb, *self.min_ent_emb]), high = np.array([*self.max_rel_emb, *self.max_ent_emb]))
    
    @property
    def observation_space(self): # required by openAI.gym
        '''
        returns a Boxed space for the representation of the observations
        it encodes the positional triple et, then the 
        '''
        return spaces.Box(low=np.array([*self.min_ent_emb, *self.min_rel_emb, *self.min_ent_emb]),
        high=np.array([*self.max_ent_emb, *self.max_rel_emb, *self.max_ent_emb]))

    def reset(self):
        '''
        Resets the environment, selecting a new triple for the agent to obtain.
        '''
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

    def select_target(self):
        '''
        choose a new target triple to find.
        returns true if valid triple is found, false otherwise.
        '''

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
        
    def cache_init(self, dataset, is_distance):
        if(not is_distance):
            self.distance_cache = None
        else:
            try:
                self.utils.verb_print("Loading cached distance for dataset, this may take a while...")
                self.distance_cache = self.dm.get_cache_for_dataset(dataset)
            except FileNotFoundError:
                self.distance_cache = pairdict()

    def save_current_cache(self, dataset):
        self.dm.save_cache_for_dataset(dataset, self.distance_cache)
    
    def reset_queries(self):
        self.queries = copy.deepcopy(self.triples)
        random.shuffle(self.queries)

    def get_current_state(self):
        '''
        gets the current state 
        '''
        return self.target_triple[0], self.target_triple[1], self.target_triple[2], self.current_node

    def get_encoded_state(self):
        '''
        The encoding of the state is just the addition of the 
        embedding representation of the triple + the node position.
        return [*e1,*r,*e2,*et]
        '''
        e1 = self.entity_emb[self.target_triple[0]]
        r = self.relation_emb[self.target_triple[1]]
        e2 = self.entity_emb[self.target_triple[2]]
        et = self.entity_emb[self.current_node]

        return [*e1,*r,*e2,*et]

    def get_encoded_observations(self):
        '''
        The encoding of the observations is just the addition of the 
        embedding representation of the query + the node position.
        return [(*e1,*r),*et]
        '''
        e1 = self.entity_emb[self.target_triple[0]]
        r = self.relation_emb[self.target_triple[1]]
        et = self.entity_emb[self.current_node]

        return [*e1,*r,*et]

    def update_actions(self):
        '''
        When called updates all posible actions you can take in the current state
        and adds the them to the enviroment.actions list.
        '''
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

    # def calculate_reward(self, last_step):
    #     # moved to agent with get_next_state_reward.
    #     pass

    # def get_current_path_and_reward(self, extra_action = None):
    #     #removed
    #     pass

    def get_embedding_info(self, evaluated_node, end_node):
        a = np.array(self.entity_emb[evaluated_node])
        b = np.array(self.entity_emb[end_node])
        
        #Dot product: max = 655
        dot = np.dot(a, b)

        #Cosine similarity: range [0-1] 1 is best.
        cos_sim = dot/(norm(a)*norm(b))

        #Euclidean distance: min = 0
        euc_dist = norm(a-b)

        return(dot, euc_dist, cos_sim)
    
    def get_distance(self, current_node, end_node):
        '''
        given the current node calculate the minimum distance to the end node
        the maximum reward being in the end node and reducing by half for each step away.
        potential problem: if the graph is very interconnected this will be always high.
        '''
        if(current_node == end_node):
            return 0

        d = 1
        to_evaluate , visited = set(), set()
        done_flag = [False]
        to_evaluate.add(current_node)

        try:
            d = self.distance_cache[(current_node, end_node)]
            # print("value is cached!")
            return d
        except:
            while(not done_flag[0]):
                to_evaluate_next_step, thread_list = set(), []
                sublist_size = (len(to_evaluate)//self.threads)

                if(len(to_evaluate)<= self.threads):
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
                    d+=1
                    if(d >= self.path_length):
                        self.utils.verb_print("path, to end node is too large...")
                        return None

                if(len(to_evaluate)==0):
                    self.utils.verb_print("no paths connecting to the end node")
                    return None

            return d

    def dist_func(self,init_index, last_index, to_evaluate,
    d, done_flag, to_evaluate_next_step, visited, current_node, end_node):
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
        '''
        Iterates over the embedding representations of the entities and 
        relations and computes the minimum and maximum values to be used in
        a gym.Spaces.Box()\n
        returns:\n 
        [*mins_rel, *mins_ent], [*maxs_rel, *maxs_ent]
        '''
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

# d = DataManager()
# cache = pairdict()
# cache[("a","b")] = "c"
# d.save_cache_for_dataset("TEST", cache)
# cache = d.get_cache_for_dataset("TEST")
# print(cache)
# cache["t"] = "broken?"