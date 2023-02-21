# Local imports
import sys, os, time

from enum import Enum
from pathlib import Path
from typing import Union, List
from pydantic import BaseModel
import GPUtil as gputil

from fastapi import HTTPException

from multiprocessing import Process, Manager, cpu_count

import networkx as nx

# Process queues:
manager = Manager()

embgen_queue, cache_queue, experiment_queue, test_queue = manager.dict(), manager.dict(), manager.dict(), manager.dict()
embgen_idx, cache_idx, exp_idx, test_idx = 0,0,0,0


# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()


# RELATIVE IMPORTS.
genpath = Path(f"{parent_path}/model/data/generator").resolve()

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, str(genpath))

import generate_trans_embeddings as embgen

def QueueProcessHandler():
    """
    Handles the queues 
    """
    nxt_eg, nxt_chq, nxt_expq, nxt_tstq = None, None, None, None
    while True:
        if embgen_queue: nxt_eg = next(iter(embgen_queue.items()))
        if cache_queue: nxt_chq = next(iter(cache_queue.items()))
        if experiment_queue: nxt_expq = next(iter(experiment_queue.items()))
        if test_queue: nxt_tstq = next(iter(test_queue.items()))

        print(nxt_eg, nxt_chq, nxt_expq, nxt_tstq)
        print(embgen_queue, cache_queue, experiment_queue, test_queue)

        time.sleep(2)

    
    # p = mp.Process(target=embgen.generate_embedding,
    # args=(dataset.name, models, use_gpu,regenerate_existing, normalize, add_inverse_path, fast_mode))
    # embgen_queue[len(embgen_queue)] = p

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()
agents_path = Path(f"{parent_path}/model/data/agents").resolve()
tests_path = Path(f"{parent_path}/model/data/results").resolve()
datasets_path = Path(f"{parent_path}/datasets").resolve()

# Helper function required before definition of pydantic models.
# As it defines the dataset Enum.
def get_datasets() -> Enum:
    res = {}
    for name in os.listdir(datasets_path):
        dirpath = Path(f"{datasets_path}/{name}").resolve()
        if os.path.isdir(dirpath):
            res[name] = name

    res = Enum('DATASETS', res)
    return res

DATASETS = get_datasets()
ALLOWED_EMBEDDINGS = Enum("ALLOWED_EMBEDDINGS",
{"TRANS-E":"TransE_l2", "DIST-MULT":"DistMult",
"COMPLEX":"ComplEx", "TRANS-R":"TransR"})

# Config.
permanent_config = {
    "gpu_acceleration": True, # wether to use GPU(S) to perform fast training & embedding generation.
    "multithreaded_dist_reward": False, 
    "verbose": False, # prints detailed information every episode.
    "debug": False, # offers information about crashes, runs post-mortem
    "print_layers": False, # if debug is active wether to print the layer wheights on crash info.
    "restore_agent": False, # continues the training from where it left off and loads the agent if possible.
    "log_results": False, # Logs the results in the logs folder of episode training.
    "use_episodes": False, 
    "episodes":0, # number of episodes to run.
}

changeable_config = {
    "available_cores": cpu_count(), #number of cpu cores to use when computing the reward
    "guided_reward": True, # wether to follow a step-based reward or just a reward at the end of the episode.
    "guided_to_compute":["terminal", "embedding"], #"distance","terminal","embedding"
    "regenerate_embeddings":False, # if embedding is found and true re-calculates them.
    "normalize_embeddings":True,
    "use_LSTM": True, # wether to add LSTM layers to the model

    "alpha": 0.9, # [0.8-0.99] previous step network learning rate (for PPO only.) 
    "gamma": 0.99, # [0.90-0.99] decay rate of past observations for backpropagation
    "learning_rate": 1e-3, #[1e-3, 1e-5] neural network learning rate.

    "activation":'leaky_relu', # relu, prelu, leaky_relu, elu, tanh
    "regularizers":['kernel'], #"kernel", "bias", "activity" 
    "algorithm": "PPO", #BASE, PPO
    "reward_type": "simple", # retropropagation, simple
    "action_picking_policy":"probability",# "probability", "max"
    "reward_computation": "one_hot_max", #"max_percent", "one_hot_max", "straight"

    "path_length":5, #the length of path exploration.
    "random_seed":True, # to repeat the same results if false.
    "seed":78534245, # sets the seed to this number.
}

# Pydantic models

# FastAPI is built with these pydantic models models in mind as a return,
# the idea is to define one of these for each response type you provide.
class Error():
    def __init__(self, name:str, desc:str):
        raise HTTPException(status_code=400, 
        detail=f"{name} - {desc}")

class Triple(BaseModel):
    e1: str
    r: str
    e2: str

class Experiment(BaseModel):
    name: str
    dataset: DATASETS
    single_relation: bool
    embedding : ALLOWED_EMBEDDINGS
    laps : int
    relation_to_train : Union[str, None]

class Test(BaseModel):
    name: str
    agent_name :str
    episodes: int
    embedding : Union[ALLOWED_EMBEDDINGS, None]
    dataset: Union[DATASETS, None]
    single_relation: Union[bool, None]
    relation_to_train : Union[str, None]

class EmbGen(BaseModel):
    dataset: DATASETS
    models:List[ALLOWED_EMBEDDINGS] = []
    use_gpu:bool = gputil.getAvailable()!= 0
    regenerate_existing:bool = False
    normalize:bool = True
    add_inverse_path:bool = True
    fast_mode:bool = False

# Main Functions
def validate_experiment(exp: Experiment):
    reasons = []
    agents = get_agents()
    
    if len(exp.name) > 200 and len(exp.name) < 10:
        reasons.append("Experiment name must be between 10-200 characters\n")

    f = list(filter(lambda l_exp: True if l_exp.name == exp.name else False, experiment_queue.values()))

    if exp.name in agents or len(f) != 0:
        reasons.append("Agent already exists, try a different name.\n")

    # dataset value can't be outside of enum scope, no need for extra validation
    
    if exp.single_relation == True:
        if exp.relation_to_train is None:
            reasons.append("Provide a relation name if you wish to train for it.\n")
        
        elif not check_for_relation_in_dataset(exp.dataset, exp.relation_to_train):
            reasons.append("Provided relation is not in dataset.\n")
    
    if exp.laps < 10 or exp.laps > 10_000:
        reasons.append("laps must be from 10-10.000\n")

    if len(reasons) !=0: Error(name = "ExperimentError", desc = "".join(reasons))
    
def validate_test(tst: Test):
    reasons = []
    tests = get_tests()
    
    if len(tst.name) > 200 and len(tst.name) < 10:
        reasons.append("Test name must be between 10-200 characters\n")

    f = list(filter(lambda l_tst: True if l_tst.name == tst.name else False, test_queue.values()))

    if tst.name in tests or len(f) != 0:
        reasons.append("Test with that name already exists, try again.\n")

    # dataset value can't be outside of enum scope, no need for extra validation

    if tst.episodes < 10 or tst.episodes > 10_000:
        reasons.append("episodes must be from 10-10.000\n")
    
    try:
        config_used = open(f"{agents_path}/{tst.agent_name}/config_used.txt")
        for ln in config_used.readlines():
            if ln.startswith("dataset: "):
                dataset = ln.lstrip("dataset: ").strip()

            if ln.startswith("embedding: "):
                embedding = ln.lstrip("embedding: ").strip()

            if ln.startswith("single_relation_pair: "):
                aux = ln.lstrip("single_relation_pair: ").strip()
                aux = aux.replace("[", "").replace("]","").replace(" ", "").replace("\'", "").strip().split(",")
                single_relation, relation_to_train = [aux[0]=="True", None if aux[1] == "None" else aux[1]]
    except:
        reasons.append(f"Incoherent Test detected, agent name was: {tst.agent_name}\n")
    
    f = list(filter(lambda dtst: True if dataset == dtst.value else False, DATASETS))

    if(len(f) == 0):
        reasons.append(f"no valid dataset detected for test, have you deleted it?\n")

    if(single_relation and relation_to_train is None):
        reasons.append(f"test is marked as single relation and None was provided.\n")

    # Raises error if reasons is not empty.
    if len(reasons) !=0: Error(name = "ExperimentError", desc = "".join(reasons))
    tst.embedding = embedding
    tst.dataset = dataset
    tst.single_relation = single_relation
    tst.relation_to_train = relation_to_train

    return tst

def add_dataset(dataset_name:str, triples: List[Triple]) -> Union[None, Error]:
    p = Path(f"{datasets_path}/{dataset_name}")
    try:
        os.mkdir(p)
    except:
        Error(name="DatasetExists",
        desc="Dataset with that name already exists, try again.")
    
    with open(Path(f"{p}/graph.txt"), "w") as f:
        res = ""
        for t in triples:
            res += f"{t.e1}\t{t.r}\t{t.e2}\n"
        f.write(res)
        
# Helper functions:
def get_agents():
    agent_list = os.listdir(agents_path)
    agent_list.remove('.gitkeep')
    agent_list.remove('TRAINED')
    return agent_list

def get_tests():
    test_list = os.listdir(tests_path)
    test_list.remove('.gitkeep')
    return test_list

def check_for_relation_in_dataset(dataset_name:str, relation_name:str):
    relation_in_graph = False
    remove_prefix = lambda text, prefix: text[len(prefix):] if text.startswith(prefix) else text
    dataset_name = remove_prefix(str(dataset_name), "DATASETS.")
    filepath = Path(f"{datasets_path}/{dataset_name}/graph.txt").resolve()

    with open(filepath) as d:
        for l in d.readlines():
            if(l.split("\t")[1] == relation_name):
                relation_in_graph = True
                break
    
    return relation_in_graph

