# Local imports
import multiprocessing, os

from enum import Enum
from pathlib import Path
from typing import Union, List
from pydantic import BaseModel

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

EXPERIMENTS, TESTS = {}, {}

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
    "available_cores": multiprocessing.cpu_count(), #number of cpu cores to use when computing the reward
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
class Error(BaseModel):
    name: str
    desc: str

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
    embedding : ALLOWED_EMBEDDINGS
    dataset: DATASETS
    single_relation: bool
    relation_to_train : Union[str, None]

# Helper functions:
def validate_experiment(exp:Experiment):
    print(EXPERIMENTS)

    reasons = []
    agents = get_agents()
    
    if len(exp.name) > 200 and len(exp.name) < 10:
        reasons.append("Experiment name must be between 10-200 characters\n")

    f = list(filter(lambda l_exp: True if l_exp.name == exp.name else False, EXPERIMENTS.values()))

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

    

    return (True, None) if len(reasons) == 0 else (False, Error(name = "ExperimentError", desc = "".join(reasons)))
    
def validate_test():
    pass

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