# Local imports
from subprocess import run
import multiprocessing
from pathlib import Path

# FastAPI imports
from enum import Enum

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import HTTPException

from pydantic import BaseModel
from typing import Union, List

app = FastAPI()

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()
agents_path = Path(f"{parent_path}/model/data/agents").resolve()
datasets_path = Path(f"{parent_path}/datasets").resolve()

# TODO replace for get_datasets() function.
DATASETS = Enum('DATASETS', ['COUNTRIES', 'UMLS', 'KINSHIP'])
ALLOWED_EMBEDDINGS = []

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

EXPERIMENTS, TESTS = {}, {}
exp_idx, test_idx = 0, 0

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
    embeddings : List[str]
    laps : int
    relation_to_train : Union[str, None]

class Test(BaseModel):
    name: str
    agent_name :str
    episodes: int
    embeddings : List[str]
    dataset: DATASETS
    single_relation: bool
    relation_to_train : Union[str, None]


@app.get("/", response_class=PlainTextResponse)
def root():
    text = """
         __             __          __                  __             __ 
        /\ \           /\ \        / /\                /\ \           /\ \ 
       /  \ \         /  \ \      / /  \              /  \ \         /  \ \ 
      / /\ \_\       / /\ \ \    / / /\ \            / /\ \ \       / /\ \ \ 
     / / /\/_/      / / /\ \_\  / / /\ \ \          / / /\ \ \     / / /\ \_\ 
    / / / ______   / / /_/ / / / / /  \ \ \        / / /  \ \_\   / /_/_ \/_/ 
   / / / /\_____\ / / /__\/ / / / /___/ /\ \      / / /    \/_/  / /____/\ 
  / / /  \/____ // / /_____/ / / /_____/ /\ \    / / /          / /\____\/ 
 / / /_____/ / // / /\ \ \  / /_________/\ \ \  / / /________  / / /______ 
/ / /______\/ // / /  \ \ \/ / /_       __\ \_\/ / /_________\/ / /_______\ 
\/___________/ \/_/    \_\/\_\___\     /____/_/\/____________/\/__________/ 

Hi! you are in the API for the GRACE Framework.
try /docs to check all the things I can do!
"""
    return text

@app.get("/config/")
def get_config() -> dict:
    return changeable_config

@app.put("/config/")
def set_config(param:str, value:Union[str, None] = Query()) -> dict:
    if(param is None or value is None):
        return Error(name = "MissingValueError",
        desc = f"One or more parameters are missing, param = {param}, value = {value}")

    if param not in changeable_config.keys():
        return Error(name = "ParameterDoesNotExist",
        dest = f"{param} is not a config param.")
    
    t = type(changeable_config[param])
    if  t != type(value):
        return Error(name = "TypeMismatchError",
        desc = f"{value} must be of type {t}, was {type(value)} instead.")

    changeable_config[param] = value

    return changeable_config

@app.get("/experiments/")
def get_experiments():
    return EXPERIMENTS

@app.post("/experiments/")
def add_experiment(experiment:Experiment):
    #TODO: check that the supplied experiment values are valid...


    EXPERIMENTS[exp_idx] = experiment

    return EXPERIMENTS

if __name__ == "__main__":
    command = f"uvicorn main:app --reload".split()
    run(command, cwd = current_dir)