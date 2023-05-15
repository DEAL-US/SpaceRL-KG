# Local imports
import sys, os, atexit
import multiprocessing as mp

from subprocess import run
from pathlib import Path

import utils

from utils import DATASETS, ALLOWED_EMBEDDINGS
from utils import permanent_config, changeable_config
from utils import Experiment, Test, Error, Triple, EmbGen, infodicttype
from utils import validate_test, validate_experiment, add_dataset, get_agents, get_info_from, send_message_to_handler
from utils import add_embedding, add_cache, add_experiment, add_test
from utils import convert_var_to_config_type

# FastAPI imports
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import HTTPException

from typing import Union, List, Dict

app = FastAPI()

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()
agents_path = Path(f"{parent_path}/model/data/agents").resolve()
datasets_path = Path(f"{parent_path}/datasets").resolve()

# Local threads:
embgen_process_queue = []

@app.get("/", response_class=PlainTextResponse)
def root() ->str:
    text = """

   _____                       ____  __         __ ________
  / ___/____  ____ _________  / __ \/ /        / //_/ ____/
  \__ \/ __ \/ __ `/ ___/ _ \/ /_/ / /  ______/ /< / / __  
 ___/ / /_/ / /_/ / /__/  __/ _, _/ /__/_____/ /| / /_/ /  
/____/ .___/\__,_/\___/\___/_/ |_/_____/    /_/ |_\____/   
    /_/                                                    


Hi! you are in the API for the SpaceRL-KG Framework.
try /docs to check all the things I can do!
"""
    return text



# CONFIG OPERATIONS
@app.get("/config/")
def get_config() -> dict:
    return changeable_config

@app.put("/config/")
def set_config(param:str, value) -> dict:
    # for p in changeable_config.keys():
    #     print(p, param)
    #     if p == param:
    #         print(f"these are equal {param}, {p}")

    if param not in changeable_config.keys():        
        Error("ParameterDoesNotExist",
        f"{param} is not a config param.")
    
    value = convert_var_to_config_type(param, value)    

    if(len(get_info_from(infodicttype.EXPERIMENT)) != 0 
       or len(get_info_from(infodicttype.TEST)) !=0):
        Error(name = "BusyResourcesError",
        desc = f"There are is an active test/train suite")

    changeable_config[param] = value

    return changeable_config


# DATASET OPERATIONS
@app.get("/datasets/")
def get_dataset():
    return {i.name: i.value for i in DATASETS}

@app.post("/datasets/")
def set_dataset(name:str, triples: List[Triple]):
    add_dataset(name, triples)
    return {"message":"dataset added successfully."}

# PUT and DELETE require much more work than these, might be added later...
# TODO: add background task that cleans up generated info for previous dataset.
# Delete Chaches, embedding files, agents and tests related to the DATASET being edited or deleted.


# EMBEDDING OPERATIONS
@app.get("/embeddings/")
def get_embeddings():
    return {i.name: i.value for i in ALLOWED_EMBEDDINGS}

@app.post("/embeddings/")
def gen_embedding(embedding: EmbGen):
    add_embedding(embedding)

# AGENTS
@app.get("/agents/")
def agents():
    return get_agents()



# EXPERIMENTS OPERATIONS
@app.get("/experiments/")
def get_experiment(id:int = None) -> Union[Dict[(int, Experiment)], Experiment]:
    if id is None:
        return get_info_from(infodicttype.EXPERIMENT)
    else:
        try:
            return get_info_from(infodicttype.EXPERIMENT, id)
        except:
            Error(name="NonexistantExperiment",
            desc = f"There is no experiment with id {id}")

@app.post("/experiments/") 
def add_exp(experiment:Experiment) -> Dict[(int, Experiment)]:
    validate_experiment(experiment)
    add_experiment(experiment)

    
@app.delete("/experiments/") 
def remove_experiment(id:int):
    try:
        send_message_to_handler(f"delete;experiment;{id}")
        # del experiment_queue[id]
        return {"message":"experiment was removed successfully."}

    except:
        Error(name="NonexistantExperiment",
        desc = f"There is no experiment with id {id}")



# TEST OPERATIONS
@app.get("/tests/")
def get_test(id:int = None) -> Union[Dict[(int, Test)], Test]:
    if id is None:
        return get_info_from(infodicttype.TEST)
    else:
        try:
            return get_info_from(infodicttype.TEST, id)
        except:
            Error(name="NonexistantTest",
            desc = f"There is no test with id {id}")

@app.post("/tests/") 
def add_tst(test:Test) -> Dict[(int, Test)]:
    validate_test(test)
    add_test(test)
    
@app.delete("/tests/") 
def remove_test(id:int):
    try:
        send_message_to_handler(f"delete;test;{id}")

        # del test_queue[id]
        return {"message":"test was removed successfully."}
    except:
        Error(name="NonexistantTest",
        desc = f"There is no test with id {id}")


if __name__ == "__main__":
    command = f"uvicorn main:app --reload".split()
    run(command, cwd = current_dir)