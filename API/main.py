# Local imports
from subprocess import run
import multiprocessing
from pathlib import Path

from utils import DATASETS, ALLOWED_EMBEDDINGS
from utils import EXPERIMENTS, TESTS
from utils import permanent_config, changeable_config
from utils import Experiment, Test, Error, Triple
from utils import validate_test, validate_experiment, add_dataset

# FastAPI imports

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import HTTPException

from typing import Union, List, Dict

app = FastAPI()

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()
agents_path = Path(f"{parent_path}/model/data/agents").resolve()
datasets_path = Path(f"{parent_path}/datasets").resolve()

exp_idx, test_idx = 0, 0


@app.get("/", response_class=PlainTextResponse)
def root() ->str:
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

# CONFIG OPERATIONS
@app.get("/config/")
def get_config() -> dict:
    return changeable_config

@app.put("/config/")
def set_config(param:str, value) -> dict:
    if param not in changeable_config.keys():
        Error(name = "ParameterDoesNotExist",
        dest = f"{param} is not a config param.")
    
    t = type(changeable_config[param])
    if  t != type(value):
        Error(name = "TypeMismatchError",
        desc = f"{value} must be of type {t}, was {type(value)} instead.")

    changeable_config[param] = value

    return changeable_config

# DATASET OPERATIONS
@app.get("/datasets/")
def get_dataset():
    return DATASETS

@app.post("/datasets/")
def set_dataset(name:str, triples: List[Triple]):
    add_dataset(name, triples)
    return {"dataset added successfully."}

# PUT and DELETE require much more work than these, might be added later...
# TODO: add background task that cleans up generated info to previous dataset.
# Delete Chaches, embedding files, agents and tests related to the DATASET being edited or deleted.

# EMBEDDING OPERATIONS
@app.get("/embeddings/")
def get_embeddings() -> Dict[(str, str)]:
    return ALLOWED_EMBEDDINGS

# EXPERIMENTS OPERATIONS
@app.get("/experiments/")
def get_experiment(id:int = None) -> Dict[(int, Experiment)]:
    if id is None:
        return EXPERIMENTS
    else:
        try:
            return EXPERIMENTS[id]
        except:
            Error(name="NonexistantExperiment",
            desc = f"There is no experiment with id {id}")

@app.post("/experiments/") 
def add_experiment(experiment:Experiment) -> Dict[(int, Experiment)]:
    validate_experiment(experiment)
    global exp_idx
    EXPERIMENTS[exp_idx] = experiment
    exp_idx += 1
    return EXPERIMENTS
    
@app.delete("/experiments/") 
def remove_experiment(id:int):
    try:
        del EXPERIMENTS[id]
    except:
        Error(name="NonexistantExperiment",
        desc = f"There is no experiment with id {id}")

# TEST OPERATIONS
@app.get("/tests/")
def get_test(id:int = None) -> Dict[(int, Test)]:
    if id is None:
        return TESTS
    else:
        try:
            return TESTS[id]
        except:
            Error(name="NonexistantTest",
            desc = f"There is no test with id {id}")

@app.post("/tests/") 
def add_test(test:Test) -> Dict[(int, Test)]:
    validate_test(test)
    global test_idx
    TESTS[test_idx] = test
    test_idx += 1
    return TESTS
    
@app.delete("/tests/") 
def remove_test(id:int):
    try:
        del TESTS[id]
    except:
        Error(name="NonexistantTest",
        desc = f"There is no test with id {id}")

        
if __name__ == "__main__":
    command = f"uvicorn main:app --reload".split()
    run(command, cwd = current_dir)