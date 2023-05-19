# Local imports
import sys, os, atexit
import multiprocessing as mp

from multiprocessing import Process
from subprocess import run

from utils import DATASETS, ALLOWED_EMBEDDINGS
from utils import permanent_config, changeable_config
from utils import Experiment, Test, Error, Triple, EmbGen, infodicttype
from utils import validate_test, validate_experiment, add_dataset, remove_dataset, get_agents, get_info_from
from utils import add_embedding, add_cache, add_experiment, add_test
from utils import convert_var_to_config_type

# FastAPI imports
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import HTTPException

from typing import Union, List, Dict
from pathlib import Path

app = FastAPI()

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()
agents_path = Path(f"{parent_path}/model/data/agents").resolve()
datasets_path = Path(f"{parent_path}/datasets").resolve()


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

    print(f"recieved param value {value}")

    if param not in changeable_config.keys():        
        Error("ParameterDoesNotExist",
        f"{param} is not a config param.")
    
    value = convert_var_to_config_type(param, value)

    print(f"value after conversion {value}")    

    exp_info = get_info_from(infodicttype.EXPERIMENT)
    test_info = get_info_from(infodicttype.TEST)

    print(f"experiment information: {exp_info}, test information: {test_info}")

    if(len(exp_info) != 0 or len(test_info) !=0):
        Error(name = "BusyResourcesError",
        desc = f"There are is an active test/train suite")

    print(f"setting {param} config param to value: {value}")

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

@app.delete("/datasets/")
def delete_dataset(name:str):
    remove_dataset(name)
    return {"message":"dataset and all related content was removed successfully"}

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


# INITIALIZATION:
# connection data.
from multiprocessing import cpu_count, Process, Manager
import multiprocessing as mp

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
SERVER_PORT = 6539
MAXBYTES = 4096

# Client port handlling
# CLIENT_PORTS = dict()  # The ports used by the server
# for i in range(65336, 65435):
#     CLIENT_PORTS[i] = False

# def assign_first_available_port() -> int:
#     for k, v in CLIENT_PORTS.items():
#         if not v:
#             CLIENT_PORTS[k] = True
#             return k
    
#     Error("BusyError", "system is busy, try again later.")
#     return None

# def unassign_port(port:int):
#     CLIENT_PORTS[port] = False

# # response manager.

manager = mp.Manager()

from threaded_elements_handler import start_server, start_client, send_msg_to_server

p = Process(target=start_server, args=(HOST, SERVER_PORT, MAXBYTES))
p.start()

# Testing...
client_socket = start_client(HOST, SERVER_PORT)
send_msg_to_server(client_socket, b"test_msg", MAXBYTES)

send_msg_to_server(client_socket, b"another", MAXBYTES)


if __name__ == "__main__":
    command = f"uvicorn main:app --reload".split()
    run(command, cwd = current_dir)