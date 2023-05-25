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

from threaded_elements_handler import start_server, start_client, send_msg_to_server
from multiprocessing import Process, Manager, Pool

import atexit

app = FastAPI()

class ConnectionManager():
    def __init__(self):
        # Server constants.
        HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
        SERVER_PORT = 6539
        MAXBYTES = 4096

        p = Process(target=start_server, args=(HOST, SERVER_PORT, MAXBYTES))
        p.start()

        self.client_socket = start_client(HOST, SERVER_PORT)

        if(self.client_socket is None):
            quit("exiting...")

def exit_handler():
    try:
        print("SERVER IS CLOSING")
        close_server()
    except Exception as e :
        print(f"unable to close server before exiting, related exception was: {e}")   
   
atexit.register(exit_handler)
conn = ConnectionManager()

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

    # print(f"recieved param value {value}")

    if param not in changeable_config.keys():        
        Error("ParameterDoesNotExist",
        f"{param} is not a config param.")
    
    value = convert_var_to_config_type(param, value)

    # print(f"value after conversion {value}")    

    exp_info = get_info_from(infodicttype.EXPERIMENT, conn.client_socket)
    test_info = get_info_from(infodicttype.TEST, conn.client_socket)

    # print(f"experiment information: {exp_info}, test information: {test_info}")

    if(len(exp_info) != 0 or len(test_info) !=0):
        Error(name = "BusyResourcesError",
        desc = f"There are is an active test/train suite")

    # print(f"setting {param} config param to value: {value}")

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
    emb = embedding.dict()
    emb['dataset'] = emb['dataset'].value
    aux = []
    for e in emb['models']:
        aux.append(e.value)

    emb['models'] = aux

    add_embedding(conn.client_socket, emb)

# AGENTS
@app.get("/agents/")
def agents():
    return get_agents()


# EXPERIMENTS OPERATIONS
@app.get("/experiments/")
def get_experiment(id:int = None) -> Union[Dict[(int, Experiment)], Experiment]:
    if id is None:
        return get_info_from(infodicttype.EXPERIMENT, conn.client_socket)
    else:
        try:
            return get_info_from(infodicttype.EXPERIMENT, conn.client_socket, id)
        except:
            Error(name="NonexistantExperiment",
            desc = f"There is no experiment with id {id}")

@app.post("/experiments/") 
def add_exp(experiment:Experiment) -> Dict[(int, Experiment)]:
    validate_experiment(conn.client_socket, experiment)

    exp = experiment.dict()
    exp['dataset'] = exp['dataset'].value
    exp['embedding'] = exp['embedding'].value
    add_experiment(conn.client_socket, exp)
    
@app.delete("/experiments/") 
def remove_experiment(id:int):
    try:
        send_msg_to_server(client_socket,f"delete;experiment;{id}")
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
        send_msg_to_server(conn.client_socket,f"delete;test;{id}")

        # del test_queue[id]
        return {"message":"test was removed successfully."}
    except:
        Error(name="NonexistantTest",
        desc = f"There is no test with id {id}")

@app.get("/close/")
def close_server():
    send_msg_to_server(conn.client_socket, "quit")

@app.get("/check/")
def close_server():
    send_msg_to_server(conn.client_socket, "check")


if __name__ == "__main__":
    # command = f"uvicorn main:app".split()
    # run(command, cwd = current_dir)

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)