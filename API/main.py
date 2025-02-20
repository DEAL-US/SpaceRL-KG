# Local imports
import sys, os, atexit, ast, traceback, atexit

import multiprocessing as mp
from multiprocessing import Process, Manager, Pool
from subprocess import run
from pathlib import Path

# from apiutils import DATASETS, ALLOWED_EMBEDDINGS
# from apiutils import permanent_config, changeable_config, convert_var_to_config_type
# from apiutils import Experiment, Test, Error, Triple, EmbGen, infodicttype
# from apiutils import validate_test, validate_experiment, add_dataset, remove_dataset, get_agents, get_info_from
# from apiutils import add_embedding, add_cache, add_experiment, add_test, run_experiments, run_tests
from apiutils import *

from threaded_elements_handler import start_server, start_client


# FastAPI imports
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import HTTPException

from typing import Union, List, Dict

app = FastAPI(title="SpaceRL-KG", version="1.0.0", contact={"deault": "mail@xyz.com"})

# manages client connetions
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

conn = ConnectionManager()

def exit_handler():
    try:
        print("SERVER IS CLOSING")
        send_msg_to_server(conn.client_socket, "quit")
    except Exception as e :
        print(f"unable to close server before exiting, related exception was: {e}")   
   
atexit.register(exit_handler)

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
try /redoc to check all the things I can do or /docs, to try it out!
we are OpenAPI compliant, check /openapi.json.
"""
    return text




# CONFIG OPERATIONS
@app.get("/config/")
def get_config() -> dict:
    return changeable_config

@app.put("/config/")
def set_config(param:str, value) -> dict:
    if param not in changeable_config.keys():        
        Error("ParameterDoesNotExist",
        f"{param} is not a config param.")
    
    value = convert_var_to_config_type(param, value)

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


# DATASET CACHE GENERATION.
@app.get("/cache/")
def get_caches():
    return get_all_caches()

@app.post("/cache/", 
description = "Generates a node distance cache for the selected datasets with the desired depth.\nWARNING: this may hoard a LOT of ram and WILL crash your system for large datasets and depth, proceed with caution.") 
def generate_cache(cache: CacheGen):
    ch = cache.dict()
    aux = []
    for e in ch['datasets']:
        aux.append(e.value)
    ch['datasets'] = aux

    add_cache(conn.client_socket, ch)


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
        except Exception:
            Error(name="NonexistantExperiment",
            desc = f"There is no experiment with id {id}")

@app.post("/experiments/") 
def add_exp(experiment:Experiment):
    validate_experiment(conn.client_socket, experiment)

    exp = experiment.dict()
    exp['dataset'] = exp['dataset'].value
    exp['embedding'] = exp['embedding'].value
    return add_experiment(conn.client_socket, exp)

@app.post("/experiments/run/", description = "Provide list of experiment ids to run them or an empty list to run all currently in queue.") 
def run_exp(ids:List[int] = None):
    return run_experiments(conn.client_socket, ids)
    
@app.delete("/experiments/") 
def remove_experiment(id:int):
    try:
        exp = get_info_from(infodicttype.EXPERIMENT, conn.client_socket, id)
        return delete_experiment(conn.client_socket, id)
    except:
        Error(name="NonexistantExperiment",
        desc = f"There is no experiment with id {id}")




# TEST OPERATIONS
@app.get("/tests/")
def get_test(id:int = None) -> Union[Dict[(int, Test)], Test]:
    if id is None:
        return get_info_from(infodicttype.TEST, conn.client_socket)
    else:
        try:
            return get_info_from(infodicttype.TEST, conn.client_socket, id)
        except:
            Error(name="NonexistantTest",
            desc = f"There is no test with id {id}")

@app.post("/tests/") 
def add_tst(test:Test):
    t = validate_test(conn.client_socket, test)
    
    return add_test(conn.client_socket, t)

@app.post("/tests/run/", description = "Provide list of test ids to run them or an empty list to run all currently in queue.") 
def run_tst(ids:List[int] = []):
    return run_tests(conn.client_socket, ids)
    
@app.delete("/tests/") 
def remove_test(id:int):
    try:
        tst = get_info_from(infodicttype.TEST, conn.client_socket, id)
        return delete_test(conn.client_socket, id)
    except:
        Error(name="NonexistantTest",
        desc = f"There is no test with id {id}")



@app.get("/check/")
def check_processes():
    send_msg_to_server(conn.client_socket, "check")

if __name__ == "__main__":
    import uvicorn

    # command = f"lsof -t -i tcp:8000 | xargs kill -9"
    # # run(command, cwd = current_dir, shell=True)
    # os.system(command)

    uvicorn.run(app, host="127.0.0.1", port=8080)