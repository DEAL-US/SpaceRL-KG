# Local imports
import sys, os, time, collections, json, re, shutil, ast

from enum import Enum
from pathlib import Path
from typing import Union, List, Dict
from pydantic import BaseModel
import GPUtil as gputil

from multiprocessing import cpu_count

from fastapi import HTTPException

# from threaded_elements_handler import start_server, start_client, send_msg_to_server

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()
agents_path = Path(f"{parent_path}/model/data/agents").resolve()
tests_path = Path(f"{parent_path}/model/data/results").resolve()
caches_path = Path(f"{parent_path}/model/data/caches").resolve()
datasets_path = Path(f"{parent_path}/datasets").resolve()

# Helper function required before definition of pydantic models.
# It generates the dataset Enum which is used a return type.
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
    # Exceptionally this error class automatically raises an HTTP exception
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

class CacheGen(BaseModel):
    datasets: List[DATASETS]
    depth: int = 3



# Main Functions
# Validation
def validate_experiment(socket, exp: Experiment):
    reasons = []
    agents = get_agents()
    
    if len(exp.name) > 200 and len(exp.name) < 10:
        reasons.append("Experiment name must be between 10-200 characters\n")

    experiment_queue = get_info_from(infodicttype.EXPERIMENT, socket)
    f = list(filter(lambda l_exp: True if l_exp.name == exp.name else False, experiment_queue.values()))

    if exp.name in agents or len(f) != 0:
        reasons.append("Agent already exists, try a different name.\n")

    # dataset value can't be outside of enum scope, no need for extra validation
    
    if exp.single_relation == True:
        if exp.relation_to_train is None:
            reasons.append("Provide a relation name if you wish to train for it.\n")
        
        elif not check_for_relation_in_dataset(exp.dataset, exp.relation_to_train):
            reasons.append("Provided relation is not in dataset.\n")
    
    if exp.laps < 1 or exp.laps > 10_000:
        reasons.append("laps must be from 1-10.000\n")

    if len(reasons) !=0: Error(name = "ExperimentError", desc = "".join(reasons))
    
def validate_test(socket, tst: Test):
    reasons = []
    tests = get_tests()
    
    if len(tst.name) > 200 and len(tst.name) < 10:
        reasons.append("Test name must be between 10-200 characters\n")

    test_queue = send_msg_to_server(socket, f"get;{infodicttype.TEST}")
    f = list(filter(lambda l_tst: True if l_tst.name == tst.name else False, test_queue.values()))

    if tst.name in tests or len(f) != 0:
        reasons.append("Test with that name already exists, try again.\n")

    # dataset value can't be outside of enum scope, no need for extra validation

    if tst.episodes < 1 or tst.episodes > 10_000:
        reasons.append("episodes must be from 1-10.000\n")
    
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
    if len(reasons) !=0: Error(name = "TestError", desc = "".join(reasons))
    
    tst.embedding = embedding
    tst.dataset = dataset
    tst.single_relation = single_relation
    tst.relation_to_train = relation_to_train

    return tst


# CRUD

# datasets

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

def remove_dataset(dataset_name:str) -> bool:
    # TODO: Delete Chaches, embedding files, agents and tests related to the DATASET being edited or deleted.

    p = Path(f"{datasets_path}/{dataset_name}")
    try:
        shutil.rmtree(p, ignore_errors=False, onerror=None)
    except Exception as e:
        print(e.args)
        Error(name="DatasetRemovalError",
        desc="There was a problem deleting that dataset...")
    
    return True


# embeddings are computed directly if the system is not busy.
def add_embedding(socket, embedding:EmbGen):
    return send_msg_to_server(socket, f"post;embedding;{embedding}")

# caches are run directly if ths sytem is not busy.
def add_cache(socket, cache:CacheGen):
    return send_msg_to_server(socket, f"post;cache;{cache}")

def get_all_caches():
    cache_lst = os.listdir(caches_path)
    cache_lst.remove('.gitkeep')
    cache_lst = list(map(lambda x: x.replace(".pkl", ""), cache_lst))
    return cache_lst

# Experiment and Tests are added to a queue and triggered 
# to run if the system is not busy.
def add_experiment(socket, exp:Experiment):
    return send_msg_to_server(socket, f"post;{infodicttype.EXPERIMENT.value};{exp}")

def add_test(socket, test:Test):
    return send_msg_to_server(socket, f"post;{infodicttype.TEST.value};{test}")


def delete_experiment(socket, id):
    return send_msg_to_server(socket, f"delete;{infodicttype.EXPERIMENT.value};{id}")

def delete_test(socket, id):
    return send_msg_to_server(socket, f"delete;{infodicttype.TEST.value};{id}")


def run_experiments(socket, ids):
    return send_msg_to_server(socket, f"post;{infodicttype.EXPERIMENT.value};run;{ids}")

def run_tests(socket, ids):
    return send_msg_to_server(socket, f"post;{infodicttype.TEST.value};run;{ids}")


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

def convert_var_to_config_type(param:str, value:str):
    if param in ["seed", "available_cores", "path_length"]:
        try:
            value = int(value)
        except:
            Error("InvalidValueError", f"The value '{value}' is not applicable to param '{param}'")

    elif param in ["guided_reward", "regenerate_embeddings",
    "normalize_embeddings", "use_LSTM", "random_seed"]:
        if value.strip().lower() == "false":
            value = False
        elif value.strip().lower() == "true":
            value = True
        else:
            Error("InvalidValueError", f"The value '{value}' is not applicable to param '{param}'")

    elif param in ["alpha", "gamma", "learning_rate"]:
        try:
            value = float(value)
        except:
            Error("InvalidValueError", f"The value '{value}' is not applicable to param '{param}'")

    elif param in ["guided_to_compute", "regularizers"]:
        rep = {"[":"", "]":"", "\"":"", "\'":""}
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        value = pattern.sub(lambda m: rep[re.escape(m.group(0))], value)
        value = value.split(",")
        value = list(map(lambda x: x.strip(), value))
        value = list(set(value))
    
    validate_config_value(param, value)

    return value

def validate_config_value(param:str, value):
    validation_dict = {
    "path_length": [3, 10],
    "available_cores": [1, cpu_count()], 

    "guided_to_compute":["distance", "terminal", "embedding"],
    "regularizers": ["kernel", "bias", "activity"],

    "alpha": [0.8, 0.99],
    "gamma": [0.90, 0.99],
    "learning_rate": [1e-3, 1e-5],

    "activation": ['relu', 'prelu', 'leaky_relu', 'elu', 'tanh'], #opt
    "algorithm": ['BASE', 'PPO'], #opt
    "reward_type": ['retropropagation', 'simple'], #opt
    "action_picking_policy": ["probability", "max"], #opt
    "reward_computation": ["max_percent", "one_hot_max", "straight"], #opt

    }

    if param in ["available_cores", "path_length"]:
        l = validation_dict[param]
        if(value < l[0] or value > l[1]):
            Error("WrongValueError", f"{value} must be in range {l}, was {value} for param: {param}.")

    elif param in ["alpha", "gamma", "learning_rate"]:
        l = validation_dict[param]
        if(value < l[0] or value > l[1]):
            Error("WrongValueError", f"{value} must be in range {l}, was {value} for param: {param}.")

    elif param in ["guided_to_compute", "regularizers"]:
        l = validation_dict[param]
        for v in value:
            if v not in l:
                Error("WrongValueError", f"value(s) must be one of the following: {l}, was {value} instead.")

    else:
        try:
            l = validation_dict[param]
            if v not in l:
                Error("WrongValueError", f"value must be one of the following: {l}, was {value} instead.")
        except:
            if param not in changeable_config:
                Error("WrongValueError", f"param was not found.")

def dict_to_exp(exp_dict: dict) -> Experiment:

    # class Test(BaseModel):
    #     name: str
    #     agent_name :str
    #     episodes: int
    #     embedding : Union[ALLOWED_EMBEDDINGS, None]
    #     dataset: Union[DATASETS, None]
    #     single_relation: Union[bool, None]
    #     relation_to_train : Union[str, None]

    dtst = [e for e in DATASETS if e.value == exp_dict['dataset']][0]
    emb = [e for e in ALLOWED_EMBEDDINGS if e.value == exp_dict['embedding']][0]

    res = Experiment(name=exp_dict['name'], dataset=dtst, single_relation=exp_dict['single_relation'],
    embedding=emb, laps=exp_dict['laps'], relation_to_train=exp_dict['relation_to_train'])

    return res

def dict_to_tst(tst_dict: dict) -> Test:
    dtst = [e for e in DATASETS if e.value == exp_dict['dataset']][0]
    emb = [e for e in ALLOWED_EMBEDDINGS if e.value == exp_dict['embedding']][0]

    res = Experiment(name=exp_dict['name'], dataset=dtst, single_relation=exp_dict['single_relation'],
    embedding=emb, laps=exp_dict['laps'], relation_to_train=exp_dict['relation_to_train'])

    return res

class infodicttype(Enum):
    CACHE = "cache"
    EXPERIMENT = "experiment"
    TEST = "test"

def get_info_from(opt:infodicttype, socket, id:int = None):
    if(opt == infodicttype.CACHE):
        msg = f"get;{infodicttype.CACHE.value}"

    elif(opt == infodicttype.EXPERIMENT):
        msg = f"get;{infodicttype.EXPERIMENT.value}"

    elif(opt == infodicttype.TEST):
        msg = f"get;{infodicttype.TEST.value}"

    else:
        Error("BadRequestError",f"you asked for the wrong thing bucko!\n{opt}")
    

    if(id is not None):
        msg += f";{id}"

    res = send_msg_to_server(socket, msg)

    if(id is None):
        for e in res.items():
            if opt == infodicttype.EXPERIMENT:
                res[e[0]] = dict_to_exp(e[1])
            elif opt == infodicttype.TEST:
                res[e[0]] = dict_to_tst(e[1])

    else:
        if opt == infodicttype.EXPERIMENT:
            res = dict_to_exp(res)
        elif opt == infodicttype.TEST:
            res = dict_to_tst(res)

    return res

def get_config():
    return permanent_config, changeable_config

# Server OPS:

def send_msg_to_server(client_socket, msg, MAXBYTES = 4096):
    msg = bytes(msg, 'utf-8')

    # send message to server
    client_socket.sendall(msg)

    # await server answer
    data = client_socket.recv(MAXBYTES)
    data = data.decode("utf-8")

    # process server answer
    print(f"GOT MESSAGE BACK FROM SERVER {data}")
    data = response_handler(data)

    if(data == 'multi'):
        print("DATA IS MULTIPART, NEED TO RECIEVE MORE!")
        done = False
        while(not done):
            data = s.recv(MAXBYTES)
            data = data.decode("utf-8")
            done = data != 'multi'

    return data

def response_handler(r:str):
    msg = r.split(';', maxsplit=3)
    var = msg[0]

    if var == 'error':
        Error(msg[1], msg[2])
    
    if var == 'success':
        return {"Success" : msg[1]}
    
    if var == 'multi':
        multi_idx, part_idx, msg_part = msg[1], *msg[2].split(';', maxsplit=2)

        if(part_idx == 'L'):
            od = collections.OrderedDict(sorted(big_responses.items()))
            reslist = list(od.values())
            reslist.append(msg_part)
            return response_handler("".join(reslist))

        elif multi_idx not in big_responses:
            big_responses[multi_idx] = dict()
            big_responses[multi_idx][part_idx] = msg_part
        
        return 'multi'

    if var == 'dict': # recieved whole list of objects.
        if len(msg) == 3: # returning dict with ID associated.
            return ast.literal_eval(msg[2])

        elif len(msg) == 2: # returning all entries.
            aux = ast.literal_eval(msg[1])
            print(aux)
            # for k, v in aux.items():
            #     aux[k] = ast.literal_eval(v)
            return aux