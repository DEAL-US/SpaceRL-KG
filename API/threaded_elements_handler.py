import sys, os, time, socket, json, ast

from typing import List

from pathlib import Path
from random import randint
from enum import Enum

from fastapi import HTTPException
import multiprocessing as mp

from apiutils import get_config

class Error():
    def __init__(self, name:str, desc:str):
        raise HTTPException(status_code=400, 
        detail=f"{name} - {desc}")

class infodicttype(Enum):
    CACHE = "cache"
    EXPERIMENT = "experiment"
    TEST = "test"

class Process(mp.Process):
    def __init__(self, logs: bool, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self.logs = logs

    def run(self):
        self.initialize_logging()
        mp.Process.run(self)
        
    def initialize_logging(self):
        if self.logs:
            sys.stdout = open(f"./logs/p_{self.name}.out", "a", buffering=-1)
            sys.stderr = open(f"./logs/p_{self.name}.err", "a", buffering=-1)
        else:
            sys.stdout = open(os.devnull, "w", buffering=-1)
            sys.stderr = open(os.devnull, "w", buffering=-1)

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()

# RELATIVE IMPORTS.
genpath = Path(f"{parent_path}/model/data/generator").resolve()
modelpath = Path(f"{parent_path}/model").resolve()
miscpath = Path(f"{parent_path}/misc/").resolve()

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, str(genpath))
sys.path.insert(0, str(modelpath))
sys.path.insert(0, str(miscpath))


import generate_trans_embeddings as embgen
import tester, trainer

from generate_cache_for_dataset import main as cache_generator
from config import Experiment, Test

experiment_queue, test_queue = dict(), dict()
exp_idx, test_idx = 0,0

# CHANGE TO REDIRECT OUTPUT OF CHILD PROCESSES TO LOGS FOLDER.
use_logs = True

# Server message handler
def server_message_handler(data:str, MAXBYTES) -> str:
    # THIS SHOULD ANSWER VERY FAST AND DO A PROCESS FOR SLOW OPS.
    global experiment_queue, test_queue, exp_idx, test_idx

    res = ""

    if(data == "quit"):
        return None

    if(data == "check"):
        print(f"live children processes: {mp.active_children()}")
        return f"success;{mp.active_children()}"

    msg = data.split(';', maxsplit=3)
    try:
        petition, variant = msg[0], msg[1]
    except:
        return f"error;BadRequest;recieved message was malformed, please check the docs."
    
    # print(f"PROCESSING MESSAGE:\n {data} \n")
    
    if(petition == 'get'):
        if(len(msg) == 2): # res = the complete dict 
            if(variant == infodicttype.CACHE.value):
                res = f"dict;{cache_queue}"
            
            elif(variant == infodicttype.EXPERIMENT.value):
                res = f"dict;{experiment_queue}"
            
            elif(variant == infodicttype.TEST.value):
                res = f"dict;{test_queue}"
            
            else:
                return f"error;MalformedGetRequest;get request does not match expected input, please check spelling..."
            
        elif(len(msg) == 3): #res = requested id.
            if(variant == infodicttype.CACHE.value):
                try:
                    res = f"dict;{msg[2]};{cache_queue[int(msg[2])]}"
                except:
                    res = f"error;IDNotFound;id \"{msg[2]}\" for cache does not exist."
            
            elif(variant == infodicttype.EXPERIMENT.value):
                try:
                    res = f"dict;{msg[2]};{experiment_queue[int(msg[2])]}"
                except Exception as e:
                    res = f"error;IDNotFound;id \"{msg[2]}\" for experiment does not exist."
            
            elif(variant == infodicttype.TEST.value):
                try:
                    res = f"dict;{msg[2]};{test_queue[int(msg[2])]}"
                except:
                    res = f"error;IDNotFound;id \"{msg[2]}\" for test does not exist."
            
            else:
                return f"error;MalformedGetRequest;get request does not match expected input, please check spelling..."

        else:
            return f"error;MalformedGetRequest;couldn't parse petition."

    elif(petition == 'post'):

        if(len(msg) == 3): 
            if(variant == infodicttype.CACHE.value):
                if not enough_free_resources():
                    return f"error;SystemBusy;not enough resources free to run the requested process..."

                data = ast.literal_eval(msg[2])

                proc = Process(target=cache_generator, 
                args=[data['datasets'], data['depth']], 
                name=f"CacheGenerator", logs=use_logs)
                proc.start()

                res = f"success;cache is being generated, please be patient"
            

            elif(variant == 'embedding'):
                if not enough_free_resources():
                    return f"error;SystemBusy;not enough resources free to run the requested process..."

                data = ast.literal_eval(msg[2])

                dataset = data['dataset']
                models = data['models']
                gpu = data['use_gpu']
                regen = data['regenerate_existing']
                normalize = data['normalize']
                inverse = data['add_inverse_path']
                mode = data['fast_mode']

                proc = Process(target=embgen.generate_embedding, 
                args=[dataset, models, gpu, regen, normalize, inverse, mode, 1, True], 
                name=f"EmbeddingGenerator", logs=use_logs)
                proc.start()

                res = f"success; Embedding Calculation Launched"


            elif(variant == infodicttype.EXPERIMENT.value):
                experiment = msg[2]
                experiment_queue[exp_idx] = ast.literal_eval(experiment)
                exp_idx += 1

                res = f"success;experiment successfully added to queue."
            

            elif(variant == infodicttype.TEST.value):
                test = msg[2]
                test_queue[test_idx] = ast.literal_eval(test)
                test_idx += 1

                res = f"success;test successfully added to queue."

            else:
                return f"error;MalformedPostRequest;request does not match expected input, please check spelling..."
       
        elif(len(msg) == 4 and msg[2] == "run"):
            if not enough_free_resources():
                    return f"error;SystemBusy;not enough resources free to run the requested process..."

            if(variant == infodicttype.EXPERIMENT.value):
                exp_list = ast.literal_eval(msg[3])
                
                # print(f"recieved experiments to run: {exp_list}")

                if type(exp_list) != list:
                    return f"error;UnparseableIdList; you sent something that wasnt a list of ids."
                
                to_run = []
                if len(exp_list) == 0:
                    to_run = list(experiment_queue.items())
                
                else:
                    to_run = [(p[0], p[1]) for p in experiment_queue.items() if p[0] in exp_list]

                if len(to_run) == 0:
                    return f"error;NonExistentID;none of the provided ids match to any experiment.."
                
                per_cnfg, mut_config = get_config()

                # print(f"configs are: \n{per_cnfg}\n\n{mut_config}\n")

                cnfg = {**per_cnfg, **mut_config}

                # print(f"the complete config is: \n{cnfg}")

                exp_list_to_run = []
                for e in to_run:
                    del experiment_queue[e[0]]
                    ex = e[1]
                    exp_list_to_run.append(Experiment(experiment_name=ex['name'],
                    dataset_name=ex['dataset'], embeddings = [ex['embedding']],
                    laps=ex['laps'], single_relation = ex['single_relation'],
                    relation = ex['relation_to_train'] if ex['single_relation'] else ''))

                api_conn = dict()
                api_conn["config"] = cnfg
                api_conn["experiments"] = exp_list_to_run

                p = Process(target=trainer.main, args=[False, api_conn, None],
                name=f"ExperimentRunner", logs=use_logs)
                p.start()

                return f"success;Experiments Launched."
            
            elif(variant == infodicttype.TEST.value):
                tst_list = ast.literal_eval(msg[3])

                if type(tst_list) != list:
                    return f"error;UnparseableIdList; you sent something that wasnt a list of ids."
                
                to_run = []
                if len(tst_list) == 0:
                    to_run = list(test_queue.items())
                
                else:
                    to_run = [(p[0], p[1]) for p in test_queue.items() if p[0] in tst_list]

                print(f" tests to run: {to_run}")

                if len(to_run) == 0:
                    return f"error;NonExistentID;none of the provided ids match to any test.."
                
                per_cnfg, mut_config = get_config()
                cnfg = {**per_cnfg, **mut_config}

                tst_list_to_run = []
                for e in to_run:
                    del test_queue[e[0]]
                    t = e[1]
                    tst_list_to_run.append(
                    Test(
                        test_name=t['name'],
                        agent_name=t['agent_name'],
                        embeddings = [t['embedding']],
                        episodes=t['episodes']
                        )
                    )

                api_conn = dict()
                api_conn["config"] = cnfg
                api_conn["tests"] = tst_list_to_run

                p = Process(target=tester.main, args=[False, api_conn, None],
                name=f"TestRunner", logs=use_logs)
                p.start()

                return f"success;Tests Launched."

    elif(petition == 'delete'):
        if(variant == infodicttype.EXPERIMENT.value):
            del experiment_queue[int(msg[2])]
            res = f"success;removed expetiment successfully"

        elif(variant == infodicttype.TEST.value):
            del test_queue[int(msg[2])]
            res = f"success;removed test successfully"
        
        else:
            return f"error;MalformedDeleteRequest;couldn't parse petition."

    else:
        return f"error;MalformedRequest;petition type is unknown."

    if (res == ""):
        return f"error;UnkownError;Something went wrong... Please try again. {msg}"

    if (len(res) > MAXBYTES): #msg is multipart.
        multipart_idx = randint(0, sys.maxsize) # we believe in a 0 collision world here.
        reslist = []
        done = False
        head = f"multi;{multipart_idx};0;"
        a, b, c = 0, MAXBYTES-len(head), 0

        while not done:
            c += 1

            if(b > len(res)):
                done = True
                head = f"multi;{multipart_idx};L;"

            reslist.append(f"{head}{res[a:b]}")

            head = f"multi;{multipart_idx};{c};"
            overhead = len(head)
            a = b + 1
            b = b + MAXBYTES-overhead

        print("SENDING MESSAGE AS MULTIPART:")
        print("".join[reslist])

        multipart_idx += 1
        return "".join[reslist]
    
    return res

def start_client(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tries = 0
    while(True):
        try:
            tries += 1
            s.connect((host, port))
        except Exception as e:
            if tries == 10:
                return None

            if e.args[0] == 106: #server is connected
                return s
            
            time.sleep(1)

def start_server(host, port, MAXBYTES):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except:
            print("Port is busy... Maybe there is an instance running?")
            quit()

        s.listen()
        print("Server is listening")

        conn, addr = s.accept()
        print(f"Connected by {addr}")

        with conn:
            while True:
                data = conn.recv(MAXBYTES)
                print(f"data has been recieved by server: {data}")

                if not data:
                    print(f"data recieved was wrong {data}")
                else:
                    response = server_message_handler(data.decode("utf-8"), MAXBYTES)                
                    if (response is None):
                        s.close()
                        quit()
                    else:
                        conn.sendall(bytes(response, 'utf-8'))


def enough_free_resources():
    active = mp.active_children()
    return len(active) < 2