import sys, os, time, socket

from pathlib import Path
from random import randint

from fastapi import HTTPException

class Error():
    def __init__(self, name:str, desc:str):
        raise HTTPException(status_code=400, 
        detail=f"{name} - {desc}")

import multiprocessing as mp

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()

# RELATIVE IMPORTS.
modelpath = Path(f"{parent_path}/model/data/generator").resolve()

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, str(modelpath))

import generate_trans_embeddings as embgen

# Server message handler
def server_message_handler(data:str, MAXBYTES) -> str:
    # THIS SHOULD ANSWER VERY FAST AND DO A PROCESS FOR HEAVY OPS.

    multipart_idx = None

    if(data == "quit"):
        return None

    msg = data.split(';', maxsplit=3)
    try:
        petition, variant = msg[0], msg[1]
    except:
        return f"error;BadRequest;recieved message was malformed, please check the docs."
    
    
    if(petition == 'get'):
        if(len(msg) == 2): # res = the complete dict 
            if(variant == 'caches'):
                res = f"cache;{cache_queue}"
            
            elif(variant == 'experiments'):
                res = f"experiment;{experiment_queue}"
            
            elif(variant == 'tests'):
                res = f"test;{test_queue}"
            
            else:
                return f"error;MalformedGetRequest;get request does not match expected input, please check spelling..."
            
        if(len(msg) == 3): #res = requested id.
            if(variant == 'caches'):
                try:
                    res = f"cache;{msg[2]};{cache_queue[msg[2]]}"
                except:
                    res = f"error;IDNotFound;id \"{msg[2]}\" for cache does not exist."
            
            elif(variant == 'experiments'):
                try:
                    res = f"experiment;{msg[2]};{experiment_queue[msg[2]]}"
                except:
                    res = f"error;IDNotFound;id \"{msg[2]}\" for experiment does not exist."
            
            elif(variant == 'tests'):
                try:
                    res = f"test;{msg[2]};{test_queue[msg[2]]}"
                except:
                    res = f"error;IDNotFound;id \"{msg[2]}\" for test does not exist."
            
            else:
                return f"error;MalformedGetRequest;get request does not match expected input, please check spelling..."

    if(petition == 'post'):
        if(variant == 'cache'):
            cache = msg[2]
            print(cache)
            cache_queue[cache_idx] = cache
            cache_idx += 1

            res = f"success;cache successfully added to queue."
        
        elif(variant == 'embedding'):
            embedding = msg[2]
            print(embedding)
            print(msg)

            quit()
            
            embgen.generate_embedding(embedding)

            res = f"success;embedding successfully added to queue."

        elif(variant == 'experiment'):
            experiment = msg[2]
            print(experiment)
            experiment_queue[exp_idx] = experiment
            exp_idx += 1

            res = f"success;experiment successfully added to queue."
        
        elif(variant == 'tests'):
            test = msg[2]
            print(test)
            test_queue[test_idx] = test
            test_idx += 1

            return f"success;test successfully added to queue."

        else:
            return f"error;MalformedPostRequest;request does not match expected input, please check spelling..."

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


# Client message handling...
def response_handler(r:str):
    msg = r.split(';', maxsplit=3)
    var = msg[0]

    if var == 'error':
        Error(msg[1], msg[2])
    
    if var == 'success':
        return {"message":msg[1]}
    
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

    if len(msg)==2: # recieved whole list of objects.
        resp = msg[1]

        if var == 'cache':
            pass

        if var == 'experiment':
            pass
            
        if var == 'test':
            pass
            
        return json.loads(resp)

    if len(msg)==3:
        idx, resp = msg[1], msg[2]
    
        return json.loads(resp)

def start_client(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tries = 0
    print(f"trying to connect to server...")
    while(True):
        try:
            tries += 1
            s.connect((host, port))
        except Exception as e:
            print(e)            
            if tries == 10:
                print(f"Connection to server failed. Timeout.")
                return None

            if e.args[0] == 106: #server is connected
                return s
                print("Connection to server stablished.")
            
            time.sleep(1)

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

    print(f"data being returned is of type: {type(data)}")
    return data

def start_server(host, port, MAXBYTES):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except:
            print("Port is busy... Maybe there is an instance running?")
            quit()

        print("server has started")
        s.listen()
        conn, addr = s.accept()
        print(f"Connected by {addr}")

        with conn:
            while True:
                print("server is ready to recieve data.")

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