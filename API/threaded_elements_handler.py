import sys, os, time
import socket

from pathlib import Path

embgen_queue, cache_queue, experiment_queue, test_queue = dict(), dict(), dict(), dict()
embgen_idx, cache_idx, exp_idx, test_idx = 0,0,0,0

multipart_idx = 0

# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()

# RELATIVE IMPORTS.
genpath = Path(f"{parent_path}/model/data/generator").resolve()

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, str(genpath))

import generate_trans_embeddings as embgen


# Simple server code.
def message_handler(data:str) -> str:
    global embgen_queue, cache_queue, experiment_queue, test_queue
    global embgen_idx, cache_idx, exp_idx, test_idx 
    global multipart_idx 

    if(data == "quit"):
        quit()

    msg = data.split(';', maxsplit=3)
    petition, variant = msg[0], msg[1]

    embgen_queue, cache_queue, experiment_queue, test_queue
    if(petition == 'get'):
        if(len(msg) == 2): # res = the complete dict 
            if(variant == 'caches'):
                res = f"cache;{cache_queue}"
            
            elif(variant == 'experiments'):
                res = f"experiment;{experiment_queue}"
            
            elif(variant == 'tests'):
                res = f"test;{test_queue}"
            
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
                
    if(petition == 'post'):
        if(variant == 'caches'):
            cache = msg[2]
            print(cache)
            cache_queue[cache_idx] = cache
            cache_idx += 1

            return f"success;cache successfully added to queue."
        
        if(variant == 'embeddings'):
            embedding = msg[2]
            print(embedding)
            embgen_queue[embgen_idx] = embedding
            embgen_idx += 1

            return f"success;embedding successfully added to queue."

        elif(variant == 'experiments'):
            experiment = msg[2]
            print(experiment)
            experiment_queue[exp_idx] = experiment
            exp_idx += 1

            return f"success;experiment successfully added to queue."
        
        elif(variant == 'tests'):
            test = msg[2]
            print(test)
            test_queue[test_idx] = test
            test_idx += 1

            return f"success;test successfully added to queue."

    if (len(res) > MAXBYTES): #msg is multipart.
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

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
MAXBYTES = 1024

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(MAXBYTES)
                if not data:
                    break

                response = message_handler(data.decode("utf-8"))                
                conn.sendall(bytes(response, 'utf-8'))