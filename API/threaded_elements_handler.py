import sys, os, time
import socket

from pathlib import Path

embgen_queue, cache_queue, experiment_queue, test_queue = dict(), dict(), dict(), dict()
embgen_idx, cache_idx, exp_idx, test_idx = 0,0,0,0


# Folder paths:
current_dir = Path(__file__).parent.resolve()
parent_path = current_dir.parent.resolve()

# RELATIVE IMPORTS.
genpath = Path(f"{parent_path}/model/data/generator").resolve()

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, str(genpath))

import generate_trans_embeddings as embgen

def QueueProcessHandler():
    """
    Handles the queues 
    """
    nxt_eg, nxt_chq, nxt_expq, nxt_tstq = None, None, None, None
    while True:
        if embgen_queue: nxt_eg = next(iter(embgen_queue.items()))
        if cache_queue: nxt_chq = next(iter(cache_queue.items()))
        if experiment_queue: nxt_expq = next(iter(experiment_queue.items()))
        if test_queue: nxt_tstq = next(iter(test_queue.items()))

        print(nxt_eg, nxt_chq, nxt_expq, nxt_tstq)
        print(embgen_queue, cache_queue, experiment_queue, test_queue)

        time.sleep(2)

    
    # p = mp.Process(target=embgen.generate_embedding,
    # args=(dataset.name, models, use_gpu,regenerate_existing, normalize, add_inverse_path, fast_mode))
    # embgen_queue[len(embgen_queue)] = p


# Simple server code.

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

def main():
    print("LAUNCHING SERVER HANDLER")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print("SERVER IS READY TO RECIEVE CONNECTIONS")
        s.listen()
        conn, addr = s.accept()
        
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.sendall(data)