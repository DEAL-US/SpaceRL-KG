from pathlib import Path
import os
from datetime import datetime
from regex import P
import traceback

import tensorflow as tf
import pandas as pd

import random
import traceback

from tqdm import tqdm
from keras.models import Model
from data.data_manager import DataManager
from environment import KGEnv
from agent import Agent
from keras.models import load_model
import numpy as np

from config import get_config

def run_prep():
    config, TESTS = get_config(train=False)
    config ["use_episodes"] = True
    embs = ["TransE_l2", "DistMult", "ComplEx", "TransR"]

    p = Path(__file__).parent
    respath = f"{str(p)}/data/results/{datetime.now().date().isoformat()}"
    agents_path = f"{str(p)}/data/agents/testing"
    datasets_dir = os.listdir(f"{str(p.parent)}/datasets")
    num_datasets = len(datasets_dir)-1

    if num_datasets < len(TESTS):
        print("you have more tests than datasets, try combining some test for the dataset.")
        quit()

    df_index = [
        np.array([
        "hits@1","hits@1","hits@1","hits@1",
        "hits@3","hits@3","hits@3","hits@3",
        "hits@5","hits@5","hits@5","hits@5",
        "hits@10","hits@10","hits@10","hits@10",
        "MRR","MRR","MRR","MRR"]),
        np.array([*embs,*embs,*embs,*embs, *embs])
    ]

    metrics_df = pd.DataFrame(
        columns = [t.dataset for t in TESTS],
        index = df_index
    )

    try:
        os.mkdir(respath)
    except:
        print("folder exists already, replacing files.")

    return df_index, metrics_df, config, embs, TESTS, respath, agents_path

df_index, metrics_df, config, embs, TESTS, respath, agents_path = run_prep()


class Tester(object):
    '''
    Test the model and calculate MRR & Hits@N metrics.
    '''
    def __init__(self, env_config, agent_models):
        for key, val in env_config.items(): setattr(self, key, val)

        embs = ["TransE_l2", "DistMult", "ComplEx", "TransR"]
        self.selected_embedding_name = embs[self.embedding_index]

        if (self.random_seed):
            seed = random.randint(0, (2**32)-1)
        else:
            seed = self.seed

        self.set_gpu_config(self.gpu_acceleration)

        self.dm = DataManager(is_experiment=False)

        self.env = KGEnv(self.dm, self.dataset, 
        [t.single_relation, t.relation_to_train],
        self.embedding_index, seed, 8, self.path_length, 
        False, False, self.gpu_acceleration, True, 0, False)

        self.agent = Agent(self.dm, self.env, 0.99, 1e-4, True,
        "leaky_relu", [], "max_percent", [], self.action_picking_policy,
        self.algorithm, True, 0.9, self.reward_type, True, 
        verbose = self.verbose, debug = self.debug)

        if(len(agent_models) == 1):
            self.agent.policy_network = agent_models[0]
        
        elif(len(agent_models) == 2):
            self.agent.policy_network = agent_models[0]
            self.agent.critic = agent_models[1]

    def run(self):
        MRR = []
        hits_at = {i: 0 for i in (1, 3, 5, 10)}

        try:
            for x in tqdm(range(self.episodes)):
                MRR.append(0)
                for i in range(1, 11):
                    #reset and build path
                    self.env.reset()
                    for _ in range(self.path_length):
                        action = self.agent.select_action_runtime()
                        self.env.step(action)

                    # if tail entity in paths compute hits at.
                    if(self.path_contains_entity()):
                        for n, val in hits_at.items():
                            if i <= n:
                                hits_at[n] = val + 1
                        
                        MRR[x] = i
                        break

            return hits_at, MRR
            
        except Exception as e:
            print(traceback.format_exc())
            return False
    
    def path_contains_entity(self):
        visited = set()
        path = self.env.path_history 
        entity = self.env.target_triple[2]
        for s in path:
            visited.add(s[2])
        return (entity in visited)
    
    def set_gpu_config(self, use_gpu):
        if not use_gpu:
            try:
                tf.config.set_visible_devices([], 'GPU')
                visible_devices = tf.config.get_visible_devices()
                for device in visible_devices:
                    assert device.device_type != 'GPU'
            except:
                pass

def compute_metrics(mrr, hits):
    ep = config["episodes"]
    hits = [i[1]/ep for i in hits.items()]
    mrr = [1/i if(i != 0) else 0 for i in mrr]
    mrr = sum(mrr)/len(mrr)
    return hits, mrr

def get_agents(test):
    constant_path = f"{agents_path}/{test.dataset}-"
    agents = {0:None, 1:None, 2:None, 3:None}
    for e in t.embeddings:
        ppo = constant_path + e
        base = ppo + ".h5"

        ppo_exist = os.path.isdir(ppo)
        base_exist = os.path.isfile(base)

        if(ppo_exist and base_exist):
            print(f"2 agents found for embedding {e} and dataset {test.dataset}, remove one.")
        else:
            if(ppo_exist):
                actor = load_model(f"{ppo}/actor.h5")
                critic = load_model(f"{ppo}/critic.h5")
                agents[emb_mapping[e]] = [actor, critic]

            if(base_exist):
                policy_network = load_model(base)
                agents[emb_mapping[e]] = [policy_network]

    return agents

################## START ####################
emb_mapping = {"TransE_l2":0, "DistMult":1, "ComplEx":2, "TransR":3}
for t in TESTS:
    agents = get_agents(t)

    for emb_i in t.embedding_inds:
        try:
            d = t.dataset
            config["dataset"] = d
            config["embedding_index"] = emb_i
            config["episodes"] = t.episodes
            sent = agents[emb_i]
            m = Tester(config, sent)
            res = m.run()

            if(res == False):
                print("something went wrong")
                m.run_debug()
                quit()
            
            hits_raw, mrr_raw = res
            hits, mrr = compute_metrics(mrr_raw, hits_raw)

            metrics_df.at[("hits@1",embs[emb_i]), d] = hits[0]
            metrics_df.at[("hits@3",embs[emb_i]), d] = hits[1]
            metrics_df.at[("hits@5",embs[emb_i]), d] = hits[2]
            metrics_df.at[("hits@10",embs[emb_i]), d] = hits[3]
            metrics_df.at[("MRR",embs[emb_i]), d] = mrr
            
        except:
            print("error")
            traceback.print_exc()
            quit()
            continue

print(metrics_df)
metrics_df.to_csv(f"{respath}/metrics.csv")