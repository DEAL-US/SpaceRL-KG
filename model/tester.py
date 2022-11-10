from pathlib import Path
import os
import traceback

import tensorflow as tf
import pandas as pd

import random
import traceback

from tqdm import tqdm
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

    p = Path(__file__).parent.absolute()
    parent = p.parent.resolve()
    p = p.resolve()

    agents_paths , respaths = [], []

    for t in TESTS:
        respath = Path(f"{p}\\data\\results\\{t.name}").resolve()
        os.mkdir(respath)
        respaths.append(respath)

        a_path = Path(f"{p}/data/agents/{t.agent_name}")
        agents_paths.append(a_path)

    datasets_dir = os.listdir(f"{parent}/datasets")
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
        columns = config["dataset"],
        index = df_index
    )

    return df_index, metrics_df, config, embs, TESTS, agents_paths, respaths

class Tester(object):
    '''
    Test the model and calculate MRR & Hits@N metrics.
    '''
    def __init__(self, respath, env_config, agent_models):
        for key, val in env_config.items(): setattr(self, key, val)

        embs = ["TransE_l2", "DistMult", "ComplEx", "TransR"]
        self.selected_embedding_name = embs[self.embedding_index]

        if (self.random_seed):
            seed = random.randint(0, (2**32)-1)
        else:
            seed = self.seed

        self.set_gpu_config(self.gpu_acceleration)

        self.dm = DataManager(is_experiment=False, experiment_name=self.name, respath=respath)

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
            
            self.generate_MRR_boxplot_and_source(MRR)
            return hits_at, MRR
    
        except Exception as e:
            print(traceback.format_exc())
            return False
    
    def generate_MRR_boxplot_and_source(self, MRR):
        source_filepath = f"{self.dm.test_result_path}/res.txt"
        with open(source_filepath, "w") as f:
            f.write(str(MRR))
        


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

def get_agents(agent_path, dataset, embeddings):
    constant_path = f"{agent_path}/{dataset}-"
    agents = {0:None, 1:None, 2:None, 3:None}
    for e in embeddings:
        ppo = constant_path + e
        base = ppo + ".h5"

        ppo_exist = os.path.isdir(ppo)
        base_exist = os.path.isfile(base)

        if(ppo_exist and base_exist):
            print(f"2 agents found for embedding {e} and dataset {dataset}, remove one.")
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
df_index, metrics_df, config, embs, TESTS, agents_paths, respaths = run_prep()
emb_mapping = {"TransE_l2":0, "DistMult":1, "ComplEx":2, "TransR":3}

for i, t in enumerate(TESTS):
    agents = get_agents(agents_paths[i], t.dataset, t.embeddings)
    print(agents)
    quit()

    for emb_i in t.embedding_inds:
        try:
            d = t.dataset
            config["dataset"] = d
            config["embedding_index"] = emb_i
            config["episodes"] = t.episodes
            config["name"] = t.name
            sent = agents[emb_i]
            m = Tester(respaths[i], config, sent)
            res = m.run()

            if(res == False):
                print("something went wrong")
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

    print(metrics_df)
    metrics_df.to_csv(f"{respaths[i]}/metrics.csv")