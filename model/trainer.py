import os, time, threading

from GPUtil import GPUtil
# select avaliable GPU
# gpu1, gpu2 = GPUtil.getGPUs()
# available_gpu = gpu1 if gpu1.memoryFree >= gpu2.memoryFree else gpu2
# os.environ["CUDA_VISIBLE_DEVICES"]=str(available_gpu.id)

import tensorflow as tf
import numpy as np
import random

from tqdm import tqdm
from agent import Agent
from environment import KGEnv
from data.data_manager import DataManager
from keras import backend as K
from config import get_config
from utils import Utils

# Global GUI values
tr_total_iterations, tr_current_iteration = 0, 0
tr_total_iter_steps, tr_current_iter_steps = 0, 0

class Trainer(object):
    '''
    Trainer method is the entry point to the training agent.
    '''
    def __init__(self, env_config, from_gui = False, gui_connector = None):
        for key, val in env_config.items(): setattr(self, key, val)
        if (self.random_seed):
            seed = random.randint(0, (2**32)-1)
        else:
            seed = self.seed

        # GUI variables to update.
        self.update_gui_vars(progtext="Initializing Trainer...", tot_steps=1, curr_step=0)
        self.is_ready = False

        is_distance = "distance" in self.guided_to_compute
        self.set_gpu_config(self.gpu_acceleration)
        self.dm = DataManager(is_experiment = True, experiment_name=self.name)

        self.utils = Utils(self.verbose, self.log_results, self.dm)

        self.env = KGEnv(self.dm, self.dataset, self.single_relation_pair,  
        self.embedding, is_distance, seed, self.available_cores, self.path_length, 
        self.regenerate_embeddings, self.normalize_embeddings, self.gpu_acceleration,
        self.use_episodes, self.laps, verbose = self.verbose)

        self.agent = Agent(self.dm, self.env, self.gamma, self.learning_rate, self.use_LSTM, 
        self.activation, self.regularizers, self.reward_computation, self.guided_to_compute, 
        self.action_picking_policy, self.algorithm, self.guided_reward, self.reward_type,
        self.alpha, self.restore_agent, debug = self.debug, verbose = self.verbose)

        self.utils.write_log(f"START LOG FOR DATASET \"{self.dataset}\" \
        USE_LSTM: {self.use_LSTM} \
        EMBEDDING: {self.env.selected_embedding_name}\
        PATH_LENGTH: {self.path_length}\
        LR: {self.learning_rate} \
        GAMMA: {self.gamma}\
        EPISODES: {self.episodes}\
        ACTIVATION: {self.activation}\
        REGULARIZERS: {self.regularizers}\
        REWARD_COMPUTATION: {self.reward_computation}\
        ACTION_PICK_POLICY: {self.action_picking_policy}")
        
        self.score_history = []
    
    def run_prep(self):
        if(self.use_episodes):
            iterations = self.episodes
        else:
            iterations = self.laps*len(self.env.triples)
            self.episodes = iterations

        self.update_gui_vars(tot_steps = iterations, progtext="Running Trainer...")
        print(f"running for {iterations} iterations")

        to_iter_over = tqdm(range(iterations),
        f"Running episodes for: {self.dataset}-{self.env.selected_embedding_name}")

        self.is_ready = True
        return to_iter_over

    def episode_misc(self, episode, score, loss, last_action, reached_end_node):
        #Logs, debugs, saving the model
        log_msg = f"Episode: {episode+1}, Score:{score}, Loss:{loss}, Average Score:{np.mean(self.score_history)} \
        Target: {self.env.target_triple[0]}-{self.env.target_triple[1]}-{self.env.target_triple[2]}, \
        Destination: {last_action[2]}, Arrival: {reached_end_node} \
        Path: {self.agent.actions_mem}"

        self.utils.verb_print(f"\n{log_msg}\n")
        self.utils.write_log(log_msg)
        if(self.debug):
            end = self.debug_handle()
            if(end):
                return False
        
        if(((episode+1)%500 == 0 and episode != 0) or episode+1 == self.episodes):

            if(self.algorithm == "PPO"):
                model_to_save = [self.agent.policy_network, self.agent.critic]
            else:
                model_to_save = [self.agent.policy_network]

            self.dm.saveall(self.dataset, self.env.distance_cache,
            f"{self.dataset}-{self.env.selected_embedding_name}",
            model=model_to_save)

        return True

    def debug_handle(self):
        w = self.agent.policy_network.get_layer(index=1).get_weights()
        contains_nan = set(tf.math.is_nan(w[0][0]).numpy())
        if(not (True in contains_nan)):
            print("saving agent model, no nans.")
            if(self.algorithm == "PPO"):
                model_to_save = [self.agent.policy_network, self.agent.critic]
            else:
                model_to_save = [self.agent.policy_network]

            self.dm.save_agent_model(
                f"{self.dataset}-{self.env.selected_embedding_name}",
                model=model_to_save)
        else:
            print("there are nans in the agent. ending training")
            self.dm.debug_save(f"{self.env.dataset_name}-{self.env.selected_embedding_name}")
            self.utils.write_log("Network returned NaN, halted execution.\n")
            return True

    def set_gpu_config(self, use_gpu):
        if not use_gpu:
            print("not using GPU.")
            try:
                tf.config.set_visible_devices([], 'GPU')
                visible_devices = tf.config.get_visible_devices()
                for device in visible_devices:
                    assert device.device_type != 'GPU'
            except:
                print("problem setting GPUs...")
                pass
        
        if(not self.verbose):
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def run_debug(self, i = None):
        self.dm.debug_load(i, self.print_layers)
    
    def run(self):
        'Runs the environment and agent iterations.'
        to_iter_over = self.run_prep()

        for episode in to_iter_over:
            self.update_gui_vars(curr_step = episode+1)
            
            self.env.reset()
            done = False
            score = 0 

            # done changes to true when the number of steps is completed indicated by the path length.
            while not done:
                # Ask the agent to decide which action to take from this list.
                action, observation, reward, max_rew = self.agent.select_action()
                self.utils.verb_print(f"step output: action>{action}, reward>{reward}, max reward>{max_rew}\n\n ---o--- \n")
                _, done, _ = self.env.step(action)
                self.agent.remember(action, observation, reward, max_rew)
                score += reward
            
            self.score_history.append(score)

            last_action = self.env.path_history[len(self.env.path_history)-1]
            reached_end_node = last_action[2] == self.env.target_triple[2]
            
            if(self.debug):
                self.dm.update_lastest_input(np.array(self.agent.input_mem))

            # Updating NN wheights
            loss = self.agent.learn()

            cont = self.episode_misc(episode, score, loss, last_action, reached_end_node)
            if(not cont):
                return False

        print("out of loop")

        return True 

    def update_gui_vars(self, tot_steps = None, curr_step = None, progtext = None):
        if(tot_steps is not None):
            self.total_iter_steps = tot_steps
        
        if(curr_step is not None):
            self.current_iter_steps = curr_step

        if(progtext is not None):
            self.current_progress_text = progtext

class TrainerGUIconnector(object):
    def __init__(self, config: dict, experiments):
        self.active_trainer = None
        self.config, self.experiments = config, experiments
        self.started = False

        global tr_total_iterations
        global tr_current_iteration
        global tr_current_progress_text

        tr_total_iterations = len(experiments)
        tr_current_iteration = 1
        tr_current_progress_text = "Intialization" 

        self.train_thread = threading.Thread(name="trainThread", target=self.threaded_update, daemon=True)

    def start_connection(self):
        self.started = True
        self.train_thread.start()

    def update_current_trainer(self, t:Trainer):
        self.active_trainer = t

    def update_info_variables(self):
        global tr_total_iter_steps
        global tr_current_iter_steps
        global tr_current_progress_text

        tr_total_iter_steps = self.active_trainer.total_iter_steps
        tr_current_iter_steps = self.active_trainer.current_iter_steps
        tr_current_progress_text = self.active_trainer.current_progress_text

    def threaded_update(self):
        while(True):
            self.update_info_variables()
            time.sleep(1)

def main(from_file, gui_connector : TrainerGUIconnector = None):
    if(from_file):
        config, EXPERIMENTS = get_config(train=True)
    else:
        print(gui_connector.experiments)
        config, EXPERIMENTS = gui_connector.config, gui_connector.experiments

    global tr_total_iterations
    global tr_current_iteration

    aux = [len(e.embeddings) for e in EXPERIMENTS]
    tr_total_iterations = sum(aux)
    tr_current_iteration = 0

    for e in EXPERIMENTS:
        config["laps"] = e.laps 
        config["dataset"] = e.dataset
        config["single_relation_pair"] = [e.single_relation, e.relation_to_train]
        config["name"] = e.name
        config["embeddings"] = e.embeddings # for config copy.

        for emb in e.embeddings:
            tr_current_iteration +=1
            
            config["embedding"] = emb
            m = Trainer(config, from_gui=not from_file, gui_connector=gui_connector)
            
            if gui_connector is not None:
                gui_connector.update_current_trainer(m)
                if(not gui_connector.started):
                    gui_connector.start_connection()

            hasFinished = m.run()
            if(not hasFinished and config["debug"]):
                m.run_debug()

def get_gui_values():
    return tr_total_iterations, tr_current_iteration, tr_total_iter_steps, tr_current_iter_steps, tr_current_progress_text

if __name__ == "__main__":
    main(True)