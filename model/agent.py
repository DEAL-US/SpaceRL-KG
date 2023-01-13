import numpy as np
from keras import regularizers
from keras.layers import Dense, LSTM, Input, LeakyReLU, Reshape
from keras.models import Model, model_from_json
from keras.optimizers import adam_v2, rmsprop_v2
import keras.backend as K
from tqdm import tqdm
from multiprocessing import Process

import time, traceback

from data.data_manager import DataManager
from environment import KGEnv

from utils import Utils

class Agent(object):
    """
    The RL agent class, it contains the keras models, and is the one tasked with calculating the possible actions in every step,
    the probabilities, the rewards, storing the memories for each step, etc...

    :param data_manager: the instantiated data manager for this training/testing instance.
    :param environment: the linked environment for this agent.
    :param gamma: [0.90-0.99] decay rate of past observations for backpropagation
    :param learning_rate: [1e-3, 1e-5] neural network learning rate.
    :param alpha: [0.8-0.99] previous step network learning rate (for PPO only.)
    :param use_LSTM: wether to add LSTM layers to the model
    :param activation: activation function for intermediate layers. options: "relu", "prelu", "leaky_relu", "elu", "tanh"
    :param regularizers: applies L1 and L2 regularization at different stages of training. options: "kernel", "bias", "activity" 
    :param rew_comp: modifies how the y_true value is calculated. options: "max_percent", "one_hot_max", "straight"
    :param guided_reward: wether to follow a step-based reward or just a reward at the end of the episode.
    :param guided_options: if guided rewards are active, which one(s) to use. options: "distance","terminal","embedding"
    :param action_pick_policy: how the actions are chosen from the list of possible ones in every step. options: "probability", "max"
    :param algorithm: which algorithm to use when learning. options: "BASE", "PPO"
    :param reward_type: which way to feed the rewards to the network. options: "retropropagation", "simple"
    :param restore_agent: if true continues the training from where it left off and loads the agent if possible.
    :param verbose: if true prints detailed information every episode.
    :param debug: if true offers information about crashes, runs post-mortem.

    :returns: None
    """

    def __init__(self, data_manager: DataManager, environment: KGEnv, gamma:float,
    learning_rate:float, use_LSTM:bool, activation:str, regularizers:list, rew_comp:str, guided_options:list, action_pick_policy:str, 
    algorithm:str, guided_reward:bool, reward_type:str, alpha:float, restore_agent:bool, verbose:bool = False, debug:bool = False):
        
        # Global params
        self.dm = data_manager
        self.debug = debug
        self.verbose = verbose
        self.utils = Utils(self.verbose, False)

        self.activation = activation
        self.env = environment
        self.regularizers = regularizers
        self.reward_computation = rew_comp
        self.algorithm = algorithm
        self.guided_reward = guided_reward
        self.reward_type = reward_type
        self.guided_options = guided_options
        self.action_pick_policy = action_pick_policy

        self.advantages = np.zeros((self.env.path_length,1))
        self.old_pred = np.zeros((self.env.path_length,1))

        # algorithmic params
        self.gamma = gamma
        self.alpha = alpha
        
        # Embedding params
        self.entity_emb = self.env.entity_emb
        self.relation_emb = self.env.relation_emb

        # initialize memory
        self.reset()

        LTSM_layer_size = 400
        hiden_layer_size = 400

        #calculate tensor size for the observation space and action space.
        self.obs_len = self.env.observation_space.shape[0] # the state space, defined in the environment.
        self.action_len = self.env.action_space.shape[0] # the action space, defined in the environment as a property.

        # KERAS model definition:
        # the input size is determined by history layer output, + prev action taken + current observations + evaluated action
        input_size = self.action_len + self.obs_len #+ self.action_len + self.history_emb_len
        
        if(restore_agent):
            try:
                if(self.algorithm =="PPO"):
                    self.policy_network, self.critic = self.dm.restore_saved_agent_PPO(f"{self.env.dataset_name}-{self.env.selected_embedding_name}")
                    self.old_policy_network = self.build_network_from_copy(self.policy_network, learning_rate)
                else:
                    self.policy_network = self.dm.restore_saved_agent(f"{self.env.dataset_name}-{self.env.selected_embedding_name}")
            except Exception as e:
                print("couldn't restore the network, building new one.")
                print("related exception:\n")
                traceback.print_exc()
                self.policy_network, self.critic, self.old_policy_network = self.build_policy_networks(input_size, LTSM_layer_size, hiden_layer_size, learning_rate, use_LSTM)
        else:
            self.policy_network, self.critic, self.old_policy_network = self.build_policy_networks(input_size, LTSM_layer_size, hiden_layer_size, learning_rate, use_LSTM)

    ##########################
    #    NETWORK BUILDING    #
    ##########################

    def policy_config(self):
        """
        Performs some structural checks for the network building.

        :returns:
        kernel_init -> kernel initializer string description for keras model.\n
        activation -> performs a leaky ReLU initialization if the activation parameter demands for it. \n  
        k_reg -> initalizes kernel regularization  \n
        b_reg -> initalizes bias regularization  \n
        a_reg -> initalizes activity regularization  \n

        """
        kernel_init = 'glorot_uniform'
        if(self.activation == "selu"):
            kernel_init = 'lecun_normal'

        if(self.activation == "relu" or self.activation == "elu" 
        or self.activation == "selu" or self.activation == "tanh"):
            activation = self.activation
        elif(self.activation == "leaky_relu"):
            activation = LeakyReLU(alpha=0.01)

        k_reg = regularizers.L1L2(l1=1e-5, l2=1e-4) if('kernel' in self.regularizers) else None
        b_reg = regularizers.L2(1e-4) if('bias' in self.regularizers) else None
        a_reg = regularizers.L2(1e-5) if('activity' in self.regularizers) else None

        return kernel_init, activation, k_reg, b_reg, a_reg

    def build_policy_networks(self, input_size:int, LTSM_layer_size:int, hidden_layer_size:int, lr:float, use_lstm:bool):
        """
        initializes the keras model based on the size of the input layer given.
        Returns a model with a sigmoid activation as the last layer to be used as the probability for the specific action passed as an input.

        :param input_size: the input size of the network to build.
        :param LTSM_layer_size: the intermediate LSTM layer size.
        :param hidden_layer_size: the intermediate Dense layer size.
        :param lr: learning rate of the network.
        :param use_lstm: wether to use LSTM intermediate layers or not.

        :returns:
        
        policy, None, None (For BASE algorithm) **OR** actor, critic, actor_copy (For PPO algorithm)\n

        policy & actor -> represent the models that calculate the action probabilities for each step.\n
        critic -> evaluates the rewards vs the progress made and its used to better this calculation.\n
        actor_copy -> is a clone of the actor network that is one step behind, used for PPO calculations.\n
        
        """
        # Configuration dependent.
        kernel_init, activation, k_reg, b_reg, a_reg = self.policy_config()

        # Network.
        state = Input(batch_shape = (self.env.path_length, input_size), name="state") #batched
        # state = Input(shape=(input_size,), name="state") #unbatched

        dense1 = Dense(hidden_layer_size, activation=activation,
        kernel_initializer= kernel_init, kernel_regularizer= k_reg,
        bias_regularizer= b_reg, activity_regularizer= a_reg)(state)

        if(use_lstm):
            lstms = self.lstm_middle(dense1, LTSM_layer_size, hidden_layer_size,
            activation, kernel_init, k_reg, b_reg, a_reg)

            dense2 = Dense(hidden_layer_size, activation=activation,
            kernel_initializer= kernel_init, kernel_regularizer= k_reg,
            bias_regularizer= b_reg, activity_regularizer= a_reg)(lstms)
        else:
            dense2 = Dense(hidden_layer_size, activation=activation,
            kernel_initializer= kernel_init, kernel_regularizer= k_reg,
            bias_regularizer= b_reg, activity_regularizer= a_reg)(dense1)

        dense3 = Dense(hidden_layer_size, activation=activation,
        kernel_initializer= kernel_init, kernel_regularizer= k_reg,
        bias_regularizer= b_reg, activity_regularizer= a_reg)(dense2)

        probs = Dense(1, activation="sigmoid", name='actor_output')(dense3)
        
        if(self.algorithm == "PPO"):
            # the advantages given by the critic
            advantage = Input(batch_shape = (self.env.path_length, 1), name="Advantage")

            # Generally recieves the action taken as one-hot encoded,
            # we pass the prediciton for that action instead.
            old_pred = Input(batch_shape = (self.env.path_length, 1), name="Old_Prediction")
            
            optimizer = rmsprop_v2.RMSProp(learning_rate=lr)  #, clipvalue=0.5, clipnorm=1)
            actor = Model(inputs = [state, advantage, old_pred], outputs = [probs])
            actor.compile(loss = 'mean_squared_error', optimizer=optimizer)# self.proximal_policy_loss(advantage=advantage, old_prediction=old_pred)
            
            print("=== Build Actor Network ===")
            print(actor.summary())
    
            critic = self.build_critic_network(input_size, hidden_layer_size, LTSM_layer_size)
            optimizer=adam_v2.Adam(learning_rate=lr)
            critic.compile(loss="mean_squared_error", optimizer=optimizer)

            print("=== Build Critic Network ===")
            print(critic.summary())

            actor_copy = self.build_network_from_copy(actor, lr)

            return actor, critic, actor_copy

        else:
            policy = Model(inputs = [state], outputs = [probs])
            optimizer = adam_v2.Adam(learning_rate=lr)  #, clipvalue=0.5, clipnorm=1)
            policy.compile(loss = "mean_squared_error", optimizer = optimizer)

            print("=== Build Policy Network ===")
            print(policy.summary())

            return policy, None, None

    def build_critic_network(self, input_size:int, hidden_layer_size:int, LTSM_layer_size:int):
        """
        Builds a critic network.

        :param input_size: the input size of the network to build. \n
        :param LTSM_layer_size: the intermediate LSTM layer size. \n
        :param hidden_layer_size: the intermediate Dense layer size. \n

        :returns: the critic network.

        """

        kernel_init, activation, k_reg, b_reg, a_reg = self.policy_config()

        state = Input(batch_shape = (self.env.path_length, input_size), name="state")

        dense1 = Dense(hidden_layer_size, activation=activation,
        kernel_initializer= kernel_init, kernel_regularizer= k_reg,
        bias_regularizer= b_reg, activity_regularizer= a_reg)(state)

        lstms = self.lstm_middle(dense1, LTSM_layer_size, hidden_layer_size, 
        activation, kernel_init, k_reg, b_reg, a_reg)

        dense2 = Dense(hidden_layer_size, activation=activation,
        kernel_initializer= kernel_init, kernel_regularizer= k_reg,
        bias_regularizer= b_reg, activity_regularizer= a_reg)(lstms)

        dense3 = Dense(hidden_layer_size, activation=activation,
        kernel_initializer= kernel_init, kernel_regularizer= k_reg,
        bias_regularizer= b_reg, activity_regularizer= a_reg)(dense2)

        q = Dense(1, activation="sigmoid", name="critic_output_layer")(dense3)

        critic_network = Model(inputs=state, outputs=q)

        return critic_network

    def build_network_from_copy(self, network:Model, learning_rate:float):
        """
        Copies the given network

        :param network: the network to copy \n
        :param learning_rate: the new learning rate for the network. \n

        :returns: the copied network.

        """
        network_structure = network.to_json()
        network_weights = network.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        network.compile(optimizer=adam_v2.Adam(learning_rate=learning_rate), loss="mse")
        return network

    def lstm_middle(self, prev_layer:Dense, LTSM_layer_size:int, 
    hidden_layer_size:int, activation:str, kernel_init:str, k_reg:regularizers.L1L2, b_reg:regularizers.L1, a_reg:regularizers.L1):
        """
        constructs the intermediate LSTM layers for the network.

        :param prev_layer: The dense layer to connect them to.
        :param LTSM_layer_size: the intermediate LSTM layer size.
        :param hidden_layer_size: the intermediate Dense layer size.
        :param activation: activation fuction for intermediate layers.
        :param kernel_init: kernel initializer string description for keras model.
        :param k_reg: kernel regularizer
        :param b_reg: bias regularizer
        :param a_reg: activity regularizer

        :returns: the last linked LSTM layer to connect to the model.
        """
        
        reshape = Reshape((1, hidden_layer_size))(prev_layer)
        # print(type(reshape))

        lstm1 = LSTM(LTSM_layer_size, return_sequences=True, activation=activation,
        kernel_initializer= kernel_init, kernel_regularizer= k_reg,
        bias_regularizer= b_reg, activity_regularizer= a_reg)(reshape)
        # print(type(lstm1))

        lstm2 = LSTM(LTSM_layer_size, return_sequences=True, activation=activation,
        kernel_initializer= kernel_init, kernel_regularizer= k_reg,
        bias_regularizer= b_reg, activity_regularizer= a_reg)(lstm1)
        # print(type(lstm2))

        lstm3 = LSTM(LTSM_layer_size, return_sequences=False, activation=activation, name="last_lstm", 
        kernel_initializer= kernel_init, kernel_regularizer= k_reg,
        bias_regularizer= b_reg, activity_regularizer= a_reg)(lstm2)
        # print(type(lstm3))

        return lstm3

    #############################
    #    ACTIONS AND REWARDS    #
    #############################
    def encode_action(self, chosen_rel:str, chosen_ent:str):
        """
        Encodes the action as its embedding representation

        :param chosen_rel: entity to encode \n
        :param chosen_ent: relation to enconde \n

        :returns: [\*relation_embedding, \*entity_embedding]

        """
        if(chosen_rel == "NO_OP"):
            rel_emb = list(np.zeros(int(self.action_len/2)))
        else:
            rel_emb = self.relation_emb[chosen_rel]

        ent_emb = self.entity_emb[chosen_ent]
        return [*rel_emb, *ent_emb]
        
    def get_next_state_rewards(self, actions_takens:list):
        """
        Gets the reward for the next state if that action is chosen, its calculated in 3 toggleable steps

        - terminal rewards -> returns 1 if agent is located at the tail entity at the end of the episode, adds 0 otherwise. \n
        - distance -> we reward the agent based on the distance to the tail entity \n
        - embedding -> computes several metrics of embedding similarity and uses them to compute a reward. \n

        :param action_taken: actions being evaluated.
        """
        for action_taken in actions_takens:
            encoded_action = self.encode_action(action_taken[0], action_taken[1])
            input_arr = [*self.observations, *encoded_action] # [(*e1,*r),*et] [*relation_embedding, *entity_embedding]
            
            new_state_node = action_taken[1]
            dest_node = self.env.target_triple[2]

            ratio_emb, ratio_dist = 0.7, 0.3
            
            # self.guided_options -> ["distance","terminal","embedding"]
            emb_dists, distance = None, None

            if(self.guided_reward):

                if "embedding" in self.guided_options:
                    # we compare the embedding operations to the ones in the previous step
                    # and reward the agent if we are closer to the end node according to the embedding.
                    emb_dists = self.env.get_embedding_info(new_state_node, dest_node)
                    latest = self.emb_metrics_mem[-1]
                    
                    # [dot, euc_dist, cos_sim]
                    aux = [0,0,0]

                    # dot product is maximization.
                    if emb_dists[0] > latest[0]: 
                        aux[0] = 1/3
                    elif emb_dists[0] == latest[0]: # if it didn't improve we halve the reward
                            aux[0] = 1/6

                    # cos_sim and euc_dist is minimization
                    for i in range(1,3):
                        if emb_dists[i] < latest[i]: 
                            aux[i] = 1/3
                        elif emb_dists[i] == latest[i]: # if it didn't improve we halve the reward
                            aux[i] = 1/6

                    emb_rew = sum(aux)

                if "distance" in self.guided_options:
                    distance = self.env.get_distance_net_x(new_state_node, dest_node)
                    latest = self.distance_mem[-1]

                    if distance is None:
                        distance = 99
                        dist_rew = 0.005
                    else:
                        if distance < latest:
                            dist_rew = 1
                        elif distance == latest:
                            dist_rew = 1/3
                        else:
                            dist_rew = 0.005 

                if "terminal" in self.guided_options and new_state_node == dest_node:
                    # with terminal rewards active, if we are in the end node reward is max.
                    total_rew = 1
                    
                # Discourage the NO_OP unless it is to stay on the end node
                elif action_taken[0]=="NO_OP" and (new_state_node != dest_node):
                    total_rew = 0.005

                elif("embedding" in self.guided_options and "distance" in self.guided_options):
                    total_rew = dist_rew*ratio_dist + emb_rew*ratio_emb
                else:
                    if("embedding" in self.guided_options):
                        total_rew = emb_rew
                    
                    elif("distance" in self.guided_options):
                        total_rew = dist_rew

            else: # Not guided reward, return 1 if in end node.

                if new_state_node == dest_node:
                    total_rew =  1
                else:
                    total_rew = 0.05

            self.all_calculations.append([input_arr, total_rew, emb_dists, distance]) 
            
    def get_inputs_and_rewards(self):
        """
        Based on the current environment state, gets the rewards to all actions in the current state.

        :returns: 
        inputs -> all the encoded actions that the agent can perform in the current step. \n
        rewards -> the calculated rewards for every action possible. \n

        """
        self.inputs, self.state_rewards, self.step_distances, self.step_embeddings = [], [], [], [] 
        self.observations = self.env.get_encoded_observations()

        # Evaluate actions and build probability & rewards list.
        if(self.verbose):
            it = tqdm(self.env.actions, desc="iterating over actions")
        else:
            it = self.env.actions

        # initialize embedding and distance reward lists.
        origin_node, target_rel, dest_node = self.env.target_triple

        if "embedding" in self.guided_options and len(self.emb_metrics_mem) ==0:
            baseline_emb_dist = self.env.get_embedding_info(origin_node, dest_node)
            self.emb_metrics_mem.append(baseline_emb_dist)

        if "distance" in self.guided_options and len(self.distance_mem) ==0:
            # baseline_dist = self.env.get_distance(origin_node, dest_node)
            baseline_dist = self.env.get_distance_net_x(origin_node, dest_node, target_rel)

            if(baseline_dist is None):
                # 99 is an arbitrary number which will always be bigger than a valid path.
                self.distance_mem.append(99)
            else:
                self.distance_mem.append(baseline_dist)
        
        # intermediate list to store multithreaded calculations and keep the relative order
        self.all_calculations = [] 

        if len(it) > self.env.threads * 5:
            print(f"multiprocessing {len(it)} possible actions...")
            init_time = time.time()

            thread_list = []
            sublist_size = len(it)//self.env.threads

            for i in range(self.env.threads):
                init_index = (sublist_size*i)
                last_index = len(it) if self.env.threads == i+1 else sublist_size*(i+1)-1

                x = Process(target = self.get_next_state_rewards, args = [it[init_index:last_index]])

                thread_list.append(x)
                x.start()

            for t in thread_list:
                t.join()

            print(f"took {time.time() - init_time} to process")
        else:
            print(f"single processing {len(it)} possible actions...")
            init_time = time.time()

            self.get_next_state_rewards(it)

            print(f"took {time.time() - init_time} to process")

        self.all_calculations = list(map(list, zip(*self.all_calculations)))

        self.inputs.extend(self.all_calculations[0]) 
        self.state_rewards.extend(self.all_calculations[1]) 
        self.step_embeddings.extend(self.all_calculations[2]) 
        self.step_distances.extend(self.all_calculations[3])

        return self.inputs, self.state_rewards

    def get_network_outputs(self, num_actions:int, inputs:list):
        """
        Calculates the output of the network for all actions in the step

        :param action_taken: action being evaluated.

        :returns: all the network outputs for each input.

        """

        # this number was chosen for a 3080 TI with 10GB of memory
        # if you find an error near here you might want to lower it
        # of if you have a graphics card with more memory you might wanna raise it.
        sub_batch_size = 750

        if num_actions > sub_batch_size:  
            n_sub_batches = int(num_actions/sub_batch_size) + 1 
            sub_batch_len = int(num_actions/n_sub_batches) + 10
            
            self.utils.verb_print(f"action batch is big, partitioning.\n number of sub_batchs for {num_actions} actions is: {n_sub_batches} and their length is: {sub_batch_len}")
            
            subinputs = [inputs[i : i + sub_batch_len] for i in range(0, num_actions, sub_batch_len)]
            
            outputs = []
            for s in subinputs:
                inputs_stacked = np.vstack(np.array(s))
                start = time.time()

                if(self.algorithm == "PPO"):
                    suboutputs = self.policy_network([inputs_stacked, self.advantages, self.old_pred]) 
                else:
                    suboutputs = self.policy_network([inputs_stacked])
            
                self.utils.verb_print(f"calculated {len(s)} actions in: {time.time() - start} seconds")
                
                outputs.extend(suboutputs)
        else:
            inputs_stacked = np.vstack(np.array(inputs)) #(batch_len,1)
            start = time.time()

            if(self.algorithm == "PPO"):
                outputs = self.policy_network([inputs_stacked, self.advantages, self.old_pred]) 
            else:
                outputs = self.policy_network([inputs_stacked]) 
            
            self.utils.verb_print(f"calculated {len(self.env.actions)} actions in: {time.time() - start} seconds")

        #remove the array wrapper from values.
        outputs = [float(item[0]) for item in outputs]

        return outputs

    def pick_action_from_outputs(self, outputs:list):
        """
        Given the network outputs, calculate the action to be taken

        :param outputs: the network outputs for all actions.

        :returns: 
        chosen_action -> the chosen action \n
        chosen_action_index -> the index of the action in the action list. \n

        """
        # if network returns all zeroes avoid nan by choosing random.
        m = max(outputs)
        if not np.array(outputs).any():
            chosen_action_index = np.random.choice(len(self.env.actions))
        else:    
            if(self.action_pick_policy == "max"):
                chosen_action_index = outputs.index(m)
            elif(self.action_pick_policy == "probability"):
                probs = outputs/np.sum(outputs)
                chosen_action_index = np.random.choice(range(len(self.env.actions)), p=probs)
        
        chosen_action = self.env.actions[chosen_action_index]
        
        return chosen_action, chosen_action_index #, probs[chosen_action_index]

    def select_action(self):
        """
        Evaluating the actions to be taken, as well as the query triple and the location of exploration, calculates the action to follow.

        :returns:
        chosen_action -> the literal representation action that was chosen to be followed.\n
        inputs[chosen_action_index] -> the encoded representation of the action that was selected. \n
        rewards[chosen_action_index] -> the rewards that were calculated for that action \n
        max(rewards) -> the best reward in the episode. \n

        """

        inputs, rewards = self.get_inputs_and_rewards()
        outputs = self.get_network_outputs(len(self.env.actions), inputs)
        chosen_action, chosen_action_index = self.pick_action_from_outputs(outputs)
   
        self.utils.verb_print(f"predicted output from network:\n {outputs}\nrewards:\n {rewards} \nchosen action for step was: {chosen_action}")
        
        if(self.guided_reward):
            if "distance" in self.guided_options:
                self.utils.verb_print(f"calculated distances to end_node {self.step_distances}")
                self.distance_mem.append(self.step_distances[chosen_action_index])
                
            if "embedding" in self.guided_options:
                self.utils.verb_print(f"calculated embedding metrics {self.step_embeddings}")
                self.emb_metrics_mem.append(self.step_embeddings[chosen_action_index])

        return chosen_action, inputs[chosen_action_index], rewards[chosen_action_index], max(rewards)

    def select_action_runtime(self):
        """
        Simplified version of the action selector, it calculates nothing, only gets the outputs from the trained model. Used by tester.

        :returns: the literal representation of the action that was chosen by the agent.

        """

        inputs = []
        observations = self.env.get_encoded_observations()
        for a in self.env.actions:
            encoded_action = self.encode_action(a[0], a[1])
            input_arr = [*observations, *encoded_action] #[(*e1,*r),*et] [*relation_embedding, *entity_embedding]
            inputs.append(input_arr)

        num_actions = len(inputs)
        sub_batch_size = 750
        if num_actions > sub_batch_size:
            n_sub_batches = int(num_actions/sub_batch_size) + 1 
            sub_batch_len = int(num_actions/n_sub_batches) + 10
            subinputs = [inputs[i : i + sub_batch_len] for i in range(0, num_actions, sub_batch_len)]
            outputs = []
            for s in subinputs:
                inputs_stacked = np.vstack(np.array(s))
                if(self.algorithm == "PPO"):
                    suboutputs = self.policy_network([inputs_stacked, self.advantages, self.old_pred]) 
                else:
                    suboutputs = self.policy_network([inputs_stacked])
                outputs.extend(suboutputs)
        else:
            inputs_stacked = np.vstack(np.array(inputs)) #(batch_len,1)
            if(self.algorithm == "PPO"):
                outputs = self.policy_network([inputs_stacked, self.advantages, self.old_pred]) 
            else:
                outputs = self.policy_network([inputs_stacked])

        outputs = [float(item[0]) for item in outputs]
        probs = outputs/np.sum(outputs)
        chosen_action_index = np.random.choice(range(len(self.env.actions)), p=probs)
        chosen_action = self.env.actions[chosen_action_index]

        return chosen_action
        
    ########################
    #    POLICY UPDATES    #
    ########################

    def update_target_network(self):
        """
        Updates the old network in PPO before aplying the changes to it.

        """
        self.old_policy_network.set_weights(
        self.alpha * np.array(self.policy_network.get_weights()) +
        (1-self.alpha) * np.array(self.old_policy_network.get_weights()))

    def get_y_true(self, rew_mem:float, max_rew_mem:float):
        """
        Calculates the y_true based on the max reward for the episode and the reward of the chosen action.

        :param rew_mem: the calculated reward for the chosen action
        :param max_rew_mem: the maximum reward for the episode.

        :returns: y_true (the information to feedback to the network based on its action chosen.)
        """
        # CALCULATE THE Y_TRUE WE ARE TRYING TO APROXIMATE EITHER AS A
        # ONE-HOT ENCODED REPRESENTATION OF THE OPTIMAL PATH TO TAKE OR AS A
        # PERCENTAGE (ONE-NORM [0-1]) OF THE MAXIMUM REWARD.
        if(self.reward_computation == "max_percent"):
            y_true = rew_mem/max_rew_mem
            self.utils.verb_print(f"rew % correct: {y_true}")

        elif(self.reward_computation == "one_hot_max"):
            comb = list(zip(rew_mem, max_rew_mem))
            y_true = np.array([1 if(a == b) else 0 for a,b in comb])
            self.utils.verb_print(f"combined list: {comb}\none_hot_rew = {y_true}")

        elif(self.reward_computation == "straight"):
            y_true = rew_mem
            
        return y_true

    def learn(self):
        """
        Updates the policy network using the NN model.
        This function is used after the MC sampling is done following the function:
        
        Δθ = α * gradient + log(probabilities)

        :returns: loss
        """
        state_mem, _ , rew_mem, max_rew_mem, _ = self.numpyfy_memories()
            
        y_true = self.get_y_true(rew_mem, max_rew_mem)
        
        self.utils.verb_print(f"\
        y_true: {y_true}\n\
        rew_mem: {rew_mem}\n\
        max_rew_mem: {max_rew_mem}")

        if(self.algorithm == "BASE"):
            if(self.reward_type == "retropropagation"):
                discounted_r = self.calculate_backpropagation_rewards(rew_mem)
                self.utils.verb_print(f"discounted rewards: {discounted_r}")
                y_true = discounted_r/max(discounted_r)
                
            # TRAINING
            loss = self.policy_network.train_on_batch([state_mem], y_true)

        elif(self.algorithm == "PPO"):
            if(not self.guided_reward and (self.env.target_triple[-1] == self.actions_mem[-1][1])):
                self.rewards_mem[-1] = 1

            if self.reward_type == 'retropropagation':
                _, rew = self.calculate_PPO_rewards(self.rewards_mem[-1])
            else:
                rew = rew_mem
            
            values = self.critic([state_mem])
            values = np.hstack(values)

            # Compute advantages
            self.old_pred = self.old_policy_network([state_mem, self.advantages, self.old_pred])
            self.advantages = rew - values

            self.utils.verb_print(f"\
            discounted rewards: {rew}\n\
            critic pred: {values}\n\
            old_pred: {self.old_pred}\n\
            advantages pred: {self.advantages}")

            # TRAINING PPO
            loss = self.policy_network.train_on_batch([state_mem, self.advantages, self.old_pred], y_true)
            rew = np.vstack(np.array(rew))
            self.critic.train_on_batch([state_mem], rew)
            self.update_target_network()

        # reset memories after learning
        self.reset()
        return loss

    def calculate_backpropagation_rewards(self, rew_mem:float):
        """
        As per the REINFORCE implementation we calculate the backpropagation of the rewards for each step of the training
        we then use these values to train the NN.

        :param rew_mem: the calculated reward for the chosen action

        :returns: the propagated rewards calculated for each step.
        """
        propagation_rew = []
        for t in range(self.env.path_length):
            Gt = 0
            for pw, r in enumerate(rew_mem[t:], start = 1):
                Gt = Gt + self.gamma**pw * r

            propagation_rew.append(Gt)
                
        return np.array(propagation_rew)
    
    def calculate_PPO_rewards(self, last_rew:float):
        """
        Calculates the rewards for the PPO algorithm following its formula.

        :param last_rew: the last calculated reward before the end of the episode.

        :returns: 
        discounted_r_old -> the reward from the old policy network.
        discounted_r -> the calculated reward for the actor network.
        """
        Gt = 0
        discounted_r_old = np.zeros_like(self.rewards_mem, dtype="float64")
        for i in reversed(range(0,len(self.rewards_mem))):
            Gt = Gt * self.gamma + self.rewards_mem[i]     
            discounted_r_old[i] = Gt 

        discounted_r_old -= np.mean(discounted_r_old) #normalization
        discounted_r_old /= np.std(discounted_r_old)

        discounted_r = []
        v = last_rew
        for r in self.rewards_mem[::-1]:
            v = r + self.gamma * v
            discounted_r.append(v)

        discounted_r.reverse()
        return discounted_r_old, discounted_r

    #######################
    #    AUXILIARY OPS    #
    #######################

    def reset(self):
        """
        resets the memories of the episode
        """
        self.actions_mem = []
        self.input_mem = []
        self.rewards_mem = []
        self.max_rew_mem = []
        self.output_mem = []
        self.distance_mem = []
        self.emb_metrics_mem = []

    def remember(self, action, input, reward, max_rew):
        """
        stores the memories for the episode
        """
        self.input_mem.append(input)
        self.actions_mem.append(action)
        self.rewards_mem.append(reward)
        # self.output_mem.append(prob)
        self.max_rew_mem.append(max_rew)

    def numpyfy_memories(self):
        """ 
        converts the memories into numpy arrays and returns them
        """
        return np.array(self.input_mem), np.array(self.actions_mem), np.array(self.rewards_mem), np.array(self.max_rew_mem), np.array(self.output_mem)

    def stringify_actions(self, action_list):
        """ 
        converts the memories into numpy arrays 
        """
        res = []
        for a in action_list:
            res.append(f"{a[0]}, {a[1]}")

        return res
