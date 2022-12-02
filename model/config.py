config = {
    # general parameters do not affect the results of the training
    # they can be used to get more insight into what's happening behind the curtains
    # or to allocate more resources to the agents.

    # training specific parameters must be payed atention to when training a new agent
    # as they can vastly affect the outcome of said training.

    ######################
    # GENERAL PARAMETERS #
    ###################### 
    "available_cores": 4, #number of cpu cores to use when computing the reward
    "gpu_acceleration": False, # wether to use GPU(S) to perform fast training & embedding generation.

    "verbose": False, # prints detailed information every episode.
    "log_results": False, # Logs the results in the logs folder of episode training.

    "debug": False, # offers information about crashes, runs post-mortem
    "print_layers": False, # if debug is active wether to print the layer wheights on crash info.
    
    #####################
    # TRAINING SPECIFIC #
    ##################### 
    "restore_agent": True, # continues the training from where it left off and loads the agent if possible.
    
    "guided_reward": True, # wether to follow a step-based reward or just a reward at the end of the episode.
    # if guided rewards are active, which one(s) to use:
    # distance: computes the distance to the final node and gives a score based on it.
    # embedding: based on the similarity to the end node we reward the agent.
    # terminal: reward if we are on the final node, 0 otherwise.
    "guided_to_compute":["terminal", "embedding"], #"distance","terminal","embedding"

    "regenerate_embeddings":False, # if embedding is found and true re-calculates them.
    # if re-calculation is active, normalizes embedding values to be on the center of the N-dim embedding array
    # as well as normalizing its stdev based on the whole dataset embedding values.
    "normalize_embeddings":True,

    # LSTM layers allow the traine agents to leverage on the previous steps
    # its recommended to always leave this on unless to perform an ablation study.
    "use_LSTM": True, # wether to add LSTM layers to the model

    # if True will train for N episodes otherwise it will loop over the dataset
    # for an ammount defined in the laps attribute of the training.
    "use_episodes": False, 
    "episodes":0, # number of episodes to run.

    "alpha": 0.9, # [0.8-0.99] previous step network learning rate (for PPO only.) 
    "gamma": 0.99, # [0.90-0.99] decay rate of past observations for backpropagation
    "learning_rate": 1e-3, #[1e-3, 1e-5] neural network learning rate.
    
    # activation function for intermediate layers.
    # NOTE: tensorflow makes it so tanh activation functions increase the efficiency of
    # LSTM layers so they calculate much faster consider using them when possible.
    "activation":'leaky_relu', # relu, prelu, leaky_relu, elu, tanh
    
    # applies L1 and L2 regularization at different stages of training.
    "regularizers":['kernel'], #"kernel", "bias", "activity" 
    
    # which algorithm to use when learning.
    # PPO uses actor critic networks and BASE is a simple feed-forward network.
    # you can then choose retropropagation of rewards to compute as a REINFORCE model or simple to keep the rewards 
    # based on the results of the episode without adding any aditional computation to the reward.
    "algorithm": "BASE", #BASE, PPO

    # which way to feed the rewards to the network.
    # retroprogation causes the rewards closer to the end of the episode have more 
    # influence over the neural network, whereas simple offers a homogenous distibution.
    "reward_type": "simple", # retropropagation, simple
    
    # how the actions are chosen from the list of possible ones in every step.
    # probability-> action calculation biased by the probability of the network output.
    # max -> pick the highest value offered by the network.
    # probability makes training stochasting while max makes it deterministic.
    "action_picking_policy":"probability",# "probability", "max"

    # modifies how the y_true value is calculated.
    # max_percent -> computes the maximum reward for the step and gives 0-1 according to how close we got to it.
    # one_hot_max -> gives 1 if we got the maximum reward for the episode 0 otherwise
    # straight -> gives the reward straight to the network as calculated from the step.
    "reward_computation": "one_hot_max", #"max_percent", "one_hot_max", "straight"
    
    #################
    # SHARED PARAMS #
    #################
    # these parameters are shared by the trainer and tester suites.

    "path_length":5, #the length of path exploration.

    "random_seed":True, # to repeat the same results if false.
    "seed":78534245, # sets the seed to this number.
}

import pathlib

class Experiment():
    'defines the experiment to run.'
    def __init__(self, experiment_name : str, dataset_name : str, 
    embeddings, laps : int = 0, single_relation : bool = False, relation : str = ""):

        self.name = experiment_name
        self.dataset = dataset_name
        self.single_relation = single_relation
        self.embeddings = embeddings

        self.laps = laps
        if(self.single_relation):
            self.relation_to_train = relation
        else:
            self.relation_to_train = None

current_dir = pathlib.Path(__file__).parent.resolve()
agents_folder = pathlib.Path(f"{current_dir}/data/agents").resolve()

class Test():
    def __init__(self, test_name:str, agent_name:str, embeddings, episodes : int):
        self.name = test_name
        self.agent_name = agent_name
        self.episodes = episodes
        self.embeddings = embeddings
        self.to_delete = False

        agent_path = pathlib.Path(f"{agents_folder}/{self.agent_name}").resolve()
        try:
            config_used = open(f"{agent_path}/config_used.txt")
            for ln in config_used.readlines():
                if ln.startswith("dataset: "):
                    self.dataset = ln.lstrip("dataset: ").strip()

                if ln.startswith("single_relation_pair: "):
                    aux = ln.lstrip("single_relation_pair: ").strip()
                    aux = aux.replace("[", "").replace("]","").replace(" ", "").replace("\'", "").strip().split(",")
                    self.single_relation, self.relation_to_train = [aux[0]=="True", None if aux[1] == "None" else aux[1]]
        except:
            print(f"Incoherent Test detected: {self.name}")
            self.to_delete = True

EXPERIMENTS = [
    # Experiment("asd1", "COUNTRIES", ["TransE_l2"], 1)
    # Experiment("film_genre_FB_Base_PPO_embedding_22", "FB15K-237", ["TransE_l2"], 22, True, relation = "/film/film/genre"),
    # Experiment("countiesall", "COUNTRIES", ["TransR"], 1) 
    # Experiment("Umls-distancerew-125laps-PPO", "UMLS", ["TransE_l2"], 10),
    # Experiment("embedding_testing", "NELL-995", ["TransE_l2"], 10, True, relation = "concept:animalpreyson"),
    # Experiment("Countries 500 base", "COUNTRIES", ["TransE_l2"], 500, single_relation=False, relation="neighborOf")
]

TESTS = [
    # Test("countries-test-name", "countries-1", ["TransE_l2"], 5),
    # Test("another-test", "Countries-distancerewonly-250laps-PPO", ["TransE_l2"], 10),
    # Test("good_test", "countiesall", ["TransE_l2", "DistMult", "ComplEx"], 10),
]

TESTS = [t for t in TESTS if not t.to_delete]

def get_config(train, only_config = False):
    if(only_config):
        return config
    else:
        if train:
            return config, EXPERIMENTS
        else:
            return config, TESTS