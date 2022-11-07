config = {
    "available_cores": 6, #number of cpu cores to use when computing the reward
    "gpu_acceleration": True, # wether to use GPU(S) to perform fast training & embedding generation.

    "verbose": False, # prints detailed information every episode.
    "log_results": False, # Logs the results in the logs folder of episode training.

    "debug": False,
    "print_layers": False, # if debug is active wether to print the layer wheights on crash info.
    
    "guided_reward": True, # wether to follow a step-based reward or just a reward at the end of the episode.
    # if guided rewards are active, which one(s) to use:
    # distance: computes the distance to the final node and gives a score based on it.
    # embedding: based on the similarity to the end node we reward the agent.
    # terminal: reward if we are on the final node, 0 otherwise.
    "guided_to_compute":["terminal", "distance"], #"distance","terminal","embedding"

    "restore_agent": False, # continues the training from where it left off and loads the agent if possible.
    "regenerate_embeddings":False, # if embedding is found and true re-calculates them.
    # if re-calculation is active, normalizes embedding values to be on the center of the N-dim embedding array
    # as well as normalizing its stdev based on the whole dataset embedding values.
    "normalize_embeddings":True,
    "use_LSTM": True, # wether to add LSTM layers to the model.

    "random_seed":True, # to repeat the same results if false.
    "seed":78534245, # sets the seed to this number.

    # if True will use N episodes to train and random triples in the dataset
    # otherwise it will loop randomly over the dataset the specified ammount
    "use_episodes": False, 
    "episodes":0, # number of episodes, recommended -> 20k - 1mil, depending on dataset size.

    "path_length":5, #range: 1-y, the length of the discovered paths y>10 is discouraged.

    "alpha": 0.9, # previous step network learning rate (for PPO only.)
    "gamma": 0.99, # decay rate of past observations for backpropagation

    "learning_rate": 1e-3, # neural network learning rate.
    
    # NOTE: tensorflow makes it so tanh activation functions increase the efficiency of
    # LSTM layers so they calculate much faster consider using them when possible.
    # activation function for intermediate layers.
    "activation":'leaky_relu', # relu, prelu, leaky_relu, elu, tanh
    
    # applies L1 and L2 regularization.
    "regularizers":['kernel'], #"kernel", "bias", "activity"

    # PPO uses actor critic networks and BASE is a simple single network training
    # you can then choose retropropagation of rewards to compute as a REINFORCE model or simple to keep the rewards 
    # based on the results of the episode without adding any aditional computation to the reward.
    "algorithm": "BASE", #BASE, PPO
    "reward_type": "simple", # retropropagation, simple

    # modifies how the y_true value is calculated.
    # max_percent -> computes the maximum reward for the step and gives 0-1 according to how close we got to it.
    # one_hot_max -> gives 1 if we got the maximum reward for the episode 0 otherwise
    # straight -> gives the reward straight to the network as calculated from the step.
    "reward_computation": "one_hot_max", #"max_percent", "one_hot_max", "perfection", "straight"
    
    # probability->calculates an action with the weights provided by the probabilities.
    # max -> chooses the maximum probability value outputed by the network.
    # probability makes training stochasting while max makes it deterministic.
    "action_picking_policy":"probability",# "probability", "max"
}

class Experiment():
    'defines the experiment to run.'
    def __init__(self, experiment_name : str, dataset_name : str, 
    embeddings, laps : int = 0, single_relation : bool = False, relation : str = ""):

        self.name = experiment_name
        self.dataset = dataset_name
        self.single_relation = single_relation

        self.embeddings = []
        emb_mapping = {"TransE_l2":0, "DistMult":1, "ComplEx":2, "TransR":3}
        for e in embeddings:
            a = emb_mapping[e]
            if (type(a) == int):
                self.embeddings.append(a)

        self.laps = laps
        if(self.single_relation):
            self.relation_to_train = relation
        else:
            self.relation_to_train = None

class Test():
    def __init__(self, test_name, dataset_name : str, embeddings,
    episodes : int, single_relation : bool = False, relation : str = ""):

        self.name = test_name
        self.dataset = dataset_name
        self.single_relation = single_relation
        self.episodes = episodes
        self.embeddings = embeddings
        self.embedding_inds = []

        emb_mapping = {"TransE_l2":0, "DistMult":1, "ComplEx":2, "TransR":3}

        for e in self.embeddings:
            a = emb_mapping[e]
            if (type(a) == int):
                self.embedding_inds.append(a)
        
        if(self.single_relation):
            self.relation_to_train = relation
        else:
            self.relation_to_train = None

   
EXPERIMENTS = [
    # Experiment("Umls-distancerew-125laps-PPO", "UMLS", ["TransE_l2"], 10),

    Experiment("film_genre_FB_Base_simple_distance_100", "FB15K-237", ["TransE_l2"], 100, True, relation = "/film/film/genre"),

    # Experiment("embedding_testing", "NELL-995", ["TransE_l2"], 10, True, relation = "concept:animalpreyson"),
    # Experiment("distance_testing", "COUNTRIES", ["TransE_l2"], 10, True, relation = "concept:animalpreyson"),

    # Experiment("Countries 500 base", "COUNTRIES", ["TransE_l2"], 
    # 500, single_relation=False, relation="neighborOf")
]

TESTS = [
    Test("test-UMLS-embedding", "COUNTRIES", ["TransE_l2"], 150),
    # Test("COUNTRIES", ["TransE_l2"], 500, single_relation=True, relation="neighborOf")
]

# "COUNTRIES":([0],500)
# "COUNTRIES":([0,3],100), #521.550 iters
# "UMLS":([0],40), #172.128 iters
# "KINSHIP":([1,2],40), #213.600 iters 
# "WN18RR":([0,1,2,3],12), #1.042.020 iters
# "FB15K-237":([0,1,2,3],4), #1.088.460 iters
# "NELL-995":([0,1,2,3],7), #1.045.877 iters

def get_config(train):
    if train:
        return config, EXPERIMENTS
    else:
        return config, TESTS