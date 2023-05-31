config = {
    # general parameters do not affect the results of the training
    # they can be used to get more insight into what's happening behind the curtains
    # or to allocate more resources to the agents.

    # training specific parameters must be payed atention to when training a new agent
    # as they can vastly affect the outcome of said training.

    ######################
    # GENERAL PARAMETERS #
    ###################### 
    "available_cores": 6, #number of cpu cores to use when computing the reward
    "gpu_acceleration": True, # wether to use GPU(S) to perform fast training & embedding generation.
    # calculates the distance reward in the datasets by splitting the load into several
    # subprocesses, has a high overhead so its only recommended for large datasets.
    # if you precomputed the distance caches for the dataset its recommended to leave it false.
    "multithreaded_dist_reward": False, 

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
    "guided_to_compute":["terminal", "shaping"], #"distance","terminal","embedding","shaping"

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
    "algorithm": "PPO", #BASE, PPO

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

    "path_length":3, #the length of path exploration.

    "random_seed":True, # to repeat the same results if false.
    "seed":78534245, # sets the seed to this number.
}

import pathlib

class Experiment():
    """
    Defines an experiment suite to be carried out, and thus the agents to be created.

    :param experiment_name: the name of the experiment (directory name)
    :param dataset_name: the dataset to be used.
    :param embeddings: the embeddings to be used. options -> "TransE_l2", "DistMult", "ComplEx", "TransR"
    :param laps: agents trained by doing several laps around the dataset in order to minimize randomness, bigger datasets may require more laps but this will in turn increase training time, choose a single relation training in this case.
    :param single_relation: wether to train for a single relation or the entire graph.
    :param relation: the name of the relation to train for.

    """
    def __init__(self, experiment_name : str, dataset_name : str, 
    embeddings:list, laps : int = 0, single_relation : bool = False, relation : str = ""):

        self.name = experiment_name
        self.dataset = dataset_name
        self.single_relation = single_relation
        self.embeddings = embeddings

        self.laps = laps
        if(self.single_relation):
            self.relation_to_train = relation
        else:
            self.relation_to_train = None
    
    def __str__(self):
        return f"{self.name}, {self.dataset}, {self.single_relation}, {self.embeddings}, {self.laps}, {self.single_relation}, {self.relation_to_train}" 

    def __repr__(self):
        return f"{self.name}, {self.dataset}, {self.single_relation}, {self.embeddings}, {self.laps}, {self.single_relation}, {self.relation_to_train}" 

current_dir = pathlib.Path(__file__).parent.resolve()
agents_folder = pathlib.Path(f"{current_dir}/data/agents").resolve()

class Test():
    """
    Defines an test suite to be carried out and performs integrity checks.
    If no agent is found to support the required testing, the created test is discarded.

    :param test_name: the name of the test (directory name)
    :param agent_name: the agent to be used, if not found, test is discarded
    :param embeddings: the embeddings to be used. options -> "TransE_l2", "DistMult", "ComplEx", "TransR"
    :param episodes: the number of paths to evaluate in testing.

    """
    def __init__(self, test_name: str, agent_name: str, embeddings: list, episodes: int):
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
    Experiment("countries_reward_shaping", "COUNTRIES", ["TransE_l2"], 200),

    # Experiment("film_genre_FB_PPO_distance_22", "FB15K-237", ["TransE_l2"], 22, True, relation = "/film/film/genre"),

    # RL1 doing theese.
    # Experiment("_hypernym_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 100, True, "_hypernym"),
    # Experiment("_instance_hypernym_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 100, True, "_instance_hypernym"),
    # Experiment("_member_meronym_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 100, True, "_member_meronym"),
    # Experiment("_synset_domain_topic_of_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 100, True, "_synset_domain_topic_of"),

    # RL2 doing these.
    # Experiment("_also_see_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 200, True, "_also_see"),
    # Experiment("_has_part_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 100, True, "_has_part"),
    # Experiment("_member_of_domain_usage_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 200, True, "_member_of_domain_usage"),
    # Experiment("_member_of_domain_region_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 200, True, "_member_of_domain_region"),
    # Experiment("_verb_group_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 150, True, "_verb_group"),
    # Experiment("_similar_to_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 500, True, "_similar_to"),
    # Experiment("_derivationally_related_form_WN18_generic_PPO_simple_embedding_100", "WN18RR", ["TransE_l2"], 100, True, "_derivationally_related_form"),
    # Experiment("WN18_generic_PPO_simple_embedding_50", "WN18RR", ["TransE_l2"], 50),

    # Experiment("music_artist_genre_NELL_PPO_simple_distance_150", "NELL-995", ["TransE_l2"], 150, True, relation = "concept:musicartistgenre"),
    # Experiment("has_color_NELL_PPO_simple_distance_250", "NELL-995", ["TransE_l2"], 250, True, relation = "concept:thinghascolor"),
]

TESTS = [

    Test("countries_reward_shaping", "countries_reward_shaping", ["TransE_l2"], 5000),

    # Test("WN18_generic_PPO_simple_embedding_50", "WN18_generic_PPO_simple_embedding_50", ["TransE_l2"], 5000),
    # Test("film_genre_FB_PPO_distance_22", "film_genre_FB_PPO_distance_22", ["TransE_l2"], 5000),
    
    #WN18 TESTS
    # Test("also-see-wn18-PPO-simple-emb-100", "_also_see_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("derivationally-related-from-wn18-PPO-simple-emb-100", "_derivationally_related_form_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("has-part-wn18-PPO-simple-emb-100", "_has_part_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("hypernym-wn18-PPO-simple-emb-100", "_hypernym_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("instance-hypernim-wn18-PPO-simple-emb-100", "_instance_hypernym_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("member-meronym-wn18-PPO-simple-emb-100", "_member_meronym_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("member-domain-region-wn18-PPO-simple-emb-100", "_member_of_domain_region_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("member-domain-usage-wn18-PPO-simple-emb-100", "_member_of_domain_usage_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("similar-to-wn18-PPO-simple-emb-100", "_similar_to_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("synset-domain-topic-wn18-PPO-simple-emb-100", "_synset_domain_topic_of_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),
    # Test("verb-group-wn18-PPO-simple-emb-100", "_verb_group_WN18_generic_PPO_simple_embedding_100", ["TransE_l2"], 5000),

    # # FREEBASE TESTS
    # Test("film-genre-FB-base-simple-distance-22-test", "film_genre_FB_Base_simple_distance_22", ["TransE_l2"], 2500),
    # Test("film-genre-FB-base-simple-embedding-22-test", "film_genre_FB_Base_simple_embedding_22", ["TransE_l2"], 2500),
    # Test("film-genre-FB-PPO-simple-embedding-22-test", "film_genre_FB_PPO_embedding_22", ["TransE_l2"], 2500),

    # # NELL-TESTS
    # Test("has-color-nell-base-dist-250", "has_color_NELL_BASE_simple_distance_250", ["TransE_l2"], 5000),
    # Test("has-color-nell-base-emb-250", "has_color_NELL_BASE_simple_embedding_250", ["TransE_l2"], 5000),
    # Test("has-color-nell-ppo-emb-250", "has_color_NELL_PPO_embedding_250", ["TransE_l2"], 5000),
    # Test("has-color-nell-ppo-dist-250", "has_color_NELL_PPO_simple_distance_250", ["TransE_l2"], 5000),
    
    # Test("is-taller-nell-base-dist-250", "is_taller_NELL_BASE_simple_distance_250", ["TransE_l2"], 5000),
    # Test("is-taller-nell-base-emb-250", "is_taller_NELL_BASE_simple_embedding_250", ["TransE_l2"], 5000),
    # Test("is-taller-nell-ppo-emb-250", "is_taller_NELL_PPO_simple_embedding_250", ["TransE_l2"], 5000),
    # Test("is-taller-nell-ppo-dist-250", "is_taller_NELL_PPO_simple_distance_250", ["TransE_l2"], 5000),

    # Test("music_artist_genre-nell-base-dist-150", "music_artist_genre_NELL_BASE_simple_distance_150", ["TransE_l2"], 5000),
    # Test("music_artist_genre-nell-base-emb-150", "music_artist_genre_NELL_BASE_simple_embedding_150", ["TransE_l2"], 5000),
    # Test("music_artist_genre-nell-ppo-emb-150", "music_artist_genre_NELL_PPO_simple_embedding_150", ["TransE_l2"], 5000),
    # Test("music_artist_genre-nell-ppo-dist-150", "music_artist_genre_NELL_PPO_simple_distance_150", ["TransE_l2"], 5000),

    # Test("another-test", "Countries-distancerewonly-250laps-PPO", ["TransE_l2"], 10),
    # Test("good_test", "countiesall", ["TransE_l2", "DistMult", "ComplEx"], 10),
]

TESTS = [t for t in TESTS if not t.to_delete]

def get_config(train: bool, only_config: bool = False):
    """
    fetchs the configuration and either the experimentation list or the test list.

    :param train: whether to get the training or the tests.
    :param only_config: if true only the configuration is returned.

    :returns:
    config -> the configuration dictionary \n
    EXPERIMENTS (optional) -> the experiment list. \n
    TESTS (optional) -> the tests list. \n
    """
    if(only_config):
        return config
    else:
        if train:
            return config, EXPERIMENTS
        else:
            return config, TESTS