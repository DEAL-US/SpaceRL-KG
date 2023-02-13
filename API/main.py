from subprocess import run
import multiprocessing

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import FastAPIError

from pathlib import Path

app = FastAPI()

current_dir = Path(__file__).parent.resolve()

permanent_config = {
    "gpu_acceleration": True, # wether to use GPU(S) to perform fast training & embedding generation.
    "multithreaded_dist_reward": False, 
    "verbose": False, # prints detailed information every episode.
    "debug": False, # offers information about crashes, runs post-mortem
    "print_layers": False, # if debug is active wether to print the layer wheights on crash info.
    "restore_agent": False, # continues the training from where it left off and loads the agent if possible.
    "log_results": False, # Logs the results in the logs folder of episode training.
    "use_episodes": False, 
    "episodes":0, # number of episodes to run.
}

changeable_config = {
    "available_cores": multiprocessing.cpu_count(), #number of cpu cores to use when computing the reward
    "guided_reward": True, # wether to follow a step-based reward or just a reward at the end of the episode.
    "guided_to_compute":["terminal", "embedding"], #"distance","terminal","embedding"
    "regenerate_embeddings":False, # if embedding is found and true re-calculates them.
    "normalize_embeddings":True,
    "use_LSTM": True, # wether to add LSTM layers to the model

    "alpha": 0.9, # [0.8-0.99] previous step network learning rate (for PPO only.) 
    "gamma": 0.99, # [0.90-0.99] decay rate of past observations for backpropagation
    "learning_rate": 1e-3, #[1e-3, 1e-5] neural network learning rate.

    "activation":'leaky_relu', # relu, prelu, leaky_relu, elu, tanh
    "regularizers":['kernel'], #"kernel", "bias", "activity" 
    "algorithm": "PPO", #BASE, PPO
    "reward_type": "simple", # retropropagation, simple
    "action_picking_policy":"probability",# "probability", "max"
    "reward_computation": "one_hot_max", #"max_percent", "one_hot_max", "straight"

    "path_length":5, #the length of path exploration.
    "random_seed":True, # to repeat the same results if false.
    "seed":78534245, # sets the seed to this number.
}


@app.get("/", response_class=PlainTextResponse)
def root():
    text = """
         __             __          __                  __             __ 
        /\ \           /\ \        / /\                /\ \           /\ \ 
       /  \ \         /  \ \      / /  \              /  \ \         /  \ \ 
      / /\ \_\       / /\ \ \    / / /\ \            / /\ \ \       / /\ \ \ 
     / / /\/_/      / / /\ \_\  / / /\ \ \          / / /\ \ \     / / /\ \_\ 
    / / / ______   / / /_/ / / / / /  \ \ \        / / /  \ \_\   / /_/_ \/_/ 
   / / / /\_____\ / / /__\/ / / / /___/ /\ \      / / /    \/_/  / /____/\ 
  / / /  \/____ // / /_____/ / / /_____/ /\ \    / / /          / /\____\/ 
 / / /_____/ / // / /\ \ \  / /_________/\ \ \  / / /________  / / /______ 
/ / /______\/ // / /  \ \ \/ / /_       __\ \_\/ / /_________\/ / /_______\ 
\/___________/ \/_/    \_\/\_\___\     /____/_/\/____________/\/__________/ 

Hi! this is the API endtry point for the GRACE Framework.
try /docs to check all the commands I can do!
"""
    return text

@app.get("/config")
def get_config():
    return changeable_config

@app.post("/config/{param}:{value}")
def set_config(param:str, value):

    # Raise is not good, gives 500 server error. check docs.
    if param not in changeable_config.keys():
        raise FastAPIError(f"{param} not present in config.")
    
    t = type(changeable_config[param])
    if  t != type(value):
        raise FastAPIError(f"{value} must be of type {t}, was {type(value)} instead.")

    changeable_config[param] = value

    return changeable_config


if __name__ == "__main__":
    command = f"uvicorn main:app --reload".split()
    run(command, cwd = current_dir)