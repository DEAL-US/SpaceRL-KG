from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from config import get_config  # Replace "my_module" here with the module name.
sys.path.pop(0)

import pickle
import pprint
from pathlib import Path
import numpy as np
from datetime import datetime
from keras import Model
from keras.models import load_model
import os
import shutil

class DataManager(object):
    def __init__(self, is_experiment=True, experiment_name = "default_experiment_name", respath=""):
        self.data_path = Path(__file__).parent.resolve()
        self.rel_data_path = "data"
        self.datasets_path = str(Path(self.data_path).parent.parent.absolute()) + "/datasets"
        self.caches_path = str(Path(self.data_path))+"/caches"
        self.agents_path = f"{str(Path(self.data_path))}/agents/{experiment_name}"
        self.test_result_path = f"{respath}/{experiment_name}"

        if(not os.path.isdir(self.agents_path) and is_experiment):
            os.mkdir(self.agents_path)
            self.copy_config()

        if(not os.path.isdir(self.test_result_path) and not is_experiment):
            os.mkdir(self.test_result_path)
        

    def get_dataset(self, dataset, embedding_name):
        '''
        Returns the triples that make up a required dataset, the embedding representation of the components \n
        of the dataset, the length of those representations and initializes the log file for that particular dataset \n
        Parameters: \n
            dataset (str) => name of the folder containing the dataset \n
            embedding_name (str) => the name of the embedding type to use. \n
        Returns: \n
            (triples, relations_emb, entities_emb, embedding_len) \n
            triples (list) => the triples in format => [(e1, r , e2), ..., (e1n, r , e2n)] \n
            relations_emb (dict), entities_emb(dict) => the dataset {embeddings} in the following format: \n
            {e1:[0.34,...,0.565], [...], en:[...], [...] ,r1:[...], rn:[...]} \n
            embedding_len (int) => the length of the embedding space \n
        '''

        self.logs_file_path = f"{self.data_path}/logs/{dataset}_{embedding_name}.log"
        selected_file = "graph.txt"
        dataset_root_dir = f"{self.datasets_path}/{dataset}" 
        
        # Extracting triples from file
        triples = []
        with open(f"{dataset_root_dir}/{selected_file}",'r') as datfile:
            for l in datfile.readlines():
                triple = l.replace("\n","").split("\t")
                triples.append(triple)

        # extracting entites and relation embeddings from file.
        with open(f"{dataset_root_dir}/embeddings/{embedding_name}_relations.pkl", 'rb') as emb_rel:
            relations_emb = pickle.load(emb_rel)
            
        with open(f"{dataset_root_dir}/embeddings/{embedding_name}_entities.pkl", 'rb') as emb_ent:
            entities_emb = pickle.load(emb_ent)

        ent_iter = iter(entities_emb.items())
        ent_len = len(list(next(ent_iter))[1])

        # checking if logfile exists, if not, create it:
        if(not Path.is_file(Path(self.logs_file_path))):
            with open(self.logs_file_path, "w") as f:
                f.write("")

        # we mark as active since we've asked for a dataset somewhere.
        self.active_dataset = True

        return (triples, relations_emb, entities_emb, ent_len)

    def save_agent_model(self, name, model :Model):
        '''
        Save the keras model values into a .npy file [array] in the agents directory.\n
        Parameters: \n
            model_layers(numpy array): [[0.2323,0.6788,...],[0.5675,0.978,...],..] \n
            name (str): the name to save the agent. \n
        '''
        if len(model) == 1:
            saved_agent_dir = f"{self.agents_path}/{name}.h5"
            model[0].save(saved_agent_dir, save_format="h5")
        else:
            folder_agent_dir = f"{self.agents_path}/{name}"
            if(not os.path.isdir(folder_agent_dir)):
                os.mkdir(folder_agent_dir)
            model[0].save(f"{folder_agent_dir}/actor.h5", save_format="h5")
            model[1].save(f"{folder_agent_dir}/critic.h5", save_format="h5")

    def restore_saved_agent(self, name):
        '''
        Returns the agent model 
        Parameters:
            name(str): the name of the agent.
        Returns:
            models(numpy array): [[0.2323,0.6788,...],[0.5675,0.978,...],..]
        '''
        print("restoring model...")
        saved_agent_dir = f"{self.agents_path}/{name}.h5"
        return load_model(saved_agent_dir) 
    
    def restore_saved_agent_PPO(self, name):
        '''
        Returns the agent model 
        Parameters:
            name(str): the name of the agent.
        Returns:
            models(numpy array): [[0.2323,0.6788,...],[0.5675,0.978,...],..]
        '''
        print("restoring model...")
        saved_agent_dir = f"{self.agents_path}/{name}"
        actor = load_model(f"{saved_agent_dir}/actor.h5")
        critic = load_model(f"{saved_agent_dir}/actor.h5") 

        return actor, critic

    def write_log(self, content):
        '''
        writes the content in the corresponding logfile, the logfile is automatically swapped when asking for a dataset.
        '''
        if(self.active_dataset):
            current_date_formated = datetime.strftime(datetime.now(), "%d-%m-%Y %H:%M:%S")
            with open(self.logs_file_path, "a") as f:
                f.write(f"[{current_date_formated}] - {content}\n")
        else:
            print("no dataset is loaded, cannot log, call get_dataset before logging.")

    def get_cache_for_dataset(self, dataset):
        '''
        gets the cache for a given dataset, raises FileNotFoundException if not avaliable.
        '''
        filepath = self.caches_path+"/"+dataset+".pkl"
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def save_cache_for_dataset(self, dataset :str, cache):
        '''
        saves the cache into a pickle file for the given dataset.
        '''
        filepath = self.caches_path+"/"+dataset+".pkl"
        with open(filepath, "wb") as f:
            pickle.dump(cache, f)

    def saveall(self, dataset, cache, agent_name, model):
        print("saving data...")
        self.save_cache_for_dataset(dataset, cache)
        self.save_agent_model(agent_name, model)

    def debug_save(self, name):
        '''
        saves the current keras model as well as 
        the input that triggered the NN error.
        '''
        current_date_formated = datetime.strftime(datetime.now(), "%d-%m-%Y_%H:%M:%S")
        saved_agent_dir = f"{self.agents_path}/{name}.h5"
        crash_agent_dir = f"{self.data_path}/debug/{current_date_formated}_crash_report/model.h5"
        os.mkdir(f"{self.data_path}/debug/{current_date_formated}_crash_report/")
        shutil.copyfile(saved_agent_dir, crash_agent_dir)

        saved_array_dir = f"{self.data_path}/debug/{current_date_formated}_crash_report/input.npy"
        np.save(saved_array_dir, self.latest_inputs)

    def debug_load(self, folder_index, print_layers):
        debug_dir = f"{self.rel_data_path}/debug/"
        dirs = os.listdir(debug_dir)
        if(folder_index is None):
            crashdir = dirs[len(dirs)-1]
        else:
            crashdir = dirs[folder_index]
        input_arr = np.load(f"{debug_dir}{crashdir}/input.npy")
        model = load_model(f"{debug_dir}{crashdir}/model.h5")

        if(print_layers):
            for l in model.layers:
                print(f"layer weights for {l.name} are:\n")
                weights = l.get_weights()
                print(weights)

        print(f"input that triggered this was: {input_arr}")

    def update_lastest_input(self, latest_input):
        self.latest_inputs = latest_input

    def copy_config(self):
        config, _ = get_config(False)
        # make a copy of the dict and remove the unwanted config lines.
        c = dict(config)
        del c['available_cores']
        del c['gpu_acceleration']
        del c['verbose']
        del c['log_results']
        del c['debug']
        del c['print_layers']
        del c['restore_agent']
        del c['regenerate_embeddings']
        del c['normalize_embeddings']
        del c['random_seed']
        del c['seed']
        res = ""
        for i in c.items():
            res += f"{str(i[0])}: {str(i[1])}\n"

        with open(f"{self.agents_path}/config_used.txt", "w") as f:
            f.write(res)