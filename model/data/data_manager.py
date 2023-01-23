from inspect import getsourcefile
import os.path as path, sys
# add the parent directory to path so you can import config into data manager. 
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from config import get_config
sys.path.pop(0)

import pickle
from pathlib import Path
import numpy as np
from datetime import datetime
from keras import Model
from keras.models import load_model
import os
import shutil

class DataManager(object):
    """
    The data manager class is tasked with organizing and saving the data during testing, experimentation and embedding generation
    Each instance of data manager covers a single test or experiment instance.

    :param is_experiment: wether this instance is for an experiment or a test.
    :param name: the name of the experiment or test to manage.
    :param respath: the path to the folder to save the test result. If it does not exist its created.

    :returns: None
    """
    def __init__(self, is_experiment:bool=True, name:str = "default_name", respath:str=""):
        self.data_path = Path(__file__).parent.resolve()

        self.datasets_path = f"{self.data_path.parent.parent.absolute().resolve()}/datasets"
        self.caches_path = f"{self.data_path}/caches"
        self.base_agent_path = f"{self.data_path}/agents"
        self.agents_path = f"{self.base_agent_path}/{name}"

        self.test_result_path = respath
        self.is_experiment = is_experiment
        self.name = name

        self.run_integrity_checks()

        if(not os.path.isdir(self.agents_path) and is_experiment):
            os.mkdir(self.agents_path)
            self.copy_config()

        if(not os.path.isdir(self.test_result_path) and not is_experiment):
            os.mkdir(self.test_result_path)

    #####################
    # DATASETS & CACHES #
    #####################

    def get_dataset(self, dataset:str, embedding_name:str):
        """
        Returns the triples that make up a required dataset, the embedding representation of the component of the dataset,\
        the length of those representations and initializes the log file for that particular dataset

        :param dataset: name of the folder containing the dataset
        :param embedding_name: the name of the embedding type to use. 

        :returns:  
        (triples, relations_emb, entities_emb, embedding_len) \n
        triples (list) => the triples in format => [(e1, r , e2), ..., (e1n, r , e2n)]  \n
        relations_emb (dict) & entities_emb(dict) => the dataset {embeddings} separated in entities and relations in the following format:
        {e1:[0.34,...,0.565], [...], en:[...], [...] ,r1:[...], rn:[...]} \n
        embedding_len (int) => the length of the embedding space  \n

        """

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

        return triples, relations_emb, entities_emb, ent_len

    def get_cache_for_dataset(self, dataset:str):
        """
        gets the cache for a given dataset

        :param dataset: the name of the dataset.

        :returns: The reward cache for the specified dataset
        :raises FileNotFoundException: if not avaliable
        """
        filepath = f"{self.caches_path}/{dataset}.pkl"
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def save_cache_for_dataset(self, dataset:str, cache:dict):
        """
        Saves the given cache for the specified dataset.

        :param dataset: the name of the dataset
        :cache: the cache to save.

        :returns: None
        """

        filepath = self.caches_path+"/"+dataset+".pkl"
        with open(filepath, "wb") as f:
            pickle.dump(cache, f)

    ##################
    # AGENT & MODELS #
    ##################

    def save_agent_model(self, name:str, model :Model):
        """
        Save the keras model values into a <name>.h5 or a directory <name> with actor.h5 and critic.h5 if PPO.

        :param name: the name of the agent to save.
        :param model: the keras model to save
    
        :returns: None
        """
        if len(model) == 1:
            saved_agent_dir = f"{self.agents_path}/{name}.h5"
            model[0].save(saved_agent_dir, save_format="h5")
        else:
            folder_agent_dir = f"{self.agents_path}/{name}"
            if(not os.path.isdir(folder_agent_dir)):
                os.mkdir(folder_agent_dir)
            model[0].save(f"{folder_agent_dir}/actor.h5", save_format="h5")
            model[1].save(f"{folder_agent_dir}/critic.h5", save_format="h5")

    def restore_saved_agent(self, name:str):
        """
        returns the saved agent as a keras model to be used

        :param name: the name of the agent to restore. 

        :returns: a keras model of the agent.
        :raises FileNotFoundException: if the specified agent does not exist 
        """

        print("restoring model...")
        saved_agent_dir = f"{self.agents_path}/{name}.h5"
        return load_model(saved_agent_dir) 
    
    def restore_saved_agent_PPO(self, name:str):
        """
        returns the saved agent as keras models to be used 

        :param name: the name of the agent to restore

        :returns: actor, critic => the acotor and critic keras models.  
        :raises FileNotFoundException: if the specified agent does not exist 
        """

        print("restoring model...")
        saved_agent_dir = f"{self.agents_path}/{name}"
        actor = load_model(f"{saved_agent_dir}/actor.h5")
        critic = load_model(f"{saved_agent_dir}/actor.h5") 

        return actor, critic

    def saveall(self, dataset:str, cache:dict, agent_name:str, model:Model):
        """
        Performs every saving operation linked to a dataset
        
        :param dataset: the name of the dataset
        :param cache: the cache to save.
        :param agent_name: the name of the agent
        :param model: the keras model to save.

        :returns: None
        """
        print("saving data...")
        self.save_cache_for_dataset(dataset, cache)
        self.save_agent_model(agent_name, model)

    ################
    # DEBUG & LOGS #
    ################
    
    def write_log(self, content:str):
        """
        writes the content in the corresponding logfile, the logfile is automatically swapped when calling the get_dataset function

        :param content: what to add to the log 

        :returns: None
        """
    
        if(self.active_dataset):
            current_date_formated = datetime.strftime(datetime.now(), "%d-%m-%Y %H:%M:%S")
            with open(self.logs_file_path, "a") as f:
                f.write(f"[{current_date_formated}] - {content}\n")
        else:
            print("no dataset is loaded, cannot log, call get_dataset before logging.")

    def debug_save(self, name:str):
        """
        If debug mode is active save the crash information on agent crash
        Saves the current keras model as well as the input that triggered the NN error.

        :param name: agent name 

        :returns: None
        """
        current_date_formated = datetime.strftime(datetime.now(), "%d-%m-%Y_%H:%M:%S")
        saved_agent_dir = f"{self.agents_path}/{name}.h5"
        crash_agent_dir = f"{self.data_path}/debug/{current_date_formated}_crash_report/model.h5"
        os.mkdir(f"{self.data_path}/debug/{current_date_formated}_crash_report/")
        shutil.copyfile(saved_agent_dir, crash_agent_dir)

        saved_array_dir = f"{self.data_path}/debug/{current_date_formated}_crash_report/input.npy"
        np.save(saved_array_dir, self.latest_inputs)

    def debug_load(self, folder_index:int, print_layers:bool):
        """
        loads the crash information and prints it.

        :param folder_index: which folder to load by its index in the debug directory.
        :print_layers: wether to print the NN intermediate layers before the crash. 

        :returns: None
        """
        debug_dir = "data/debug/"
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

    #################
    # MISCELLANEOUS #
    #################

    def update_lastest_input(self, latest_input):
        """
        updates latest input.

        :param latest_input: the value of the input.

        :returns: None
        """
        self.latest_inputs = latest_input

    def copy_config(self):
        """
        copies the current config information into the experiment folder

        :returns: None
        """
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

    def run_integrity_checks(self):
        """
        checks for abnormal states in the folder structure and corrects them. 

        :returns: None
        """
        print("running integrity checks")
        subfolders = [f.name for f in os.scandir(self.datasets_path) if f.is_dir()]
        for s in subfolders:
            try:
                embs_path = f"{self.datasets_path}/{s}/embeddings"
                embs_dir = [f.name for f in os.scandir(embs_path) if f.is_dir()]
                for e in embs_dir:
                    emb_dir = f"{embs_path}/{e}"
                    self.remove_folders(emb_dir, 0) # removes empty folders
            except:
                print(f"No embeddings have been generated for {s}")

        subfolders = [f.name for f in os.scandir(self.base_agent_path) if f.is_dir()]
        for s in subfolders:
            agent_dir = f"{self.base_agent_path}/{s}"
            self.remove_folders(agent_dir, 1) # removes folders with only config_used.txt
        
    def remove_folders(self, path_abs:str, filecount:int):
        """
        helper method to delete incongruent folders

        :param path_abs: path to the folder to check
        :param filecount: file count for the folder to be deleted.

        :returns: None
        """
        files = os.listdir(path_abs)
        if len(files) == filecount:
            print(f"removing path {path_abs}")
            shutil.rmtree(path_abs)

if __name__ == "__main__":
    dm = DataManager()
