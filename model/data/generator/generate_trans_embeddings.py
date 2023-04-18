import numpy as np
import pathlib
import os
import pickle
import shutil
import pandas as pd
import torch


def generate_embedding(dataset: str, models = [], use_gpu :bool = True, regenerate_existing:bool = False,
normalize:bool = False,  add_inverse_path:bool = True, fast_mode:bool = True, available_cores:int = 1):
    """
    Generates any number of embeddings for the specified dataset in their relevant folders.

    :param dataset: which dataset to generate embeddings for.  
    :param models: which embeddings to generate for the specified dataset, options are "TransE_l2", "DistMult", "ComplEx", "TransR". If left empty calculates all.
    :param use_gpu: whether to use gpu for embedding calculation.
    :param regenerate_existing: if true recalculates the embeddings for selected options, skip them otherwise.
    :param normalize: wether to normalize the embedding dimensional space.
    :param add_inverse_path: wether to add the inverse edge to all relations so if (e1, r, e2) is present, so will be (e2, ¬r, e1)
    :param fast_mode: dictates if we use a deep embedding training or we use a fast one (number of iterations 6000 vs 24000)
    :param available_cores: the number of cpu cores to use for calculation, useful if we are not using the gpu. 

    :returns: None
    :raises FileNotFoundError: if the specified files do not exist.
    """

    local_dir = pathlib.Path(__file__).parent.resolve()
    dataset_dir = str(pathlib.Path(local_dir).parent.parent.parent.absolute()) + "/datasets"
    datafolder = f"{dataset_dir}/{dataset}"
    print(f"datafolder is: {datafolder}")

    if(len(models) == 0):
        models =  ["TransE_l2","DistMult", "ComplEx", "TransR"]

    print(f"generating embeddings for dataset {dataset} and models {models}")

    for model in models:
        # reference: https://aws-dglke.readthedocs.io/en/latest/train.html 
        command = f"DGLBACKEND=pytorch dglke_train --model_name {model} \
--data_path \"{local_dir}/raw_data\" --save_path \"{datafolder}/embeddings/\" \
--dataset {dataset} --data_files {dataset}-raw.txt \
--format raw_udd_hrt --delimiter , --batch_size 1000 --neg_sample_size 200 --hidden_dim 200 \
--log_interval 100 --batch_size_eval 16 -adv --regularization_coef 1.00E-09 --gamma 19.9 --lr 0.25"

        if(fast_mode):
            command += " --max_step 6000"
        else:
            command += " --max_step 24000"

        #GPU vs CPU
        if(use_gpu & torch.cuda.is_available()):
            command += " --gpu 0"
        else:
            command += f" --num_thread {available_cores}" #--num_proc {available_cores}

        # Structural checks:
        
        if(not os.path.exists(datafolder)):
            raise FileNotFoundError(f"the selected dataset folder \"{datafolder}\" does not exist")
        
        graphfile = f"{datafolder}/graph.txt"
        if(not os.path.isfile(graphfile)):
            raise FileNotFoundError("we couldn't find the graph.txt for the specified dataset, check the file name.")

        emb_folder = f"{datafolder}/embeddings"
        if(not os.path.exists(emb_folder)):
            os.mkdir(emb_folder)

        # remove dir if regen or remove empty in no_regen.
        dgl_ke_folder = f"{datafolder}/embeddings/{model}_{dataset}_0"
        
        if(regenerate_existing):
            shutil.rmtree(dgl_ke_folder, ignore_errors=True)
        else:
            try:
                os.rmdir(dgl_ke_folder) # removes if empty.
            except Exception as e:
                pass
                # print("couldn't remove folder, its probable not empty: " + str(e))

        entity_file = f"{dgl_ke_folder}/{dataset}_{model}_entity.npy"
        relation_file = f"{dgl_ke_folder}/{dataset}_{model}_relation.npy"

        # if embedding files not exist generate them.
        if(not os.path.isfile(entity_file) and not os.path.isfile(relation_file)):
            # if the embeddings are not present, generate and process them
            generate_raw(dataset, local_dir, dataset_dir, add_inverse_path)

            print(f"running command {command}")
            os.system(command)
       
            shutil.copy(f"{local_dir}/raw_data/entities.tsv", f"{datafolder}/entities.tsv")
            shutil.copy(f"{local_dir}/raw_data/relations.tsv", f"{datafolder}/relations.tsv")
        else:
            print(f"Selected embedding {model} is already generated for {dataset} dataset, if you want to regenerate use the regenerate boolean option")

        process_embeddings(entity_file, relation_file,
        dataset_dir, dataset, model, regenerate_existing, normalize)

def process_embeddings(entity_file: str, relation_file:str, dataset_dir:str, dataset:str, model:str, regenerate: bool, normalize:bool):
    """
    Normalizes and processes an embedding, saves them to the apropriate file.

    :param entity_file: The path to the entities.tsv file for the desired dataset
    :param relation_file: The path to the relations.tsv file for the desired dataset
    :param dataset_dir: The path to the datasets directory
    :param dataset: The name of the desired dataset
    :param model: the embeddings to normalize for that dataset, options are "TransE_l2", "DistMult", "ComplEx", "TransR". If left empty calculates all.
    :param regenerate: wether to regenerate the embeddings 
    :param normalize: if regenerate is active normalize the embeddings.

    :returns: None
    """

    base_folder = f"{dataset_dir}/{dataset}"

    # Load files
    entities_f = open(f"{base_folder}/entities.tsv")
    relations_f = open(f"{base_folder}/relations.tsv")

    entities = entities_f.readlines()
    relations = relations_f.readlines()

    entities_f.close()
    relations_f.close()

    entity_embedding_array = np.load(entity_file)
    relation_embedding_array = np.load(relation_file)

    num_entity = len(entity_embedding_array)
    num_rel = len(relation_embedding_array)

    # Mean normalization centers the variables in the N-spatial axis
    # where N is the embedding size.
    df_entity = pd.DataFrame(entity_embedding_array) # one embedding per row
    means = df_entity.mean()
    for i in range(num_entity):
        df_entity.iloc[i] = df_entity.iloc[i] - means

    # Variance normalization expands or contracts each spatial dimension
    # to fit in a N-dimensional space homogeneously.
    variances = df_entity.var()
    for i in range(num_entity):
        df_entity.iloc[i] = df_entity.iloc[i]/variances
    
    # same operations to relation df
    df_relation = pd.DataFrame(relation_embedding_array)
    means = df_relation.mean()
    for i in range(num_rel):
        df_relation.iloc[i] = df_relation.iloc[i] - means

    variances = df_relation.var()
    for i in range(num_rel):
        df_relation.iloc[i] = df_relation.iloc[i]/variances

    entity_embedding_array = df_entity.values
    relation_embedding_array = df_relation.values

    # Assign verbatim representation with embedding.
    entities_dict = {}
    for i, l in enumerate(entities):
        ent = l.split(",")[1].replace("\"", "").replace("\n","") #clean tsv
        entities_dict[ent] = entity_embedding_array[i]

    print(str(len(entities_dict)) + " total entities")

    relations_dict = {}
    for i, l in enumerate(relations):
        rel = l.split(",")[1].replace("\"", "").replace("\n","") #clean tsv
        relations_dict[rel] = relation_embedding_array[i]

    print(str(len(relations_dict)) + " total relations")

    f = open(f"{base_folder}/embeddings/{model}_entities.pkl","wb")
    pickle.dump(entities_dict, f)
    f.close()

    f = open(f"{base_folder}/embeddings/{model}_relations.pkl","wb")
    pickle.dump(relations_dict, f)
    f.close()

def generate_raw(dataset: str, generator_dir: str, dataset_dir: str, add_inverse: bool):
    """
    process the graph.txt and generates some raw files for DGL to use.

    :param dataset: which dataset to use
    :param generator_dir: path to the generator directory
    :param dataset_dir: path to the datasets directory.
    :param add_inverse: wether to add the inverse paths to the dataset.

    :returns: None
    """
    raw_file = "graph.txt"

    file_to_read = open(f"{dataset_dir}/{dataset}/{raw_file}","r")
    file_to_write = open(f"{generator_dir}/raw_data/{dataset}-raw.txt","w")

    content = ""
    for l in file_to_read.readlines():
        linecontent = l.strip().replace("\n","").split("\t")
        content += f"{linecontent[0]},{linecontent[1]},{linecontent[2]}\n"
        if(add_inverse):
            content += f"{linecontent[2]},¬{linecontent[1]},{linecontent[0]}\n" # add the inverse of the relation.

    file_to_write.write(content)
    file_to_write.close()
    file_to_read.close()

# YOU CAN RUN THIS DIRECTLY TO GENERATE ALL THE POSSIBLE EMBEDDINGS AND AVOID WAITING IN THE FUTURE FOR ANY NEW ONES.
if __name__ == "__main__":
    use_gpu, regenerate_existing , fast_mode = False, False, True

    # add or remove the datasets you want to generate embeddings for.
    # datasets = ["COUNTRIES", "FB15K-237", "KINSHIP", "NELL-995", "UMLS", "WN18RR"]
    datasets = ["COUNTRIES", "FB15K-237", "KINSHIP", "NELL-995", "UMLS", "WN18RR"]

    for d in datasets:
        generate_embedding(d, [], use_gpu, regenerate_existing, normalize=True,
        add_inverse_path=True, fast_mode=fast_mode)