import numpy as np
import pathlib
import os
import pickle
import shutil
import pandas as pd
import torch

# datasets = ["COUNTRIES", "FB15K-237", "KINSHIP", "MOVIES", "UMLS", "WN18RR"]
# model_names =  ["TransE_l2","DistMult", "ComplEx", "RotatE"] #["TransR", "RESCAL"] # need a more powerful machine to run these.

def generate_embedding(dataset, models = [], use_gpu = True, regenerate_existing = False,
normalize = False,  add_inverse_path = True, fast_mode = True):
    '''
    Generates several positional embeddings namely: "TransE_l2", "DistMult", "ComplEx", "RotatE", "TransR" & "RESCAL" \n
    you can specify any number of them with: models = ["TransE_l2","DistMult"] \n
    we provide the following dataset raws as folders:  \n
    "COUNTRIES", "FB15K-237", "KINSHIP", "MOVIES", "NELL-995", "UMLS", "WN18RR" \n
    they must contain a "graph.txt" file that specifies the graph structure. \n
    Parameters: \n
    dataset (str): the name of the folder containing the graph.txt of the dataset. \n
    models (iter): the embedding models you want to generate, if empty generates all 6.\n
    use_gpu (bool): wether you want to train with gpu or just with cpu \n
    regenerate_existing (bool): wether to re-create embeddings for an existing model, use it in case the main dataset changed \n
    add_inverse_path (bool): wether to add the inverse edge to all relations so if (e1, r, e2) is present, so will be (e2, ¬r, e1) \n
    fast_mode (bool): dictates if we use a deep embedding training or we use a fast one (number of iterations 6000 or 24000)
    '''
    local_dir = pathlib.Path(__file__).parent.resolve()
    dataset_dir = str(pathlib.Path(local_dir).parent.parent.parent.absolute()) + "/datasets"
    datafolder = f"{dataset_dir}/{dataset}"
    print(f"datafolder is: {datafolder}")

    if(len(models) == 0):
        models =  ["TransE_l2","DistMult", "ComplEx", "RotatE", "TransR", "RESCAL"]

    print(f"generating embeddings for dataset {dataset} and models {models}")

    for model in models:
        # reference: https://aws-dglke.readthedocs.io/en/latest/train.html 
        command = f"DGLBACKEND=pytorch dglke_train --model_name {model} \
--data_path \"{local_dir}/raw_data\" --save_path \"{datafolder}/embeddings/\" \
--dataset {dataset} --data_files {dataset}-raw.txt \
--format raw_udd_hrt --delimiter , --batch_size 1000 --neg_sample_size 200 --hidden_dim 200 \
--log_interval 100 --batch_size_eval 16 -adv \
--regularization_coef 1.00E-09 --gamma 19.9 --lr 0.25"

        if(fast_mode):
            command += " --max_step 6000"
        else:
            command += " --max_step 24000"

        #GPU vs CPU
        if(use_gpu & torch.cuda.is_available()):
            command += " --gpu 0"
        else:
            command += " --num_thread 1 --num_proc 8"

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
        dataset_dir, dataset, model, normalize, regenerate_existing)

def process_embeddings(entity_file, relation_file, dataset_dir, dataset, model, normalize, regenerate):
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

def generate_raw(dataset, generator_dir, dataset_dir, add_inverse):
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

# generate_embedding("COUNTRIES", use_gpu = True, regenerate_existing = True, add_inverse_path = True, fast_mode = True)
# generate_embedding("FB15K-237", use_gpu = True, regenerate_existing = True, add_inverse_path = True, fast_mode = True)
# generate_embedding("KINSHIP", use_gpu = True, regenerate_existing = True, add_inverse_path = True, fast_mode = True)
# generate_embedding("NELL-995", use_gpu = True, regenerate_existing = True, add_inverse_path = True, fast_mode = True)
# generate_embedding("UMLS", use_gpu = True, regenerate_existing = True, add_inverse_path = True, fast_mode = True)
# generate_embedding("WN18RR", use_gpu = True, regenerate_existing = True, add_inverse_path = True, fast_mode = True)