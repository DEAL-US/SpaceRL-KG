# Generating word vectorial representation using word2vec.
  
# importing all necessary modules
from sentence_transformers import SentenceTransformer
import pathlib
import os
import pickle
from tqdm import tqdm

current_dir = pathlib.Path(__file__).parent.resolve()

transform_model = SentenceTransformer("all-distilroberta-v1", device="cpu")

#  Reads entities.txt files
for route in dataset_folders:
    data_to_save = {}
    with open(curr_file_path+"/datasets/"+ route+"/relations.txt", "r") as data:
        for line in tqdm(data.readlines()):
            word_plain = line.split('\t')[0].strip()
            embedding = transform_model.encode(word_plain)
            data_to_save[word_plain] = embedding
        
        pkfile = open(curr_file_path+"/datasets/"+ route+'/relations_embeddings.pkl','wb')
        pickle.dump(data_to_save, pkfile)
        pkfile.close()