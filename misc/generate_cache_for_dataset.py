import networkx as nx
from pathlib import Path
from tqdm import tqdm
import os, pickle


current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent.resolve()
dataset_path = Path(f"{parent_dir}/datasets").resolve()
cache_dir = Path(f"{parent_dir}/model/data/caches").resolve()


def main(datasets, max_depth):
    """
    Generates distance cache for indicated datasets.

    :param datasets: datasets to use
    :param max_depth: cutoff point for distance calculation.
    """
    # ignores these datasets to generate
    for dataset in datasets:
        G = nx.MultiDiGraph()
        file = Path(f"{dataset_path}/{dataset}/graph.txt").resolve()
        f = open(file)

        for l in tqdm(f.readlines()):
            e1, r, e2 = l.split("\t")
            e2 = e2.strip()
            G.add_edge(e1, e2, key=r)
            G.add_edge(e2, e1, key=f"¬{r}")

        f.close()
        
        print(f"Calculating distances for {dataset}, this may take a while...")
        distances = dict(nx.all_pairs_shortest_path_length(G, max_depth))

        cumsum = 0
        for d in tqdm(distances.items()):
            cumsum += len(d[1])

        print(f"mapped {cumsum} distances for {dataset}")

        with open(f"{cache_dir}/{dataset}.pkl", "wb") as f:
            pickle.dump(distances, f)
        

if __name__ == "__main__":
    # Datasets = ["COUNTRIES", "FB15K-237","KINSHIP", "UMLS", "WN18RR", "NELL-995"]
    all_datasets = [name for name in os.listdir(dataset_path) if os.path.isdir(f"{dataset_path}/{name}")] 
    exclude = ["KINSHIP", "COUNTRIES"] 
    datasets = [x for x in all_datasets if x not in exclude]
    
    max_depth = 3
    main(datasets, max_depth)