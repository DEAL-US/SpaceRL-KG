import networkx as nx
from pathlib import Path
from tqdm import tqdm
import os


current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent.resolve()
dataset_path = Path(f"{parent_dir}/datasets").resolve()

datasets = [name for name in os.listdir(dataset_path) if os.path.isdir(f"{dataset_path}/{name}")]


def main():
    for dataset in datasets:
        print(dataset)
        G = nx.MultiDiGraph()
        file = Path(f"{dataset_path}/{dataset}/graph.txt").resolve()
        f = open(file)

        for l in tqdm(f.readlines()):
            e1, r, e2 = l.split("\t")
            e2 = e2.strip()
            G.add_edge(e1, e2, key=r)
            G.add_edge(e2, e1, key=f"¬{r}")
        
        print(f"Calculating distances for {dataset}")
        distances = dict(nx.all_pairs_shortest_path_length(G, 5))

        for d in tqdm(distances.items()):
            print(d)

        f.close()
        

if __name__ == "__main__":
    main()