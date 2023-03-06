from pathlib import Path
from tqdm import tqdm
import os
from pprint import pprint as pp


current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent.resolve()
dataset_path = Path(f"{parent_dir}/datasets").resolve()

datasets = [name for name in os.listdir(dataset_path) if os.path.isdir(f"{dataset_path}/{name}")]

print(datasets)


def main(exclude):
    res = list()

    for dataset in datasets:
        dts_res = dict()

        if dataset in exclude:
            continue

        file = Path(f"{dataset_path}/{dataset}/graph.txt").resolve()
        f = open(file)

        for l in tqdm(f.readlines()):
            _, r, _ = l.split("\t")
            r = r.strip()
            try:
                dts_res[r] += 1
            except:
                dts_res[r] = 1

        f.close()

        res.append(dts_res)
    
    for r in res:
        pp(r)


if __name__ == "__main__":
    exclude = ["KINSHIP", "UMLS", "COUNTRIES", "WN18RR"] #, "NELL-995", "FB15K-237"
    max_depth = 3
    main(exclude)