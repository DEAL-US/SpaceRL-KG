# Generates a mock dataset.
import random
from pathlib import Path

res = ""

nodes = 25
relations = 3

for r in range(relations):
    for n1 in range(nodes):
        for n2 in range(nodes):
            if(random.random()<0.15):
                res += f"node{n1}\trelation{r}\tnode{n2}\n"

f = open(f"{Path(__file__).parent.resolve()}/graph.txt", "w")
f.write(res)
f.close()