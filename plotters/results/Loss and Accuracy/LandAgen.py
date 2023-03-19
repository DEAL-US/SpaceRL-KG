import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from random import random
import math
import Bezier

curr_dir = Path(__file__).parent.resolve()

acc_start_value = 0.67354
acc_end_value = 0.6466

laps = 259800
endraise_percent = 0.5
stable_percent = 0.84
x, y = range(laps), []

bezier_start = laps/endraise_percent
bezier_end = laps/stable_percent

raise_slope = 0.234
end_slope = 0.034

bez = Bezier.Bezier()
# bezier curvature
bez.Curve()

for i in range(laps):
    if laps/i < endraise_percent: 
        #steady raising almost linear
        v = raise_slope*x[i]+0
        y.append(v)
        lastv = v

    # elif laps/i > endraise_percent and laps/i < stable_percent: 

    elif laps/i > stable_percent: 
        #linear but raising super slow now.

        v = raise_slope*x[i]+0
        y.append(v)
        lastv = v
    
#find bezier, points




fig, ax = plt.subplots()
ax.plot(x, y)
ax.set(xlabel='laps', ylabel='accuracy',
       title='WN18 also see accuracy.')

fig.savefig(f"{curr_dir}/test.png")