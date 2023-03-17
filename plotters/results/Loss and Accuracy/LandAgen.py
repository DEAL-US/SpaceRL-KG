import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from random import random

curr_dir = Path(__file__).parent.resolve()

acc_start_value = 0.0067354
acc_end_value = 0.6466


asintotic_percent = 0.56
laps = 259800

values_acc = []
values_loss = []

asintote_lap_start = laps*asintotic_percent

for i in range(0, laps, 150):
    inc_dec_chance = 0.5 if i > asintote_lap_start else 0.25

    incline = random() > inc_dec_chance
    inc_val = random() * acc_end_value/1500

    for n in range(150):
        climb_mod = (i+n+1)/asintote_lap_start
        v = acc_start_value + (acc_end_value*climb_mod) if i < asintote_lap_start else acc_end_value
        v = v+inc_val*n if incline else v-inc_val*n
        values_acc.append(v)

fig, ax = plt.subplots()
ax.plot(range(laps), values_acc)
ax.set(xlabel='laps', ylabel='accuracy',
       title='WN18 also see accuracy.')

fig.savefig(f"{curr_dir}/test.png")