import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math

curr_dir = Path(__file__).parent.resolve()

acc_start_value = 0.67354
acc_end_value = 0.6466

# laps = 259800
laps = 2598

endraise_percent = 0.5
stable_percent = 0.84
x = range(laps)

y_start, y_end, y_middle = [],[],[]

bezier_start = laps/endraise_percent
bezier_end = laps/stable_percent

raise_slope = 0.234
end_slope = 0.034

first_bezier_x = int(laps*endraise_percent)
last_bezier_x = int(laps*stable_percent)

# INITIAL
for i in tqdm(range(first_bezier_x)):
        #steady raising almost linear
        v = raise_slope*x[i]+0
        y_start.append(v)

end_y_climb = v

# FINAL
for i in tqdm(range(laps, last_bezier_x-1, -1)):
        v = end_slope*x[i-1]+end_y_climb
        y_end.append(v)

start_y_stable = v
y_end.reverse()

# CURVE
midp = [first_bezier_x, start_y_stable]

P1 = [first_bezier_x, end_y_climb]
P4 = [last_bezier_x, start_y_stable]
P2 = [first_bezier_x, end_y_climb + abs(math.dist(P1, midp))/2]
P3 = [last_bezier_x - abs(math.dist(P4, midp))/2, start_y_stable] 

control_points = [P1,P2,P3,P4]
# minx = min([h[0] for h in control_points])
# miny = min([h[1] for h in control_points])
maxx = max([h[0] for h in control_points])
maxy = max([h[1] for h in control_points])

P1 = P1[0]/maxx, P1[1]/maxy
P2 = P2[0]/maxx, P2[1]/maxy
P3 = P3[0]/maxx, P3[1]/maxy
P4 = P4[0]/maxx, P4[1]/maxy

r = last_bezier_x-first_bezier_x+1
# for t in tqdm(range(first_bezier_x, last_bezier_x-1)):
for t in tqdm(range(r)):
    t /= r          

    # point = (1-t)*2*P1 + 2*(1-t)*t*P2 + t*2*P3
    c_a = pow((1-t),3)
    c_b = 3 * pow((1-t),2) * t
    c_c = 3 * (1-t) * pow(t,2)
    c_d = pow(t,3)

    a = (c_a * P1[0], c_a*P1[1])
    b = (c_b * P2[0], c_b*P2[1])
    c = (c_c * P3[0], c_c*P3[1])
    d = (c_d * P4[0], c_d*P4[1])

    l = [a,b,c,d]
    point = [sum(k) for k in zip(*l)]
    y_middle.append(point[1]*r )

y = list([*y_start, *y_middle, *y_end])

fig, ax = plt.subplots()
# ax.plot(np.arange(0,1,1/r),y_middle)
ax.plot(y)
ax.set(xlabel='laps', ylabel='accuracy',
       title='WN18 also see accuracy.')

fig.savefig(f"{curr_dir}/test.png")