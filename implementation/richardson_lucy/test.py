
import numpy as np
import drawnow
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

def draw_figs():
    plt.plot(x, y)


x = np.linspace(0, 1)
#drawnow.figure()
plt.figure()
mpl.interactive = True
for p in [0, 1, 2, 3, 4, 5, 6]:
    print('p = {}'.format(p))
    y = x**p
    drawnow.drawnow(draw_figs)
    time.sleep(1)
