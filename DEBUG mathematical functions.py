import matplotlib.pyplot as plt
import numpy as np
from nlsfunc import *

Temp = np.arange(0, 60, 0.1)

fig, ax = plt.subplots()
ax.plot(Temp, gau(Temp, [100, 30, 5]))
ax.plot(Temp, qua(Temp, [-0.2, 5, 10]))
ax.plot(Temp, bet(Temp, [10, 50, 15, 60]))
ax.set_xlim([0, 60])
ax.set_ylim([0, 100])

fig, ax = plt.subplots()
ax.plot(Temp, gaugau(Temp, [1000, 20, 5, 1000, 45, 5, 1, 35]))
ax.plot(Temp, quaqua(Temp, [-0.5, 29, -301, -0.6, 32, -165, 0.7, 37.5]))
ax.plot(Temp, betbet(Temp, [10, 43, 30, 118, -6619, 45, 39, 118, 2.4, 33]))
ax.set_xlim([0, 60])
ax.set_ylim([0, 200])

fig, ax = plt.subplots()
ax.plot(Temp, quagau(Temp, [-0.6, 32, -165, 1000, 45, 5, 1, 35]))
ax.plot(Temp, quabet(Temp, [-0.5, 29, -301, -6619, 45, 39, 118, 0.7, 37.5]))
ax.plot(Temp, gaubet(Temp, [1000, 20, 5, -6619, 45, 39, 118, 2.4, 33]))
ax.set_xlim([0, 60])
ax.set_ylim([0, 200])