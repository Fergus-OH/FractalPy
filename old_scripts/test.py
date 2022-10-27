from matplotlib import units
import numpy as np
import matplotlib.pyplot as plt

dpi = 3
unit = 1/dpi

length = 5
x = np.arange(0, length, unit)
y = np.array([2]*len(x))

fig, ax = plt.subplots(figsize=(length+1,length+1), dpi=200)
num_points = (.5*unit*fig.dpi)

ax.scatter(x, y, marker='o', s=num_points**2, linewidths=0)
ax.axis([0,length,0,length])
ax.axis('off')
fig.tight_layout()
plt.show()


# x2 = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5])
# y2 = np.array([2,2,2,2,2,2,2,2,2])

# fig, ax = plt.subplots(figsize=(5,5))
# ax.scatter(x2, y2, s=4)
# plt.show()
