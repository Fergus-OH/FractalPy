from matplotlib import units
import numpy as np
import matplotlib.pyplot as plt

dpi = 100
unit = 1/np.sqrt(dpi)

length = 5
x = np.arange(0, length, unit)
y = np.array([2]*len(x))

fig, ax = plt.subplots(figsize=(length+1,length+1))

px = 1/plt.rcParams['figure.dpi']
print(plt.rcParams['figure.dpi'])

ppi = 1/np.sqrt(plt.rcParams['figure.dpi'])

num_points = (np.pi*(.5*unit*ppi)**2)
print(num_points)

ax.scatter(x, y, s=(.5*unit*ppi)**2)
ax.axis([0,length,0,length])
fig.tight_layout()
plt.show()


# x2 = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5])
# y2 = np.array([2,2,2,2,2,2,2,2,2])

# fig, ax = plt.subplots(figsize=(5,5))
# ax.scatter(x2, y2, s=4)
# plt.show()
