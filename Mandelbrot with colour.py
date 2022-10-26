import matplotlib.pyplot as plt
import numpy as np

def mandelbrot(c):
    z = complex(0,0)
    for j in range(300):
        z = z**2 + c
        if np.isnan(abs(z)):
            return False, j
    return True, 0


x_min, x_max = [-2, 1]
y_min, y_max = [-1.5, 1.5]

x_len = abs(x_max - x_min)
y_len = abs(y_max - y_min)

res = 1000
unit = 1/np.sqrt(res)

x_arr = np.arange(x_min, x_max, unit/x_len)
y_arr = np.arange(y_min, y_max, unit/y_len)

print(x_arr.shape, y_arr.shape)

grid = np.array([x_arr + y*1j for y in y_arr]).flatten()
colorchart = np.zeros(len(grid))

mand = np.array([])
for i in range(len(grid)):
    flag, pt_color = mandelbrot(grid[i])
    colorchart[i] = pt_color
    if flag:
        mand = np.append(mand, grid[i])


X = np.array([x.real for x in mand])
Y = np.array([x.imag for x in mand])

X1 = np.array([x.real for x in grid])
Y1 = np.array([x.imag for x in grid])

# plot
# fig_size = 100*unit
# fig, ax = plt.subplots(figsize=(fig_size, fig_size))
fig, ax = plt.subplots(figsize=(x_len+1, y_len+1))


px = 1/plt.rcParams['figure.dpi'] 
num_points = ((.5*unit)**2)/px
marker_size = 1*(num_points**2)

ax.scatter(X1, Y1, marker='o', s=marker_size, c=colorchart, cmap='Spectral')
ax.axis([x_min, x_max, y_min, y_max])
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
plt.show()

ax.axis('off')
fig.savefig('boo', bbox_inches='tight')

# plt.scatter(X,Y, s=0.5, color='black')
# ax.set_aspect('equal', adjustable='box')
# plt.show()