import matplotlib.pyplot as plt
import numpy as np

def mandelbrot(c):
    z = complex(0,0)
    for j in range(300):
        z = z**2 + c
        if np.isnan(abs(z)):
            return False, j
    return True, 0

    
ar = np.linspace(-2,1.5,1000)
init = np.array([ar + x*1j for x in ar]).flatten()


colorchart = np.zeros(len(init))

mand = np.array([])
for i in range(len(init)):
    trueFalse, col = mandelbrot(init[i])
    
    colorchart[i] = col
    if trueFalse:
        mand = np.append(mand, init[i])


X = np.array([x.real for x in mand])
Y = np.array([x.imag for x in mand])

X1 = np.array([x.real for x in init])
Y1 = np.array([x.imag for x in init])

plt.figure(figsize=(10,10))
plt.xlim(-2,1)
plt.ylim(-1.5,1.5)
plt.scatter(X1,Y1, s=0.5, c=colorchart, cmap='Spectral')
#plt.scatter(X,Y, s=0.5, color='black')
plt.show()