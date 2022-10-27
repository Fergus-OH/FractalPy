import matplotlib.pyplot as plt
import numpy as np

c = (-0.79 + 0.15j)

def julia(z):
    for j in range(300):
        z = z**2 + c
        if np.isnan(abs(z)):
            return False, j
    return True, 0

    
ar = np.linspace(-1.5,1.5,3000)
init = np.array([ar + x*1j for x in ar]).flatten()


colorchart = np.zeros(len(init))

mand = np.array([])
for i in range(len(init)):
    trueFalse, col = julia(init[i])
    
    colorchart[i] = col
    if trueFalse:
        mand = np.append(mand, init[i])


X = np.array([x.real for x in mand])
Y = np.array([x.imag for x in mand])

X1 = np.array([x.real for x in init])
Y1 = np.array([x.imag for x in init])

plt.figure(figsize=(36,24))
plt.title('c = '+ str(c))
plt.axis('equal')
plt.scatter(X1,Y1, s=0.1, c=colorchart, cmap='prism')
#plt.scatter(X,Y, s=0.5, color='black')
plt.show()


#Julia set coordinates:{(-0.79 + 0.15j), (0.28 + 0.008j)}