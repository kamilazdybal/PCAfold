import numpy as np
import matplotlib.pyplot as plt

color_UoU = '#CA0614'
color_ULB = '#004A8D'
hfont = {'fontname':'Charter', 'fontweight':'bold'}

fig = plt.figure(figsize=(6,8.2))
phi = np.linspace(0,206,100)
r = phi
x = r * np.cos(phi)
y = r * np.sin(phi)

ax = plt.axes()
plt.plot(x, y, '#4c4c4c')
plt.xticks([])
plt.yticks([])
plt.text(-362, -140, 'PCA', fontsize=100, color=color_UoU, **hfont)
plt.text(-155, -140, 'fold', fontsize=100, color=color_ULB, **hfont)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.axis('equal')
plt.savefig('PCAfold-logo.svg', dpi = 200, bbox_inches='tight', transparent=True)
