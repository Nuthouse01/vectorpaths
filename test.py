import numpy as np
import matplotlib.pyplot as plt
import logging

import paths

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

t = np.linspace(0,5,100)
xc = (0 + t*2)*np.cos(t*4)
yc = (0 + t*2)*np.sin(t*4)
beziers = paths.fit_cubic_bezier(xc, yc, 0.25)

print(len(beziers))

plt.plot(xc, yc, 'o')
[a.plot() for a in beziers]
plt.show()
