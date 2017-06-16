import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
data = train.y
density = gaussian_kde(data)
xs = np.linspace(0,8,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.show()