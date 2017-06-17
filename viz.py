import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
data = train.y
density = gaussian_kde(data)
xs = np.linspace(0,8,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.show(

"""""
To plot importance, use plot_importance. This function requires matplotlib to be installed.

xgb.plot_importance(bst)

To plot the output tree via matplotlib, use plot_tree, specifying the ordinal number of the target tree. This function requires graphviz and matplotlib.

xgb.plot_tree(bst, num_trees=2)

When you use IPython, you can use the to_graphviz function, which converts the target tree to a graphviz instance. The graphviz instance is automatically rendered in IPython.

xgb.to_graphviz(bst, num_trees=2)
"""""