import inline as inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
observation_set = 2000


pts1 = np.random.multivariate_normal([0,0], [[],[]], observation_set)
pts2 = np.random.multivariate_normal([0,0], [[],[]], observation_set)