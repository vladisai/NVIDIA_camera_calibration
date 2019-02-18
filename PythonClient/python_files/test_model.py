import dataset_loader
import models
import sys

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import numpy as np

path = '/home/vlad/datasets/{}'.format(sys.argv[2])
X, Y = dataset_loader.loadXY(path, 1)

model_name = sys.argv[1]
model = models.ModelFromFile(model_name)

print('loaded from file')

print('MAE is ', model.validate(X, Y))
