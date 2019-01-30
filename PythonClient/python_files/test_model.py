import dataset_loader
import models
import sys

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import numpy as np

path = '/home/vlad/datasets/{}'.format(sys.argv[2])
camera_dir = sys.argv[3]

dataset = dataset_loader.Dataset(path, camera_dir, limit=10000).get_dataset()

dataset2 = np.stack(np.array(dataset))
X = dataset2[:, 0]
Y = dataset2[:, 2]
Y = Y.astype(np.float64)

model_name = sys.argv[1]

model = models.ModelFromFile(model_name)

print('loaded from file')

prediction = model.predict_raw(X)
print('MAE is ', mean_absolute_error(prediction, Y))
