# Load pickled data# Load pickled data
import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile



# Set flags for feature engineering.  This will prevent you from skipping an important step.
is_features_normal = False
is_labels_encod = False


# TODO: Fill this in based on where you saved the training and testing data

training_file="C:/Users/carlo/Documents/CIDETEC/vision 3d/Nueva carpeta (3)/trunk/train.p"
validation_file="C:/Users/carlo/Documents/CIDETEC/vision 3d/Nueva carpeta (3)/trunk/valid.p"
testing_file="C:/Users/carlo/Documents/CIDETEC/vision 3d/Nueva carpeta (3)/trunk/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
print(y_train.shape)
print(y_train)


# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43
#print(X_test)
print(n_validation)
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape, )
print("Number of classes =", n_classes)



image_size = len(X_train[0][0])
num_labels = n_classes
num_channels = 3 # grayscale

def reformat(dataset):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset
  
X_train = reformat(X_train)
X_valid = reformat(X_valid)
X_test = reformat(X_test)
    
print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_valid.shape, y_valid.shape)
print('Test set', X_test.shape, y_test.shape)
print("test done")

X_test = (1.0/128.0)*(np.float32(X_test)-128.0)
X_train = (1.0/128.0)*(np.float32(X_train)-128.0)
X_valid = (1.0/128.0)*(np.float32(X_valid)-128.0)


# Save the data for easy access

pickle_file = 'Señales_Trafico.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('Señales_Trafico.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_valid': X_valid,
                    'y_valid': y_valid,
                    'X_test': X_test,
                    'y_test': y_test,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')









