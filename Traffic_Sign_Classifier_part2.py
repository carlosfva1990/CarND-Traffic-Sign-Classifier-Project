##esto se usa para evitar los warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
####
import tensorflow as tf
import hashlib
import pickle
from tensorflow.contrib.layers import flatten
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

from sklearn.utils import shuffle


# Reload the data

pickle_file = 'Señales_Trafico.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  X_train = pickle_data['X_train']
  y_train = pickle_data['y_train']
  X_valid = pickle_data['X_valid']
  y_valid = pickle_data['y_valid']
  X_test = pickle_data['X_test']
  y_test = pickle_data['y_test']
  del pickle_data  # Free up memory


print(" ")
print(" ")
print('Data and modules loaded.')


# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

print(" ")
print(" ")
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


    

# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
y_valid = encoder.transform(y_valid)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
y_valid = y_valid.astype(np.float32)
is_labels_encod = True



print(" ")
print(" ")
print(y_train.shape)
print(y_test.shape)
print(y_valid.shape)
print('Labels One-Hot Encoded')
    

print(" ")
print(" ")
print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_valid.shape, y_valid.shape)
print('Test set', X_test.shape, y_test.shape)
print('Test 20', X_test[20][1])

print(" ")
print(" ")


    
    



def accuracy(predictions, labels): 
  cont=np.sum(np.argmax(predictions,1) == np.argmax(labels,1))
  return (cont*100/ predictions.shape[0])

         

image_size = len(X_train[0][0])
num_labels = n_classes
num_channels = 3 
EPOCHS = 20
batch_size = 150
patch_size = 3
depth = 6
num_hidden = 250
num_hidden2 = 100
pool_size=2
dropout = 0.8
grad=0.001

# Input data.
tf_train_dataset = tf.placeholder(
  tf.float32, shape=(None, image_size, image_size, num_channels))
tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
tf_valid_dataset = tf.constant(X_valid)
tf_test_dataset = tf.constant(X_test)
  

    
  # Model.
def model(data):
#conv1 input 32x32x3 out 15x15xdepth
  conv1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth],mean=0.0 ,stddev=0.1))
  conv1_biases = tf.Variable(tf.zeros([depth]))
  
  layer1 = tf.nn.conv2d(data, conv1_weights, [1, 1, 1, 1], padding='VALID')
  hidden_layer1 = tf.nn.relu(layer1 + conv1_biases)
  #30x30xdepth
  hidden_layer1 = tf.nn.max_pool(hidden_layer1,[1, pool_size, pool_size, 1], [1, pool_size, pool_size, 1], padding='VALID')
  
  #conv2 input 15x15xdepth out 8x8xdepth*2
  conv2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth*2],mean=0.0 , stddev=0.1))
  conv2_biases = tf.Variable(tf.zeros([depth*2]))
        
  conv2 = tf.nn.conv2d(hidden_layer1, conv2_weights, [1, 1, 1, 1], padding='VALID')
  hidden_layer2 = tf.nn.relu(conv2 + conv2_biases)
  #13x13xdepth*2
  hidden_layer2 = tf.nn.max_pool(hidden_layer2,[1, pool_size, pool_size, 1], [1, pool_size, pool_size, 1], padding='VALID')
  
  #fc1 input 6x6xdepth*2 out num_hidden
  fc0= flatten(hidden_layer2)
  fc1_weights = tf.Variable(tf.truncated_normal(
      [((((image_size-2)//pool_size)-2)//pool_size)*((((image_size-2)//pool_size)-2)//pool_size) * depth*2, num_hidden], stddev=0.1))
      #[6*6 * depth*2, num_hidden], stddev=0.1))
  fc1_biases = tf.Variable(tf.zeros([num_hidden]))
  fc1=tf.nn.dropout(tf.nn.relu(tf.matmul(fc0, fc1_weights)+fc1_biases),keep_prob=dropout)
  
  #fc2 input num_hidden out num_hidden2
  fc2_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_hidden2], stddev=0.1))
  fc2_biases = tf.Variable(tf.zeros([num_hidden2]))
  fc2=tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, fc2_weights)+fc2_biases),keep_prob=dropout)
  
  #fc3 input num_hidden2 out n_classes
  fc3_weights = tf.Variable(tf.truncated_normal(
      [num_hidden2, n_classes], stddev=0.1))
  fc3_biases = tf.Variable(tf.zeros([n_classes]))
  fc3=tf.nn.relu(tf.matmul(fc2, fc3_weights)+fc3_biases)
  return fc3
  
    
  

# Training computation.
logits = model(tf_train_dataset)
loss_operation = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
# Optimizer.
optimizer = tf.train.AdamOptimizer(grad)
training_operation=optimizer.minimize(loss_operation)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits)
valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
test_prediction = tf.nn.softmax(model(tf_test_dataset))



correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf_train_labels, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={tf_train_dataset: batch_x, tf_train_labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  num_examples = len(X_train)
  
  print("Training...")
  print()
  for i in range(EPOCHS):
      X_train, y_train = shuffle(X_train, y_train)
      for offset in range(0, num_examples, batch_size):
          end = offset + batch_size
          batch_x, batch_y = X_train[offset:end], y_train[offset:end]
          l,_=sess.run([loss_operation,training_operation], feed_dict={tf_train_dataset: batch_x, tf_train_labels: batch_y})
          
      validation_accuracy = evaluate(X_valid, y_valid)
      print("EPOCH {} ...".format(i+1))
      print("Validation Accuracy = {:.3f}".format(validation_accuracy))
      print(l)
      if(l<1):
          grad=grad-0.00001
      
      print()
      
      test_accuracy = evaluate(X_test, y_test)
      print("Test Accuracy = {:.3f}".format(test_accuracy))
  saver.save(sess, './lenet')
  print("Model saved")
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  