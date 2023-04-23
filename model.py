import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
#Resample to help fix imbalance
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler

# This is basically the numerical representation of the audio file
data = np.load('arr_mfcc.npy')
data = np.moveaxis(data,2,0)

df = pd.read_csv('primary_dataframe.csv')
y = df.diagnosis_Healthy
del df # Get this out of memory

# Split into train, test, and validate
X_train, X_test, y_train, y_test = train_test_split(
    data,y,test_size = 0.2, shuffle = True
)

X_train, X_validate, y_train, y_validate = train_test_split(
    X_train,y_train,test_size = 0.25, shuffle = True
)

# Resample to fix imbalance
oversampler = ADASYN(sampling_strategy='auto')

X_flat = X_train.reshape(X_train.shape[0], -1)

X_oversampled, y_oversampled = oversampler.fit_resample(X_flat, y_train)

X_oversampled_3D = X_oversampled.reshape(-1, X_train.shape[1], X_train.shape[2])

X_trainr = X_oversampled_3D
y_trainr = y_oversampled

#BUILD MODEL SHAPE
m = data.max()
model = Sequential([
  layers.Rescaling(1./m, input_shape=(data.shape[1], data.shape[2],1)),
  layers.Conv2D(16, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),

  layers.Conv2D(32, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),

  layers.Conv2D(64, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),

  layers.Conv2D(64, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),

  layers.Conv2D(64, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(1,activation = 'sigmoid')
])

######
def weighted_binary_cross_entropy(y_true, y_pred):
    # Define the weight for the minority class
    w = tf.constant(32.33, dtype=tf.float32)
    
    # Compute the binary cross-entropy loss for each example
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    
    # Apply the weights to the loss
    weighted_bce = tf.keras.backend.mean((1 - w) * y_true * bce + w * (1 - y_true) * bce)
    
    return weighted_bce

#Compile model
model.compile(optimizer='adam',
              loss=weighted_binary_cross_entropy,
              metrics=['BinaryAccuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

#Fit model
num_batch_size = 256
num_epochs = 250

model.fit(x=X_trainr, y=y_trainr, batch_size = num_batch_size,
          validation_data=(X_validate, y_validate), epochs=num_epochs)