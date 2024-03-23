#!/usr/bin/env python
# coding: utf-8

# In[229]:


import numpy as np
import pandas as pd
import wfdb  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[230]:


# Function to preprocess the input analog ECG signal (placeholder: basic digitalization)
def preprocess_ecg(signal, num_levels=256):
    # Discretize the analog signal into 'num_levels' levels
    digital_signal = np.digitize(signal, np.linspace(signal.min(), signal.max(), num_levels))
    return digital_signal

# Load the MIT-BIH Arrhythmia Database
# Replace 'path/to/mitdb/100' with the actual path to the MIT-BIH Arrhythmia Database on your machine
record = wfdb.rdrecord(r'C:\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\100', channels=[0])
annotation = wfdb.rdann(r'C:\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\100', 'atr')

# Extract ECG data
analog_ecg_data = record.p_signal[:, 0]
labels = annotation.symbol

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
# Convert the problem to binary classification (at risk or not at risk)
# Consider any arrhythmia as "at risk" and non-arrhythmia as "safe"
y_binary = np.where(y_encoded != label_encoder.transform(['N'])[0], 1, 0)

# Ensure the arrays have the same length
min_length = min(len(analog_ecg_data), len(y_binary))
analog_ecg_data = analog_ecg_data[:min_length]
y_binary = y_binary[:min_length]

# Preprocess the analog ECG signal
digital_ecg_data = preprocess_ecg(analog_ecg_data)


# In[231]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digital_ecg_data, y_binary, test_size=0.2, random_state=42)

# Check the shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# In[232]:


# Build the neural network model for binary classification
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: sigmoid activation
])


# In[233]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[234]:


# Check the model summary
model.summary()


# In[235]:


# Train the model
# Specify the number of epochs and batch size based on your needs
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[236]:


# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.round(y_pred)  # Round to 0 or 1
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.2f}')


# In[237]:


# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print('Confusion Matrix:')
print(conf_matrix)


# In[238]:


# Print classification report
classification_rep = classification_report(y_test, y_pred_binary)
print('Classification Report:')
print(classification_rep)


# In[239]:


# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[240]:


# Print whether the patient is at risk or safe for the last sample
if y_pred_binary[-1] == 1:
    print("Patient is at risk.")
else:
    print("Patient is safe.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




