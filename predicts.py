import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Directory paths for training and validation data
train_data_dir = 'C:/Emotion_detection/train'
validation_data_dir = 'C:/Emotion_detection/test'

# Image Data Generators for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Generating batches of images and labels for training
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

# Generating batches of images and labels for validation
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

# Class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the trained models
cnn_model = tf.keras.models.load_model('saved_model.h5')
svm_model = joblib.load('best_svm_model.pkl')  # Load the SVM model

# Using validation_generator for predictions
test_generator = validation_generator

batch_size = test_generator.batch_size

# Selecting a random batch from the validation generator
Random_batch = np.random.randint(0, len(test_generator) - 1)

# Selecting random image indices from the batch
Random_Img_Index = np.random.randint(0, batch_size, 10)

# Setting up the plots for CNN and SVM predictions
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))

# Plot CNN predictions
for i, ax in enumerate(axes[0]):
    # Fetching the random image and its label
    Random_Img = test_generator[Random_batch][0][Random_Img_Index[i]]
    Random_Img_Label = np.argmax(test_generator[Random_batch][1][Random_Img_Index[i]], axis=0)

    # Making a prediction using the CNN model
    cnn_prediction = np.argmax(cnn_model.predict(tf.expand_dims(Random_Img, axis=0), verbose=0), axis=1)[0]

    # Displaying the image
    ax.imshow(Random_Img.squeeze(), cmap='gray')  # Assuming the images are grayscale
    
    # Setting the title with CNN predictions
    if class_labels[Random_Img_Label] == class_labels[cnn_prediction]:
        title = f"True: {class_labels[Random_Img_Label]}\nPredicted: {class_labels[cnn_prediction]}"
        color = "green"
    else:
        title = f"True: {class_labels[Random_Img_Label]}\nPredicted: {class_labels[cnn_prediction]}"
        color = "red"
    
    ax.set_title(title, color=color, fontsize=10)
    ax.axis('off')

# Add the title for CNN Predictions row
axes[0, 0].set_title('CNN Predictions', fontsize=12, loc='center')

# Plot SVM predictions
for i, ax in enumerate(axes[1]):
    # Fetching the random image and its label
    Random_Img = test_generator[Random_batch][0][Random_Img_Index[i]]
    Random_Img_Label = np.argmax(test_generator[Random_batch][1][Random_Img_Index[i]], axis=0)

    # For SVM, flatten the image
    flattened_img = Random_Img.flatten().reshape(1, -1)
    svm_prediction = svm_model.predict(flattened_img)[0]

    # Displaying the image
    ax.imshow(Random_Img.squeeze(), cmap='gray')  # Assuming the images are grayscale
    
    # Setting the title with SVM predictions
    if class_labels[Random_Img_Label] == class_labels[svm_prediction]:
        title = f"True: {class_labels[Random_Img_Label]}\nPredicted: {class_labels[svm_prediction]}"
        color = "green"
    else:
        title = f"True: {class_labels[Random_Img_Label]}\nPredicted: {class_labels[svm_prediction]}"
        color = "red"
    
    ax.set_title(title, color=color, fontsize=10)
    ax.axis('off')

# Add the title for SVM Predictions row
axes[1, 0].set_title('SVM Predictions', fontsize=12, loc='center')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust the layout to make room for titles
plt.show()
