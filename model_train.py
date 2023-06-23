
import readimgfile as readimg
from PIL import Image
import os
import numpy as np
import tensorflow as tf


# Path to dataset directory
dataset_dir = 'C:/Users/JahangirAlam/Desktop/AI/OCR/dataset 2/train'

# List all image files in the dataset directory
image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

# Generate training data for each image file
training_data = []
for image_file in image_files:
    # Get the corresponding annotation file
    xml_file = os.path.splitext(image_file)[0] + '.xml'
    if not os.path.exists(xml_file):
        continue
    # Generate training data
    image, labels = readimg.generate_training_data(image_file, xml_file)
    training_data.append((image, labels))

# Split the training data into training and validation sets
train_data = training_data[:int(len(training_data)*0.8)]
test_data = training_data[int(len(training_data)*0.2)]

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert the training data into a format that can be used for training the model
train_images = np.array([item[0] for item in train_data])
train_labels = np.array([item[1] for item in train_data])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Evaluate the model on the test data
test_images = np.array([item[0] for item in test_data])
test_labels = np.array([item[1] for item in test_data])
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)