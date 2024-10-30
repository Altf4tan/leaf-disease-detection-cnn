import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# data preprocessing and splitting
data_directory = "C:\\Users\\Example"

images = []
labels = []
class_labels = []

for class_label, class_dir in enumerate(os.listdir(data_directory)):
    class_labels.append(class_dir)
    class_dir_path = os.path.join(data_directory, class_dir)

    for image_name in os.listdir(class_dir_path):
        image_path = os.path.join(class_dir_path, image_name)
        image = cv2.imread(image_path)
        images.append(image)
        labels.append(class_label)

images = np.array(images)
labels = np.array(labels)

processed_images = []

for image in images:
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    processed_images.append(normalized_image)

processed_images = np.array(processed_images)

one_hot_labels = to_categorical(labels, num_classes=len(class_labels))

X_train, X_test, y_train, y_test = train_test_split(processed_images, one_hot_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Creation of model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

input_shape = (224, 224, 3)
num_classes = len(class_labels)

model = create_cnn_model(input_shape, num_classes)
model.summary()

# Compile the model and prepare for educating it
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Educate the model
batch_size = 32
epochs = 20

checkpoint_filepath = "best_model.h5"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[model_checkpoint_callback])

#Validate the educated models performance
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
