import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import Callback
from sklearn.utils.class_weight import compute_class_weight

# ✅ Stop Training Callback when Accuracy > 90%
class StopTrainingAtAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.99:  
            print("\n✅ Accuracy reached 90%, stopping training!")
            self.model.stop_training = True

# Define constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100  
train_path = "dataset/nail_images/nail/train"
validation_path = "dataset/nail_images/nail/validation"

# Get class labels dynamically from the dataset
labels = [label for label in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, label))]
labels.sort()  # Sort labels for consistency
print(f"✅ Labels used in the model: {labels}")

# ✅ Get class labels dynamically
labels = [label for label in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, label))]
labels.sort()
print(f"✅ Labels used in the model: {labels}")

# ✅ Count samples per class for class weighting
class_counts = {label: len(os.listdir(os.path.join(train_path, label))) for label in labels}
y_labels = []
for label, count in class_counts.items():
    y_labels.extend([label] * count)

# ✅ Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(labels))}
print(f"✅ Computed Class Weights: {class_weight_dict}")

# ✅ Data augmentation for training set (Enhanced)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ✅ Load a pre-trained MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base layers initially

# ✅ Build new model on top of MobileNetV2
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train the model with class weights and stopping condition
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=[StopTrainingAtAccuracy()]
)

# ✅ Fine-tune MobileNetV2 by unfreezing some layers
base_model.trainable = True  # Unfreeze base model
for layer in base_model.layers[:100]:  # Keep first 100 layers frozen
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Retrain the model with fine-tuning
history_finetune = model.fit(
    train_generator,
    epochs=EPOCHS // 2,  
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=[StopTrainingAtAccuracy()]
)

# ✅ Save the trained model
model.save("nail_disease_cnn_model.h5")
print("✅ Model training complete and saved!")

# ✅ Save Training Accuracy & Loss Plots for VS Code Visualization
def plot_training(history, filename="training_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(filename)
    print(f"📊 Training plot saved as {filename}")

plot_training(history, "training_accuracy.png")
plot_training(history_finetune, "fine_tuning_accuracy.png")
