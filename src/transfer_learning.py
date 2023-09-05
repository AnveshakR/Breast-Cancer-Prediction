import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os

# Define constants
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_EPOCHS = 1

# Define data directories
cancer_dir = r"/home/lepton/Desktop/Breast Cancer Detection/test_set/1"
non_cancer_dir = r"/home/lepton/Desktop/Breast Cancer Detection/train_set/0"

# Load images
cancer_images = [os.path.join(cancer_dir, f) for f in os.listdir(cancer_dir) if f.endswith('.png')]
non_cancer_images = [os.path.join(non_cancer_dir, f) for f in os.listdir(non_cancer_dir) if f.endswith('.png')]
all_images = cancer_images + non_cancer_images
labels = [1]*len(cancer_images) + [0]*len(non_cancer_images)

# Split images into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(all_images, labels, train_size=0.9, random_state=42)

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'val/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Create train and validation directories
os.makedirs('train/cancer', exist_ok=True)
os.makedirs('train/non_cancer', exist_ok=True)
os.makedirs('val/cancer', exist_ok=True)
os.makedirs('val/non_cancer', exist_ok=True)

# Move images to train and validation directories
for i in range(len(train_images)):
    if train_labels[i] == 1:
        os.replace(train_images[i], 'train/cancer/' + os.path.basename(train_images[i]))
    else:
        os.replace(train_images[i], 'train/non_cancer/' + os.path.basename(train_images[i]))

for i in range(len(val_images)):
    if val_labels[i] == 1:
        os.replace(val_images[i], 'val/cancer/' + os.path.basename(val_images[i]))
    else:
        os.replace(val_images[i], 'val/non_cancer/' + os.path.basename(val_images[i]))




# Load pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Freeze all layers in the VGG16 model
for layer in vgg16_model.layers:
    layer.trainable = False

# Add custom layers on top of the VGG16 model
x = Flatten()(vgg16_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Create a new model with the custom layers on top of the VGG16 model
vgg16_model = Model(inputs=vgg16_model.input, outputs=x)

# Compile the model
vgg16_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

vgg16_model.summary()

# Load pre-trained MobileNetV2 model
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Freeze all layers in the MobileNetV2 model
for layer in mobilenet_model.layers:
    layer.trainable = False

# Add custom layers on top of the MobileNetV2 model
x = Flatten()(mobilenet_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Create a new model with the custom layers on top of the MobileNetV2 model
mobilenet_model = Model(inputs=mobilenet_model.input, outputs=x)

# Compile the model
mobilenet_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

mobilenet_model.summary()

# Load pre-trained ResNet50 model
resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Freeze all layers in the ResNet50 model
for layer in resnet50_model.layers:
    layer.trainable = False

# Add custom layers on top of the ResNet50 model
x = Flatten()(resnet50_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Create a new model with the custom layers on top of the ResNet50 model
resnet50_model = Model(inputs=resnet50_model.input, outputs=x)

# Compile the model
resnet50_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

resnet50_model.summary()



vggHistory = vgg16_model.fit(
train_generator,
steps_per_epoch=len(train_generator),
epochs=NUM_EPOCHS,
validation_data=validation_generator,
validation_steps=len(validation_generator)
)


mobileHistory = mobilenet_model.fit(
train_generator,
steps_per_epoch=len(train_generator),
epochs=NUM_EPOCHS,
validation_data=validation_generator,
validation_steps=len(validation_generator)
)


resnetHistory = resnet50_model.fit(
train_generator,
steps_per_epoch=len(train_generator),
epochs=NUM_EPOCHS,
validation_data=validation_generator,
validation_steps=len(validation_generator)
)