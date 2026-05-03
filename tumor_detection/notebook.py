import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomZoom, RandomTranslation





def check_gpu_availability():
    print(f'Tensorflow Version: {tf.__version__}')
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU Available: {physical_devices[0]}")
    else:
        print("No GPU available.")


def setup_visualization():
    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams.update({'font.size': 14})




def get_data_labels(directory, shuffle=True, random_state=0):
    from sklearn.utils import shuffle
    """
    Loads data paths and labels from the specified directory.
    """
    
    data_path = []
    data_index = []
    label_dict = {label: index for index, label in enumerate(sorted(os.listdir(directory)))}

    for label, index in label_dict.items():
        label_dir = os.path.join(directory, label)
        for image in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image)
            data_path.append(image_path)
            data_index.append(index)

    if shuffle:
        data_path, data_index = shuffle(data_path, data_index, random_state=random_state)

    return data_path, data_index


def get_data_labelss(directory, shuffle=True, random_state=0):
    from sklearn.utils import shuffle
    """
    Loads data paths and labels from the specified directory.
    """
    
    data_path = []
    data_index = []
    label_dict = {label: index for index, label in enumerate(sorted(os.listdir(directory)))}

    for label, index in label_dict.items():
        label_dir = os.path.join(directory, label)
        for image in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image)
            data_path.append(image_path)
            data_index.append(index)

    if shuffle:
        data_path, data_index = shuffle(data_path, data_index, random_state=random_state)

    # Ensure data_index contains integers
    data_index = list(map(int, data_index))

    return data_path, data_index


def parse_function(filename, label, image_size, n_channels):
    """
    Parses the image file, resizes it, and returns the image with its label.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, n_channels)
    image = tf.image.resize(image, image_size)
    return image, label

def parsee_function(filename, label=None, image_size=(168, 168), n_channels=1):
    """
    Parses the image file, decodes, resizes, normalizes, and optionally returns the image with its label.
    
    Args:
    - filename (str): The path to the image file.
    - label (int, optional): The label associated with the image (default is None).
    - image_size (tuple): The target size for the image (height, width).
    - n_channels (int): Number of color channels (1 for grayscale, 3 for RGB).
    
    Returns:
    - image (tf.Tensor): Preprocessed image tensor.
    - label (int, optional): Corresponding label if provided, else None.
    """
    # Read the image file
    image_string = tf.io.read_file(filename)
    
    # Decode the image (JPEG format expected)
    image = tf.image.decode_jpeg(image_string, channels=n_channels)
    
    # Resize the image to the desired size
    image = tf.image.resize(image, image_size)
    
    # Normalize the image to [0, 1] range
    image = image / 255.0
    
    if label is not None:
        return image, label
    else:
        return image



def get_dataset(paths, labels, image_size, n_channels=1, num_classes=4, batch_size=32):
    """
    Creates a TensorFlow dataset from the provided image paths and labels.
    """
    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    image_label_ds = path_ds.map(lambda path, label: parse_function(path, label, image_size, n_channels),
                                 num_parallel_calls=tf.data.AUTOTUNE)
    return image_label_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


################################################"


def prepare_data(USER_PATH, image_dim=(168, 168), batch_size=32, seed=111):
    """
    Prepares training and testing datasets.
    
    Args:
        USER_PATH (str): The base path to the dataset directory.
        image_dim (tuple): Dimensions to resize images.
        batch_size (int): Batch size for the datasets.
        seed (int): Random seed for reproducibility.
    
    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        test_ds (tf.data.Dataset): Testing dataset.
        class_mappings (dict): Dictionary mapping class names to integers.
        inv_class_mappings (dict): Dictionary mapping integers to class names.
        class_names (list): List of class names.
    """
    train_paths, train_index = get_data_labels(os.path.join(USER_PATH, 'Training'), random_state=seed)
    test_paths, test_index = get_data_labels(os.path.join(USER_PATH, 'Testing'), random_state=seed)

    print('Training')
    print(f'Number of Paths: {len(train_paths)}')
    print(f'Number of Labels: {len(train_index)}')
    print('\nTesting')
    print(f'Number of Paths: {len(test_paths)}')
    print(f'Number of Labels: {len(test_index)}')

    train_ds = get_dataset(train_paths, train_index, image_size=image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    test_ds = get_dataset(test_paths, test_index, image_size=image_dim, n_channels=1, num_classes=4, batch_size=batch_size)

    print(f"\nTraining dataset: {train_ds}")
    print(f"\nTesting dataset: {test_ds}")

    class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}
    inv_class_mappings = {v: k for k, v in class_mappings.items()}
    class_names = list(class_mappings.keys())

    return train_ds, test_ds, class_mappings, inv_class_mappings, class_names



#################################

# notebook.py

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


def show_images(paths, label_paths, class_mappings, index_list=range(10), im_size=250, figsize=(12, 8)):
    """
    Displays a list of images based on the given index.
    
    Args:
        paths (list): List of image file paths.
        label_paths (list): List of corresponding labels.
        class_mappings (dict): Dictionary mapping class labels to class names.
        index_list (range): Range of indices of images to display.
        im_size (int): Size to resize images.
        figsize (tuple): Figure size for the plot.
        
    Returns:
        fig: Matplotlib figure object.
    """
    num_images = len(index_list)
    num_rows = (num_images + 3) // 4
    index_to_class = {v: k for k, v in class_mappings.items()}
    fig, ax = plt.subplots(nrows=num_rows, ncols=4, figsize=figsize)
    ax = ax.flatten()

    for i, index in enumerate(index_list):
        if i >= num_images:
            break
        img = image.load_img(paths[index], target_size=(im_size, im_size), color_mode='grayscale')
        ax[i].imshow(img, cmap='Greys_r')
        class_name = index_to_class[label_paths[index]]
        ax[i].set_title(f'{index}: {class_name}')
        ax[i].axis('off')

    plt.tight_layout()
    return fig  # Return the figure object

################################5 | Training Setup ###################################################



import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomZoom, RandomTranslation
from tensorflow.keras.models import Sequential

# Data augmentation sequential model
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.02, fill_mode='constant'),
    RandomContrast(0.1),
    RandomZoom(height_factor=0.01, width_factor=0.05),
    RandomTranslation(height_factor=0.0015, width_factor=0.0015, fill_mode='constant'),
])

# Training augmentation and normalization
def preprocess_train(image, label):
    # Apply data augmentation and Normalize
    image = data_augmentation(image) / 255.0
    return image, label

# For test dataset only applying normalization
def preprocess_test(image, label):
    return image / 255.0, label

# Function to apply preprocessing
def apply_preprocessing(train_ds, test_ds):
    train_ds_preprocessed = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds_preprocessed = test_ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_ds_preprocessed, test_ds_preprocessed


#########################5.1 | Visualizing Augmentated Images############################


import matplotlib.pyplot as plt

def plot_augmented_images(dataset, shape, class_mappings, figsize=(15, 6)):
    plt.figure(figsize=figsize)
    index_to_class = {v: k for k, v in class_mappings.items()}
    
    for images, label in dataset.take(1):
        for i in range(shape[0]*shape[1]):
            ax = plt.subplot(shape[0], shape[1], i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            plt.title(index_to_class[label.numpy()[i]])
            plt.axis("off")

    plt.tight_layout()
    plt.show()

def save_augmented_images(dataset, shape, class_mappings, figsize=(15, 6), filename='augmented_images.png'):
    plt.figure(figsize=figsize)
    index_to_class = {v: k for k, v in class_mappings.items()}
    
    for images, label in dataset.take(1):
        for i in range(shape[0] * shape[1]):
            ax = plt.subplot(shape[0], shape[1], i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            plt.title(index_to_class[label.numpy()[i]])
            plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(filename)


#########################5.2 | Training Setup############################


import tensorflow as tf

def preprocess_and_encode_labels(train_ds, test_ds, class_mappings, image_dim, batch_size, epochs):
    num_classes = len(class_mappings.keys())
    image_shape = (image_dim[0], image_dim[1], 1)

    # Print the details
    print(f'Number of Classes: {num_classes}')
    print(f'Image shape: {image_shape}')
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batch_size}')

    # Function to encode labels
    def encode_labels(image, label):
        return image, tf.one_hot(label, depth=num_classes)

    # Apply encoding
    train_ds_preprocessed = train_ds.map(encode_labels, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds_preprocessed = test_ds.map(encode_labels, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds_preprocessed, test_ds_preprocessed


#########################6 | Building CNN Model############################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

def build_model(image_shape, num_classes, learning_rate=0.001):
    # Building the model
    model = Sequential([
        Input(shape=image_shape),
        Conv2D(64, (5, 5), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),

        Conv2D(64, (5, 5), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),

        Conv2D(128, (4, 4), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (4, 4), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),

        Dense(512, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    # Model summary
    model.summary()

    # Compile the model with Adam optimizer
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.85, beta_2=0.9925)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


#########################6.1 | CNN Training############################

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Custom callback class for reducing learning rate at specific accuracy thresholds
class ReduceLROnMultipleAccuracies(tf.keras.callbacks.Callback):
    def __init__(self, thresholds, factor, monitor='val_accuracy', verbose=1):
        super(ReduceLROnMultipleAccuracies, self).__init__()
        self.thresholds = thresholds  # List of accuracy thresholds
        self.factor = factor  # Factor to reduce the learning rate
        self.monitor = monitor
        self.verbose = verbose
        self.thresholds_reached = [False] * len(thresholds)  # Track each threshold

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get(self.monitor)
        for i, threshold in enumerate(self.thresholds):
            if current_accuracy >= threshold and not self.thresholds_reached[i]:
                optimizer = self.model.optimizer
                old_lr = optimizer.learning_rate.numpy()
                new_lr = old_lr * self.factor
                optimizer.learning_rate.assign(new_lr)
                self.thresholds_reached[i] = True  # Mark this threshold as reached
                if self.verbose > 0:
                    print(f"\nEpoch {epoch+1}: {self.monitor} reached {threshold}. Reducing learning rate from {old_lr} to {new_lr}.")

# Function to train the model
def train_model(model, train_ds_preprocessed, test_ds_preprocessed, epochs, thresholds, factor):
    # Custom callback for reducing learning rate
    lr_callback = ReduceLROnMultipleAccuracies(thresholds=thresholds, factor=factor, monitor='val_accuracy', verbose=True)

    # Other callbacks for training
    model_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, min_lr=1e-4, patience=4, verbose=True)
    model_mc = ModelCheckpoint('model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=True)

    # Training the model
    history = model.fit(
        train_ds_preprocessed,
        epochs=epochs,
        validation_data=test_ds_preprocessed,
        callbacks=[lr_callback, model_rlr, model_mc],
        verbose=True
    )

    
    
    return history


#########################6.2 | Model Evaluation#######################

from tensorflow.keras.models import load_model

# Function to load and evaluate the saved model
def load_and_evaluate_model(model_path, test_ds_preprocessed):
    # Load the saved model
    model = load_model(model_path)

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_ds_preprocessed)

    # Return the test accuracy and loss
    return test_loss, test_acc

######################
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to plot training history and save the plot
def plot_training_history(history, save_path='plot.png'):
    _, ax = plt.subplots(ncols=2, figsize=(15, 6))

    # Plotting training and validation accuracy over epochs
    ax[0].plot(history['accuracy'], marker='o', linestyle='-', color='blue')
    ax[0].plot(history['val_accuracy'], marker='o', linestyle='-', color='orange')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(['Train', 'Validation'], loc='lower right')
    ax[0].grid(alpha=0.2)

    # Plotting training and validation loss over epochs
    ax[1].plot(history['loss'], marker='o', linestyle='-', color='blue')
    ax[1].plot(history['val_loss'], marker='o', linestyle='-', color='orange')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend(['Train', 'Validation'], loc='upper right')
    ax[1].grid(alpha=0.2)

    # Highlight lowest validation accuracy
    min_val_acc_epoch = np.argmax(history['val_accuracy'])
    min_val_acc = np.max(history['val_accuracy'])
    ax[0].plot(min_val_acc_epoch, min_val_acc, 'ro', markersize=15, alpha=0.5)
    ax[0].annotate(f'Lowest\n{min_val_acc:.4f}', xy=(min_val_acc_epoch, min_val_acc),
                   xytext=(min_val_acc_epoch - 100, min_val_acc - 100), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    # Save the plot to the specified path
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory


    ###############################################
    
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from django.core.files.storage import default_storage




# Function to load and preprocess an image
def load_and_preprocess_image(image_file, image_shape=(168, 168)):
    #image = parsee_function(image_file, label=None, image_size=image_shape, n_channels=n_channels)

    img = image.load_img(image_file, target_size=image_shape, color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add the batch dimension
    return img_array

# Function to make predictions
def predict_image(model, image_file):
    preprocessed_image = load_and_preprocess_image(image_file)
    prediction = model.predict(preprocessed_image)
    return prediction

# Function to decode predictions to labels
def decode_predictions(predictions, class_mappings):
    inv_class_mappings = {v: k for k, v in class_mappings.items()}
    predicted_labels = [inv_class_mappings[np.argmax(one_hot)] for one_hot in predictions]
    return predicted_labels
