import numpy as np
import os
import matplotlib.pyplot as plt
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import HttpResponse
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.utils import shuffle
from django.shortcuts import render
from datetime import datetime
from .notebook import train_model, plot_training_history,apply_preprocessing,save_augmented_images,build_model,load_and_evaluate_model,show_images

def index_view(request):
    return render(request, 'tumor_detection/index.html') 

def homepage_view(request):
    return HttpResponse('Welcome to Tumor Detection App!')

USER_PATH = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset"
SEED = 111
train_paths, train_index = get_data_labels(USER_PATH + '/Training', random_state=SEED)
test_paths, test_index = get_data_labels(USER_PATH + '/Testing', random_state=SEED)
# Example class names and dataset (replace with actual variables)
class_names = ['Glioma', 'Meninigioma', 'Notumor', 'Pituitary']  # Adjust as needed
#batch_size = 32
#image_dim = (168, 168)
#num_classes = 4

# Prepare datasets
#train_ds = get_dataset(train_paths, train_index, image_dim, n_channels=1, num_classes=num_classes, batch_size=batch_size)
#test_ds = get_dataset(test_paths, test_index, image_dim, n_channels=1, num_classes=num_classes, batch_size=batch_size)

# Preprocess datasets
#train_ds_preprocessed, test_ds_preprocessed = apply_preprocessing(train_ds, test_ds)
#train_ds_preprocessed, test_ds_preprocessed = preprocess_and_encode_labels(train_ds_preprocessed, test_ds_preprocessed, num_classes)


def your_view_function(request):
    check_gpu_availability()
    setup_visualization()
    # other code for handling the view


##############################################################

def train_model_view(request):
    # Set the directory containing your images
    image_directory = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset/Training"

    # Get data paths and labels
    data_paths, data_labels = get_data_labels(image_directory)

    # Set parameters for image processing
    image_size = (128, 128)  # Change as per your model's input size
    n_channels = 1  # Grayscale images
    batch_size = 32

    # Get the dataset
    dataset = get_dataset(data_paths, data_labels, image_size, n_channels, batch_size=batch_size)

    # You can now use this dataset for model training
    # ...
    return render(request, 'your_template.html', context)


def load_data(request):
    # Getting data labels
   

    # Prepare datasets
    batch_size = 32
    image_dim = (168, 168)
    train_ds = get_dataset(train_paths, train_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    test_ds = get_dataset(test_paths, test_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)

    # Define the context with paths and datasets to pass to the template
    context = {
        'training_dataset_size': len(train_paths),
        'testing_dataset_size': len(test_paths),
        'train_ds': str(train_ds),  # Convert datasets to string for visualization
        'test_ds': str(test_ds),
    }
    
    # Render the 'result.html' template with the context
    return render(request, 'tumor_detection/result.html', context)

#########################Data Visualization #################################"

def plot_data_distributions(request):
    # Generate plot
    fig, ax = plt.subplots(ncols=3, figsize=(20, 14))

    # Get data labels
    _, train_index = get_data_labels(os.path.join(USER_PATH, 'Training'), random_state=SEED)
    _, test_index = get_data_labels(os.path.join(USER_PATH, 'Testing'), random_state=SEED)

    # Ensure that train_index and test_index are flat lists of class labels
    if not all(isinstance(label, int) for label in train_index):
        raise ValueError("train_index should be a list of integers")
    if not all(isinstance(label, int) for label in test_index):
        raise ValueError("test_index should be a list of integers")

    # Check if the class names match the number of classes
    num_classes = len(class_names)
    unique_train_labels = sorted(set(train_index))
    unique_test_labels = sorted(set(test_index))
    if len(unique_train_labels) != num_classes or len(unique_test_labels) != num_classes:
        raise ValueError("Mismatch between number of class names and unique labels in data")

    # Plotting training data types
    class_counts = [len([x for x in train_index if x == label]) for label in range(num_classes)]
    ax[0].set_title('Training Data')
    ax[0].pie(
        class_counts,
        labels=[class_names[label] for label in range(num_classes)],
        colors=['#FAC500', '#0BFA00', '#0066FA', '#FA0000'],
        autopct=lambda p: '{:.2f}%\n{:,.0f}'.format(p, p * sum(class_counts) / 100),
        explode=(0.01, 0.01, 0.1, 0.01),
        textprops={'fontsize': 20}
    )

    # Plotting distribution of train test split
    ax[1].set_title('Train Test Split')
    ax[1].pie(
        [len(train_index), len(test_index)],
        labels=['Train', 'Test'],
        colors=['darkcyan', 'orange'],
        autopct=lambda p: '{:.2f}%\n{:,.0f}'.format(p, p * sum([len(train_index), len(test_index)]) / 100),
        explode=(0.1, 0),
        startangle=85,
        textprops={'fontsize': 20}
    )

    # Plotting testing data types
    class_counts = [len([x for x in test_index if x == label]) for label in range(num_classes)]
    ax[2].set_title('Testing Data')
    ax[2].pie(
        class_counts,
        labels=[class_names[label] for label in range(num_classes)],
        colors=['#FAC500', '#0BFA00', '#0066FA', '#FA0000'],
        autopct=lambda p: '{:.2f}%\n{:,.0f}'.format(p, p * sum(class_counts) / 100),
        explode=(0.01, 0.01, 0.1, 0.01),
        textprops={'fontsize': 20}
    )

    # Save plot to a file on disk
    save_dir = os.path.join('static', 'images')
    os.makedirs(save_dir, exist_ok=True)
    filename = f'plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path)

    # Save plot to a BytesIO object and encode it as a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)

    # Pass the base64 image and the file path to the template
    context = {
        'plot_url': 'data:image/png;base64,' + img_str,
        'file_path': file_path  # If you want to display or link to the saved file
    }
    return render(request, 'tumor_detection/result.html', context)

##########################Viewing Image Data##################################

def show_images_view(request):
    USER_PATH = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset"
    SEED = 111
    
    # Assuming you have these variables properly defined
    class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}

    # Load paths and labels
    train_paths, train_index = get_data_labels(os.path.join(USER_PATH, 'Training'), random_state=SEED)
    
    # Create image plot
    fig = show_images(train_paths, train_index, class_mappings, index_list=range(100, 112), im_size=350, figsize=(13, 10))

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)  # Ensure the figure is closed to free resources

    # Pass the base64 image to the template
    context = {'plot_url': 'data:image/png;base64,' + img_str}
    return render(request, 'tumor_detection/result.html', context)

def preprocess_view(request):
    USER_PATH = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset"
    train_paths, train_index = get_data_labels(USER_PATH + '/Training', random_state=SEED)
    test_paths, test_index = get_data_labels(USER_PATH + '/Testing', random_state=SEED)
    batch_size = 32
    image_dim = (168, 168)
    train_ds = get_dataset(train_paths, train_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    test_ds = get_dataset(test_paths, test_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    # Apply preprocessing to the datasets
    train_ds_preprocessed, test_ds_preprocessed = apply_preprocessing(train_ds, test_ds)

    # Context to pass to the template (this is just an example)
    context = {
        'train_ds_preprocessed': str(train_ds_preprocessed),
        'test_ds_preprocessed': str(test_ds_preprocessed)
    }
    
    # Render the template to show results (adjust as per your template structure)
    return render(request, 'tumor_detection/result.html', context)


def plot_augmented_images_view(request):
    # Load datasets and class mappings (ensure these are available)
    USER_PATH = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset"
    train_paths, train_index = get_data_labels(USER_PATH + '/Training', random_state=SEED)
    test_paths, test_index = get_data_labels(USER_PATH + '/Testing', random_state=SEED)
    batch_size = 32
    image_dim = (168, 168)
    train_ds = get_dataset(train_paths, train_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    test_ds = get_dataset(test_paths, test_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    train_ds_preprocessed, test_ds_preprocessed = apply_preprocessing(train_ds, test_ds)
    class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}


    # Filepath to save the augmented images
    filename = 'augmented_images.png'
    
    # Save the augmented images
    save_augmented_images(train_ds_preprocessed, shape=(2, 6), class_mappings=class_mappings, filename=filename)
    
    # Read the image file to send as HTTP response
    with open(filename, 'rb') as f:
        return HttpResponse(f.read(), content_type="image/png")


def preprocess_dataset_view(request):
    # Define your class mappings, image dimensions, batch size, and epochs
    class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}

    USER_PATH = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset"
    train_paths, train_index = get_data_labels(USER_PATH + '/Training', random_state=SEED)
    test_paths, test_index = get_data_labels(USER_PATH + '/Testing', random_state=SEED)
    batch_size = 32
    image_dim = (168, 168)
    train_ds = get_dataset(train_paths, train_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    test_ds = get_dataset(test_paths, test_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    epochs = 50


    # Preprocess datasets and encode labels
    train_ds_preprocessed, test_ds_preprocessed = preprocess_and_encode_labels(
        train_ds, test_ds, class_mappings, image_dim, batch_size, epochs
    )

    # Return some basic info in JSON format for verification
    data = {
        'num_classes': len(class_mappings),
        'image_shape': (image_dim[0], image_dim[1], 1),
        'epochs': epochs,
        'batch_size': batch_size
    }
    
    return JsonResponse(data)


def build_model_view(request):
    # Define the image shape and number of classes
    image_shape = (168, 168, 1)  # Adjust as per your actual image dimensions
    num_classes = 4  # Adjust based on your class mappings

    # Build the model
    model = build_model(image_shape=image_shape, num_classes=num_classes)

    # Return model information as JSON for validation
    model_info = {
        'layers': len(model.layers),
        'optimizer': str(model.optimizer.get_config())
    }

    return JsonResponse(model_info)


def train_model_view(request):
    # Define necessary parameters
    image_shape = (168, 168, 1)  # Image dimensions
    num_classes = 4  # Number of classes for classification
    epochs = 50  # Number of training epochs
    thresholds = [0.96, 0.99, 0.9935]  # Learning rate adjustment thresholds
    factor = 0.75  # Factor to reduce learning rate
    batch_size = 32  # Batch size for training

    # Path to the dataset
    USER_PATH = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset"
    train_paths, train_index = get_data_labels(USER_PATH + '/Training', random_state=SEED)
    test_paths, test_index = get_data_labels(USER_PATH + '/Testing', random_state=SEED)

    # Prepare the datasets
    train_ds = get_dataset(train_paths, train_index, image_shape[:2], n_channels=1, num_classes=num_classes, batch_size=batch_size)
    test_ds = get_dataset(test_paths, test_index, image_shape[:2], n_channels=1, num_classes=num_classes, batch_size=batch_size)

    # Apply data augmentation and preprocessing to the datasets
    train_ds_preprocessed, test_ds_preprocessed = apply_preprocessing(train_ds, test_ds)

    # Encode labels after preprocessing
    train_ds_preprocessed, test_ds_preprocessed = preprocess_and_encode_labels(
        train_ds_preprocessed, test_ds_preprocessed, {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}, image_shape[:2], batch_size, epochs
    )

    # Build the model
    model = build_model(image_shape=image_shape, num_classes=num_classes)

    # Train the model
    history = train_model(
        model=model,
        train_ds_preprocessed=train_ds_preprocessed,
        test_ds_preprocessed=test_ds_preprocessed,
        epochs=epochs,
        thresholds=thresholds,
        factor=factor
    )
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'model.keras')

    # Save the trained model without optimizer state
    model.save(MODEL_PATH)

    # Plot and save the plot_20240923_170627 history plot
    plot_path = 'training_plot.png'
    plot_training_history(history.history, save_path=plot_path)

    # Prepare a response with training metrics
    response_data = {
        'epochs_completed': len(history.epoch),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'plot_url': '/static/images/training_plot.png'
    }

    return JsonResponse(response_data)

def evaluate_model_view(request):
    # Path to the saved model
    model_path = os.path.join(settings.BASE_DIR, 'model', 'model.keras')

    class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}
    USER_PATH = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset"
    train_paths, train_index = get_data_labels(USER_PATH + '/Training', random_state=SEED)
    test_paths, test_index = get_data_labels(USER_PATH + '/Testing', random_state=SEED)
    batch_size = 32
    image_dim = (168, 168)
    train_ds = get_dataset(train_paths, train_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    test_ds = get_dataset(test_paths, test_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    epochs=50
    train_ds_preprocessed, test_ds_preprocessed = preprocess_and_encode_labels(
        train_ds, test_ds, class_mappings, image_dim, batch_size, epochs
    )

    # Assuming test_ds_preprocessed is available globally or passed in some way
    test_loss, test_acc = load_and_evaluate_model(model_path, test_ds_preprocessed)

    # Return the evaluation metrics as a JSON response
    response_data = {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }

    return JsonResponse(response_data)

########################

def plot_training_history_view(request):
    # Load or define necessary variables
    image_shape = (168, 168, 1)  # Adjust as per your actual image dimensions
    num_classes = 4  # Adjust based on your class mappings
    epochs = 50  # Number of epochs for training
    thresholds = [0.96, 0.99, 0.9935]  # Accuracy thresholds for learning rate adjustment
    factor = 0.75  # Factor by which to reduce the learning rate
    
    class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}
    USER_PATH = "C:/Users/MSI/Desktop/Stage CIMS/brain tumor/dataset"
    train_paths, train_index = get_data_labels(USER_PATH + '/Training', random_state=SEED)
    test_paths, test_index = get_data_labels(USER_PATH + '/Testing', random_state=SEED)
    batch_size = 32
    image_dim = (168, 168)
    train_ds = get_dataset(train_paths, train_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)
    test_ds = get_dataset(test_paths, test_index, image_dim, n_channels=1, num_classes=4, batch_size=batch_size)

    train_ds_preprocessed, test_ds_preprocessed = preprocess_and_encode_labels(
        train_ds, test_ds, class_mappings, image_dim, batch_size, epochs
    )

    # Build the model
    model = build_model(image_shape=image_shape, num_classes=num_classes)

    # Train the model and get the history
    history = train_model(
        model=model,
        train_ds_preprocessed=train_ds_preprocessed,
        test_ds_preprocessed=test_ds_preprocessed,
        epochs=epochs,
        thresholds=thresholds,
        factor=factor
    )

    # Define the path to save the plot
    plot_path = os.path.join(settings.STATIC_ROOT, 'images', 'training_plot.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)  # Ensure the directory exists

    # Plot and save the training history plot
    plot_training_history(history.history, save_path=plot_path)

    # Return a JSON response with the plot URL
    response_data = {
        'plot_url': 'training_plot.png'  # Provide the URL to access the plot
    }

    return JsonResponse(response_data)

# Load the model once and reuse it

model_path = os.path.join(settings.BASE_DIR, 'model', 'model.keras')
model = load_model(model_path)
def upload_image_view(request):
    predicted_label = None  # Initialize variable to store the predicted label

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        file_path = default_storage.save(uploaded_image.name, uploaded_image)

        # Load and preprocess the image
        preprocessed_image = load_and_preprocess_image(file_path)

        # Print or log the shape and sample values of the preprocessed image
        print(f"Preprocessed image shape: {preprocessed_image.shape}")
        print(f"Sample values (min, max): {preprocessed_image.min()}, {preprocessed_image.max()}")

        print("Model Summary:")
        model.summary()

        # Make prediction
        prediction = predict_image(model, file_path)
        class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}
        predicted_label = decode_predictions(prediction, class_mappings)[0]

        # Clean up the saved image file
        default_storage.delete(file_path)

    # Render the same page with the context containing the prediction result
    return render(request, 'tumor_detection/index.html', {'predicted_label': predicted_label})
