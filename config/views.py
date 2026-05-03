from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import HttpResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the model (ensure to provide the correct path to your model)
MODEL_PATH = os.path.join(settings.BASE_DIR,'model', 'model.keras')
model = load_model(MODEL_PATH)

# Define image size and class mappings as used in your model
IMAGE_SIZE = (168, 168)
CLASS_MAPPINGS = {0: 'Glioma', 1: 'Meningioma', 2: 'Notumor', 3: 'Pituitary'}

def preprocess_image(img_path):
    """ Preprocess the uploaded image for model prediction. """
    img = image.load_img(img_path, target_size=IMAGE_SIZE, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

def upload_image(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['pdf_file']
        if uploaded_file:
            # Save the uploaded image temporarily
            file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            with default_storage.open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Preprocess the image and make prediction
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = CLASS_MAPPINGS.get(predicted_class, 'Unknown')

            # Optionally, remove the file after processing
            os.remove(file_path)

            return HttpResponse(f'Predicted class: {predicted_label}')

    return render(request, 'upload.html')
