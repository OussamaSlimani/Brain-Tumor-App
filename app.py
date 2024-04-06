from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os
import imghdr
import numpy as np

app = Flask(__name__)

# Load the classification model
model_path = "classification_model.keras"
model = load_model(model_path)

class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}

# Function to preprocess the image for classification
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        imagefile = request.files['imagefile']

        # Check if the file is selected
        if imagefile.filename == '':
            return render_template('index.html', error="No file selected. Please select an image file."), 400

        # Save the file securely
        image_path = os.path.join("uploads", secure_filename(imagefile.filename))
        imagefile.save(image_path)

        # Validate the image format
        if imghdr.what(image_path) not in ['jpeg', 'png', 'gif', 'bmp']:
            os.remove(image_path)  # Remove the invalid file
            return render_template('index.html', error="Invalid image format. Please upload a valid image file."), 400

        # Preprocess the uploaded image
        img = preprocess_image(image_path)

        # Perform classification
        prediction = model.predict(img)

        # Get the predicted class
        predicted_class_index = np.argmax(prediction)
        predicted_class = list(class_mappings.keys())[predicted_class_index]

        return render_template('result.html', predicted_class=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
