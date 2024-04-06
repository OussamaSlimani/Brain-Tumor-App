from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
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
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        if file:
            # Save the uploaded file
            file_path = "uploads/" + file.filename
            file.save(file_path)
            
            # Preprocess the uploaded image
            img = preprocess_image(file_path)
            
            # Perform classification
            prediction = model.predict(img)
            
            # Get the predicted class
            predicted_class_index = np.argmax(prediction)
            predicted_class = list(class_mappings.keys())[predicted_class_index]
            
            return render_template('result.html', filename=file.filename, predicted_class=predicted_class)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)