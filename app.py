from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing import image as keras_image #type: ignore
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

class Drowsy:
    def __init__(self):
        self.model = load_model("vgg16_model.h5")
        self.class_names = ['DROWSY', 'NATURAL']

    def img_preprocessor(self, img):
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale the image

        prediction = self.model.predict(img_array)

        if prediction[0][0] > 0.5:
            return self.class_names[1], prediction[0]
        else:
            return self.class_names[0], 1 - prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img = Image.open(file_path)
            drowsy_detector = Drowsy()
            label, confidence = drowsy_detector.img_preprocessor(img)
            return render_template('index.html', label=label, confidence=confidence[0] * 100, uploaded=True, filename=file.filename)
    
    return render_template('index.html', uploaded=False)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
