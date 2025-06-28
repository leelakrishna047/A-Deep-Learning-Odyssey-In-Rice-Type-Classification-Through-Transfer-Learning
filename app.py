# app.py
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = 'models/mobilenet_rice.h5'
model = load_model(MODEL_PATH)

CLASS_NAMES = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'Sticky']

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    preds = model.predict(img_array)
    return CLASS_NAMES[np.argmax(preds)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('static/uploads', filename)
            file.save(filepath)
            prediction = model_predict(filepath, model)
            return render_template('index.html', prediction=prediction, img_path=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
