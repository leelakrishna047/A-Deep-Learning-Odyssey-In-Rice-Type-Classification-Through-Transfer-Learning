from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(_name_)
model = load_model("rice.h5")
classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/details')
def details():
    return render_template("details.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files['image']
        img_path = os.path.join("static", img_file.filename)
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = model.predict(img_array)
        predicted_class = classes[np.argmax(pred)]

        return render_template("results.html", prediction=predicted_class, img_path=img_path)

if _name_ == '_main_':
    app.run(debug=True)
