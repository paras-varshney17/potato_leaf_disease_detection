from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
import tensorflow as tf

app = Flask(__name__)
model = load_model("potato_disease_model.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    # Ensure uploads folder exists and create a unique filename
    filename = f"{int(time.time())}_{file.filename}"
    upload_folder = os.path.join("static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Preprocess the image for prediction
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    confidences = tf.nn.softmax(predictions[0]).numpy()
    predicted_class = CLASS_NAMES[np.argmax(confidences)]
    confidence_percent = round(100 * np.max(confidences), 2)

    # Convert to relative path for HTML use
    image_url = url_for('static', filename=f"uploads/{filename}")

    return render_template("result.html",
                           prediction=predicted_class,
                           confidence=confidence_percent,
                           image_path=image_url)

if __name__ == "__main__":
    app.run(debug=True)
