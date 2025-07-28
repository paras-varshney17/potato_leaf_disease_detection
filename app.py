from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
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

    filepath = os.path.join("static", "uploads", file.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))  # ← match training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    confidences = tf.nn.softmax(predictions[0]).numpy()
    predicted_class = CLASS_NAMES[np.argmax(confidences)]
    confidence_percent = round(100 * np.max(confidences), 2)

    return render_template("result.html",
                           prediction=predicted_class,
                           confidence=confidence_percent,
                           image_path=filepath)


if __name__ == "__main__":
    app.run(debug=True)
