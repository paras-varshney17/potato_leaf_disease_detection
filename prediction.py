import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 🔁 Load the saved model
model = tf.keras.models.load_model("potato_disease_model.h5")

# 🏷️ Define class names in the same order as in the training folder
class_names = ['Early Blight', 'Late Blight', 'Healthy']


# 📸 Load and preprocess a single image
def load_and_prep_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # same size used during training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape becomes (1, 150, 150, 3)
    img_array = img_array / 255.0  # Normalize
    return img_array


# 🔍 Predict function
def predict_disease(img_path):
    img = load_and_prep_image(img_path)
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    # Display image with predicted label
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {predicted_class}")
    plt.axis("off")
    plt.show()

    print("Prediction Probabilities:", prediction)
    print("Predicted Class:", predicted_class)


# 🧪 Test on an image
image_path ="C:\\Users\\paras\\OneDrive\\Desktop\\images.jpeg" # change this to your image
predict_disease(image_path)
