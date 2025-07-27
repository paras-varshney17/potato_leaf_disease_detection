import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Load datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    'C:\\Users\\paras\\OneDrive\\Desktop\\dataset\\train', batch_size=32)
validation_data = tf.keras.utils.image_dataset_from_directory(
    'C:\\Users\\paras\\OneDrive\\Desktop\\dataset\\valid', batch_size=32)
test_data = tf.keras.utils.image_dataset_from_directory(
    'C:\\Users\\paras\\OneDrive\\Desktop\\dataset\\test', batch_size=32)


# Preprocessing
def format(image, label):
    image = tf.cast(image, tf.float32) / 127.5 - 1
    image = tf.image.resize(image, [160, 160])
    return image, label


AUTOTUNE = tf.data.AUTOTUNE
train = train_data.map(format).shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
validation = validation_data.map(format).cache().prefetch(buffer_size=AUTOTUNE)
test = test_data.map(format).cache().prefetch(buffer_size=AUTOTUNE)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(53, activation='softmax')  # Final classification layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train
model.fit(train, validation_data=validation, epochs=15)

# Evaluate
test_loss, test_acc = model.evaluate(test)
print("Test Accuracy:", test_acc)
