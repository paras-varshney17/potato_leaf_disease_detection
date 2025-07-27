import tensorflow as tf
import tensorflow.keras as keras

data_dir="C:\\Users\\paras\\Downloads\\archive\\PlantVillage"  # your folder with class subfolders

# 1️⃣ Load training set (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,   # keep 20% aside
    subset="training",
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

# 2️⃣ Load remaining 20% (we'll split it again into val & test)
temp_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150 , 150),
    batch_size=32
)


# 3️⃣ Split temp_ds into 10% val and 10% test
val_batches = int(0.5 * len(temp_ds))
val_ds = temp_ds.take(val_batches)
test_ds = temp_ds.skip(val_batches)

base_model=keras.applications.MobileNetV2(input_shape=(150, 150, 3),include_top=False,weights='imagenet')
base_model.trainable=False

global_average = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer=keras.layers.Dense(3,activation='softmax')

model=keras.Sequential([
    base_model,
    global_average,
    prediction_layer
])

model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

model.fit(train_ds,epochs=10,validation_data=val_ds)

test_loss,test_acc = model.evaluate(test_ds)
print('Test accuracy:',test_acc)
print(temp_ds.class_names)
model.save("potato_disease.h5")
