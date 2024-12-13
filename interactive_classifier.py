import sys
import os
import tensorflow as tf
from utils import * 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import Callback

IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# NOTE:
# We need the correct version of tensorflow to load this model
# Maybe add a requirements.txt
base_model = base_model.load_weights("results/imagenet/trained_model/epoch_20.weights.h5")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

def classify_image(filename):
    img = image.load_img(filename, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    confidence = prediction[0][0] > 0.5
    return confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python classify.py <image_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    confidence = classify_image(filename)
    predicted_class = "Yale" if confidence > 0.5 else "Not Yale"
    print(f"Predicted class: {predicted_class} (confidence: {confidence})")
