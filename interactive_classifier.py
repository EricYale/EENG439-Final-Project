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


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model.load_weights("results/imagenet/trained_model/epoch_20.weights.h5")

def classify_image(filename):
    img = image.load_img(filename, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    return confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify.py <image_filename>")
        sys.exit(1)

    filenames = sys.argv[1:]

    predicted_yale = 0
    predicted_non_yale = 0
    for filename in filenames:
        confidence = classify_image(filename)
        predicted_class = "Yale" if confidence > 0.90 else "Not Yale"
        print(f"{filename} :: Predicted class: {predicted_class} (confidence: {confidence})")

        if predicted_class == "Yale":
            predicted_yale += 1
        else:
            predicted_non_yale += 1
    
    print(f"{predicted_yale} Yale, {predicted_non_yale} Non-Yale")

