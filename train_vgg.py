
import os
import sys
import keras 
import logging
from utils import * 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


args = parse_args()
# MODEL = "vgg16"
# BATCH_SIZE = 32 
# IMG_SIZE = (224,224)
MODEL = "VGG16"
BATCH_SIZE = args.batch_size
IMG_SIZE = args.img_size

logging.basicConfig(
    filename= MODEL +  '_training_log.txt',  # Log file name
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

# Define a custom callback for logging
class LoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logging.info(
            f"Epoch {epoch + 1}: accuracy: {logs.get('accuracy'):.4f}, "
            f"loss: {logs.get('loss'):.4f}, val_accuracy: {logs.get('val_accuracy'):.4f}, "
            f"val_loss: {logs.get('val_loss'):.4f}"
        )
        
        
# Prepare checkpoint path
checkpoint_path = os.path.join(MODEL, "epoch_{epoch:02d}.weights.h5")
os.makedirs(MODEL, exist_ok=True)

# Create a callback for saving checkpoints at every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch'
)

train_dataset, validation_dataset, test_dataset = download_data(BATCH_SIZE, IMG_SIZE) 
print('data downloaded')
class_names = train_dataset.class_names
train_dataset, validation_dataset, test_dataset = configure_data(train_dataset, validation_dataset, test_dataset)

#data augmentation 
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])
    
#Rescale pixel values 
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

#Base model 
# Create the base model from the pre-trained model 
IMG_SHAPE = IMG_SIZE + (3,)
# base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
#                                             include_top=False,
#                                             weights='imagenet')

    
        # Dynamically get the model function from tf.keras.applications
try:
    base_model = getattr(tf.keras.applications,MODEL)(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
except AttributeError:
    print(f"Error: {MODEL} is not a valid model type in tf.keras.applications.")
    sys.exit()

print(f"Loaded model: {MODEL}")
print(base_model.summary())

image_batch, label_batch = next(iter(train_dataset))
print("Image batch shape:", image_batch.shape)
print("Label batch shape:", label_batch.shape)
print("Image batch dtype:", image_batch.dtype)

feature_batch = base_model(image_batch)
print('feature_batch shape: ',feature_batch.shape)

#Feature Extraction 
image_batch = tf.cast(image_batch, tf.float32) / 255.0

#Freeze convolutional base 
base_model.trainable = False 

#Add a classification head 
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print('feature batch avg shape: ',feature_batch_average.shape)

#Add a dense layer to convert features into a single prediction per image 
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
prediction_batch = prediction_layer(feature_batch_average)
print('prediction batch shape: ',prediction_batch.shape)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

#compile the model 
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])


initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                epochs=initial_epochs,
                validation_data=validation_dataset, 
                callbacks = [LoggingCallback(), cp_callback])


# Log initial results
logging.info(f"Initial validation loss: {loss0:.4f}, validation accuracy: {accuracy0:.4f}")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# acc_loss_1 
plot_acc_loss(acc, val_acc, loss, val_loss, initial_epochs)
plt.savefig(MODEL + '_acc_loss_1.png')
print('plot')


#---------------Fine Tuning---------------------
#unfreeze the top layers of the model 

base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
#compile the model 
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])
model.summary()


#---------------------------Training----------------------------------
#continue training the model 
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs


history_fine = model.fit(train_dataset,
                        epochs=total_epochs,
                        initial_epoch=len(history.epoch),
                        validation_data=validation_dataset, 
                        callbacks=[LoggingCallback(), cp_callback])

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
    
#acc_loss_2
plot_acc_loss(acc, val_acc, loss, val_loss)
plt.savefig(MODEL + '_acc_loss_2.png')
print('plot num 2')

loss, accuracy = model.evaluate(test_dataset)
logging.info(f"Final test accuracy: {accuracy:.4f}, test loss: {loss:.4f}")
print('Test accuracy :', accuracy)
pred_images(test_dataset, model, class_names)
#%%
#seee some images
#generate_sample_images()