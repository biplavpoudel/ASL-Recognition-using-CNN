from dataset_preparation import train_preprocessed_dataset, validation_preprocessed_dataset, test_preprocessed_dataset
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CSVLogger


# Implementing Transfer Learning from pretrained model MobileNetV2

# Rescaling pixel values from [0, 255] to [-1, 1] as:
rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def rescale_img(img, label):
    img = rescale(img)
    return img, label


train_preprocessed_dataset = train_preprocessed_dataset.map(rescale_img, num_parallel_calls=tf.data.AUTOTUNE)
validation_preprocessed_dataset = validation_preprocessed_dataset.map(rescale_img, num_parallel_calls=tf.data.AUTOTUNE)
print("Datasets rescaled  from [0,255] to [-1,1] range...")

# Creating base model from the pre-trained model MobileNet V2
print("Image shape is:", IMG_SIZE)
IMG_SHAPE = IMG_SIZE + (3,)
print("Input shape is:", IMG_SHAPE)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# This feature extractor converts each 200x200x3 image into a 5x5x1280 block of features.
image_batch, label_batch = next(iter(train_preprocessed_dataset))
feature_batch = base_model(image_batch)
print("The Feature Batch Shape is: ", feature_batch.shape)


# Freezing the Convolutional base to use as a feature extractor...
# ...as freezing prevents weights from being updated during training
# Additionally, we add a classifier on top of the frozen base and train the top-level classifier.

base_model.trainable = False

# Now to generate predictions from the 1280 blocks of features, we average over the 5x5 spatial locations
# GlobalAveragePooling2D layer converts the features to a single 1280-element vector per image

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print("The Average Feature Batch Shape is: ", feature_batch_average.shape)
# gives output as (32, 1280) i.e. 1 batch of 32 images with 1 feature-vector of 1280 elements


# We apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image.
# We don't need an activation function here because this prediction will be treated as a logit (raw prediction value).
# Positive numbers predict class 1, negative numbers predict class 0.

prediction_layer = tf.keras.layers.Dense(26)
prediction_batch = prediction_layer(feature_batch_average)
print("The Prediction Batch Shape is: ", prediction_batch.shape)


# Now we create a model by chaining together base_model and feature extractor layers

inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
print("Model has been created and is compiling now...")

# Learning Rate Schedule:
base_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=base_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
print("Model successfully compiled...")

# Print model summary
model.summary()

initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_preprocessed_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

csv_logger = CSVLogger("training_history.csv")
history = model.fit(train_preprocessed_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_preprocessed_dataset,
                    callbacks=[csv_logger])


# model.save(r"model\ASL_model\model.h5")
model.save(r"model\ASL_model\model.keras")
print("Model saved successfully")

# The learning curves of the training and validation accuracy/loss
# when using the MobileNetV2 base model as a fixed feature extractor.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



