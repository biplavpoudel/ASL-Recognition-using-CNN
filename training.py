from dataset_preparation import train_preprocessed_dataset, validation_preprocessed_dataset, test_preprocessed_dataset
import tensorflow as tf
import matplotlib.pyplot as plt


# Implementing Transfer Learning from pretrained model MobileNetV2

# Rescaling pixel values from [0, 255] to [-1, 1] as:
rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)

IMG_SIZE = (200, 200)
BATCH_SIZE = 32


def rescale_img(img, label):
    img = rescale(img)
    return img, label


train_preprocessed_dataset = train_preprocessed_dataset.map(rescale_img, num_parallel_calls=tf.data.AUTOTUNE)
validation_preprocessed_dataset = validation_preprocessed_dataset.map(rescale_img, num_parallel_calls=tf.data.AUTOTUNE)


# Creating base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# This feature extractor converts each 200x200x3 image into a 5x5x1280 block of features.
image_batch, label_batch = next(iter(train_preprocessed_dataset))
feature_batch = base_model(image_batch)
print("The Feature Batch Shape is: ", feature_batch.shape)


# Freezing the Convolutional base to use as a feature extractor...
# ...as freezing prevents weights from being updated during training
# Additionally, we add a classifier on top of the frozen base and train the top-level classifier

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

inputs = tf.keras.layers.Input(shape=(200, 200, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# Compile the model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print model summary
model.summary()

initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_preprocessed_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_preprocessed_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_preprocessed_dataset)

model.save(r"model\ASL model\model.h5")

