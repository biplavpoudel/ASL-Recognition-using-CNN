from dataset_preparation import train_dataset, validation_dataset, test_dataset
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

num_classes = len(validation_dataset.class_names)
print(num_classes)

model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same',  activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

model.fit(train_dataset, validation_data=validation_dataset, epochs=5)

loss, accuracy = model.evaluate(test_dataset)
print('Accuracy', accuracy)

model.save(r'D:\ASL Recognition using CNN\model\model.h5')

