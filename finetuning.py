from training import *

# In the feature extraction experiment, we trained a few layers on top of an MobileNetV2 base model.
# The weights of the pre-trained network were not updated during training.

# To increase performance even further, we "fine-tune" the weights of the top layers of the pre-trained model...
# ...alongside the training of the classifier we added.
# The training process will force the weights to be tuned from generic feature maps...
# ...to features associated specifically with the dataset.

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# model = tf.keras.models.load_model(r"model/ASL_model/model.h5")
model = tf.keras.models.load_model(r"model/ASL_model/model.keras")
loss0, accuracy0 = model.evaluate(validation_preprocessed_dataset)
print("Initial loss: {:.2f}".format(loss0))
print("Initial accuracy: {:.2f}".format(accuracy0))

# As we are training a much larger model and want to re-adapt the pretrained weights,
# it is important to use a lower learning rate at this stage.
# Otherwise, our model could over-fit very quickly.

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
print("The model is compiled again for fine-tuning with low base_learning_rate")

model.summary()

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_preprocessed_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_preprocessed_dataset)
# model.save('model/ASL_model/fine_tuned_model.h5')
model.save('model/ASL_model/fine_tuned_model.keras')
print("Fine-tuned model saved successfully")

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss, accuracy = model.evaluate(test_preprocessed_dataset)
print('Test accuracy :', accuracy)
