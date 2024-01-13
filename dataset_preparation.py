# Importing libraries I deemed necessary
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import mediapipe as mp
import cv2

# The training data set contains 78,000 images which are 200x200 pixels. There are 26 classes for the letters A-Z.
# The test data set contains a mere 26 images, to encourage the use of real-world test images.

for dirname, _, filenames in os.walk(r'D:\ASL Recognition using CNN\Input_Images'):
    print("Data Loading....")
    for filename in filenames:
        print(os.path.join(dirname, filename))
    print("Data Successfully Loaded")


train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=r'D:\ASL Recognition using CNN\Input_Images\asl_alphabets\asl_alphabet_train',
    labels='inferred',
    label_mode='int',
    image_size=(200, 200),
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training'
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=r'D:\ASL Recognition using CNN\Input_Images\asl_alphabets\asl_alphabet_train',
    labels='inferred',
    label_mode='int',
    image_size=(200, 200),
    color_mode='rgb',
    batch_size=32,
    shuffle=False,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

# test_dataset = tf.keras.utils.image_dataset_from_directory(
#     directory=r'D:\ASL Recognition using CNN\Input_Images\asl_alphabets\asl_alphabet_test',
#     labels=None,
#     image_size=(200, 200),
#     color_mode='rgb',
#     batch_size=32,
#     shuffle=False,
# )

# Since the test_data didn't have subdirectories that reflected their class_names,
# I couldn't use tf.keras.utils.image_dataset_from_directory()
# So I had to create a function that extracts labels from file name and creates a dataset


def test_dataset_generator():
    test_image_directory = r'D:\ASL Recognition using CNN\Input_Images\asl_alphabets\asl_alphabet_test'
    batch_size = 26
    img_height, img_width = 200, 200

    # Get the list of test image file paths
    test_filepaths = tf.data.Dataset.list_files(os.path.join(test_image_directory, '*.jpg'))

    # Extract labels from file names
    def extract_label(file_path):
        # Assuming file names are like 'A_test.jpg' ....
        # and filepaths are like "D:\ASL Recognition using CNN\Input_Images\asl_alphabets\asl_alphabet_test\A_test.jpg"
        parts = tf.strings.split(tf.strings.split(file_path, '\\')[-1], '_')
        return parts[0]

    # Map file paths to images and labels
    def process_path(file_path):
        label = extract_label(file_path)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        return img, label

    # Create the test dataset
    test_datasets = test_filepaths.map(process_path)
    test_datasets = test_datasets.batch(batch_size)
    return test_datasets


test_dataset = test_dataset_generator()

# for images, labels in test_dataset:
#     labels = [label.numpy().decode() for label in labels]
#     print(labels)

# for images, labels in train_dataset.take(1):
#     print("Batch of Images:", images.shape)
#     print("Batch of Labels:", labels)


# Let's visualize the data in test_dataset
# plt.figure(figsize=(10, 10))
# for images, labels in test_dataset:
#     for i in range(9):
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(labels[i].numpy().decode())
#         plt.axis('off')
# plt.show()


# Creating Preprocessing Layers
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2)
])

data_rescaling = tf.keras.Sequential([
     layers.Rescaling(1. / 255)
])

# Applying Preprocessing Layers to the dataset
# We only augment the training data!
# Configuring the datasets for performance, using parallel reads and buffered prefetching ....
# ... to yield batches from disk without I/O become blocking.

AUTOTUNE = tf.data.AUTOTUNE


def preprocess(ds, augment=False):
    if augment:
        ds = ds.map(lambda image, label: (data_augmentation(image, training=True), label), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)


train_preprocessed_dataset = preprocess(train_dataset, augment=True)
validation_preprocessed_dataset = preprocess(validation_dataset)
test_preprocessed_dataset = preprocess(test_dataset)

# Creating Mediapipe Hands Landmark Layers
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


def feature_extraction(image):
    img = image.numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img)

    if results.multi_hand_landmarks:
        landmarks = [landmark.x for hand_landmarks in results.multi_hand_landmarks for landmark in hand_landmarks.landmark]
        return landmarks
    else:
        # If no hands are detected, return a placeholder
        # 21 hand landmarks of 3 coordinates each
        return [0.0] * 63


train_landmark_dataset = train_dataset.map(lambda x, y: (tf.py_function(feature_extraction, [x], tf.float32), y))
validation_landmark_dataset = validation_dataset.map(lambda x, y: (tf.py_function(feature_extraction, [x], tf.float32), y))

train_combined_dataset = tf.data.Dataset.zip((train_preprocessed_dataset, train_landmark_dataset))
validation_combined_dataset = tf.data.Dataset.zip((validation_preprocessed_dataset, validation_landmark_dataset))
