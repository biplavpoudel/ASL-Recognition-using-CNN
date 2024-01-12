# Importing libraries I deemed necessary
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# The training data set contains 78,000 images which are 200x200 pixels. There are 26 classes for the letters A-Z.
# The test data set contains a mere 26 images, to encourage the use of real-world test images.

# for dirname, _, filenames in os.walk(r'D:\ASL Recognition using CNN\Input_Images'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# def imgPaths(filepath):
#     labels = [str(filepath[i]).split('\\')[-2]
#               for i in range(len(filepath))]
#
#     filepath = pd.Series(filepath, name='Filepath').astype(str)
#     labels = pd.Series(labels, name='Labels')
#
#     # Concatenate filepaths and labels
#     df = pd.concat([filepath, labels], axis=1)
#
#     # Shuffling Dataframe and resetting index
#     df = df.sample(frac=1).reset_index(drop=True)
#     return df


# Assign directory variables for test and train images
# train_image_directory = Path(r'D:\ASL Recognition using CNN\Input_Images\asl_alphabets\asl_alphabet_train')
# # Note: always give full path for glob() function
# train_filepaths = list(train_image_directory.glob(r'*\*.jpg'))


# test_image_directory = Path(r'D:\ASL Recognition using CNN\Input_Images\asl_alphabets\asl_alphabet_test')
# test_filepaths = list(test_image_directory.glob(r'*\*.jpg'))
#
# # Create dataframes for the testing and training images
# train_df = imgPaths(train_filepaths)
# test_df = imgPaths(test_filepaths)
#
# df_unique = train_df.copy().drop_duplicates(subset=["Labels"]).reset_index()

# Now we preprocess the datasets
# tensorflow.keras.preprocessing.image.ImageDataGenerator is deprecated so have to find a new way

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


# Let's visualize the data in train_dataset
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_dataset.class_names[labels[i]])
        plt.axis('off')
# plt.show()

# Let's visualize the data in test_dataset
plt.figure(figsize=(10, 10))
for images, labels in test_dataset:
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i].numpy().decode())
        plt.axis('off')
# plt.show()
