# Importing libraries I deemed necessary
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from pathlib import Path


# The training data set contains 78,000 images which are 200x200 pixels. There are 26 classes for the letters A-Z
# The test data set contains a mere 26 images, to encourage the use of real-world test images.

for dirname, _, filenames in os.walk(r'D:\ASL Recognition\Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def imgPaths(filepath):
    labels = [str(filepath[i]).split('\\')[-2]
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Labels')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffling Dataframe and resetting index
    df = df.sample(frac=1).reset_index(drop=True)

    return df


# Assign directory variables for test and train images
train_image_directory = Path(r'..\Dataset\asl_alphabets\asl_alphabets_train')
train_filepaths = list(train_image_directory.glob(r'*/*.jpg'))

test_image_directory = Path(r'..\Dataset\asl_alphabets\asl_alphabets_test')
test_filepaths = list(test_image_directory.glob(r'*/*.jpg'))

# Create dataframes for the testing and training images
train_df = imgPaths(train_filepaths)
test_df = imgPaths(test_filepaths)

df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()