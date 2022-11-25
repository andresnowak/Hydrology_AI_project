import matplotlib.pyplot as plt
import cv2
from glob import glob
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os

# helper function for data visualization


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# helper function for data visualization


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = [0, 1]

    np.random.seed(0)

    def __init__(
            self,
            images_dir,
            df,
            classes=None,
            augmentation=None,
            preprocessing=None,
            shuffle=True,
    ):
        # self.ids = os.listdir(images_dir)
        if shuffle:
            df = df.iloc[np.random.permutation(len(df))]

        stage_values = df["Stage"].values
        stage_values = [[stage] for stage in stage_values]

        discharge_values = df["Discharge"].values
        discharge_values = [[discharge] for discharge in discharge_values]

        #self.stage_discharge_values = list(zip(stage_values, discharge_values))
        self.stage_discharge_values = stage_values

        time_values = df["SensorTime"].dt.month.values
        #area_values = df["RiverArea"].values
        self.time_values = [[time] for time in time_values]

        self.files = df.Filename.values
        self.images_fps = [os.path.join(images_dir, file)
                           for file in self.files]

        # convert str names to class values on masks
        self.class_values = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)

        stage_discharge_val = self.stage_discharge_values[i]

        time_val = self.time_values[i]

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image, time_val, stage_discharge_val

    def __len__(self):
        return len(self.files)


class Dataloader(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    np.random.seed(0)

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size

        input_1 = []
        input_2 = []
        output = []

        for j in range(start, stop):
            # print(self.dataset[j])
            # data.append(self.dataset[j])
            input_1.append(self.dataset[j][0])
            input_2.append(self.dataset[j][1])
            output.append(self.dataset[j][2])

        # print(np.array(input_1).shape)
        # print(np.array(input_2).shape)

        return {"input_1": np.array(input_1), "input_2": np.array(input_2)}, np.array(output)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
