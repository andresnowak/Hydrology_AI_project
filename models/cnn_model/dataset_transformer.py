import numpy as np
from glob import glob
import tensorflow as tf
import os


def make_dataset(path, batch_size, img_size, frames, df, seed=None, years=False, model="cnn"):
    np.random.seed(seed)

    def parse_image_generator(files_frames):
        for filenames in files_frames:
            images = []
            for filename in filenames:
                image = parse_image(filename)
                images.append(image)

            yield images

    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, [img_size, img_size])

        image = tf.cast(image, tf.float32)
        image = image / 255

        return image

    def configure_for_performance(ds):
        # Shuffle dataset every time, even if its divided by years
        #ds = ds.shuffle(buffer_size=800)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        #ds = ds.prefetch(buffer_size=10)
        ds = ds.batch(batch_size)
        ds = ds.repeat()
        return ds

    train_files = []
    val_files = []
    test_files = []

    if years:
        files = df[(df.Year >= 2012) & (df.Year <= 2016)].Filename.values
        train_files = [os.path.join(path, file) for file in files]

        files = df[(df.Year >= 2017) & (df.Year <= 2017)].Filename.values
        val_files = [os.path.join(path, file) for file in files]

        files = df[(df.Year >= 2018) & (df.Year <= 2019)].Filename.values
        test_files = [os.path.join(path, file) for file in files]

        # if the data is passed in order, the model just can't understand the problem so we use shuffle. Maybe because the model just can't go down the gradient because it gets confused, because the data being passed goes up then down then up then down, so it doesn't know what to do
        if model != "cnn/lstm":
            np.random.shuffle(train_files)
            np.random.shuffle(val_files)
            np.random.shuffle(val_files)

        print(len(train_files))
        print(len(val_files))
        print(len(test_files))
    else:
        # we have to grab the filename values from the dataframe
        filenames = glob(path + '/*')
        # make train, val and test splits of the dataset (70%, 10%, 20% split)
        split1 = int(0.7 * len(filenames))
        split2 = int(0.8 * len(filenames))

        np.random.shuffle(filenames)
        train_files = filenames[:split1]  # up to split 1 (ex 70%)
        val_files = filenames[split1:split2]  # from ex. 70% to 80%
        test_files = filenames[split2:]  # from ex. 80% until the end

    # create stage values
    stage_train_values = [df[df.Filename == file.split(
        '/')[-1]].Stage.values for file in train_files]
    stage_val_values = [df[df.Filename == file.split(
        '/')[-1]].Stage.values for file in val_files]
    stage_test_values = [df[df.Filename == file.split(
        '/')[-1]].Stage.values for file in test_files]

    # create discharge values
    discharge_train_values = [df[df.Filename == file.split(
        '/')[-1]].Discharge.values for file in train_files]
    discharge_val_values = [df[df.Filename == file.split(
        '/')[-1]].Discharge.values for file in val_files]
    discharge_test_values = [df[df.Filename == file.split(
        '/')[-1]].Discharge.values for file in test_files]

    # join stage and discharge values
    stage_discharge_train_values = [[np.squeeze(s), np.squeeze(
        d)] for s, d in zip(stage_train_values, discharge_train_values)]
    stage_discharge_val_values = [[np.squeeze(s), np.squeeze(
        d)] for s, d in zip(stage_val_values, discharge_val_values)]
    stage_discharge_test_values = [[np.squeeze(s), np.squeeze(
        d)] for s, d in zip(stage_test_values, discharge_test_values)]

    if model == "cnn/lstm":
        train_len = len(train_files)
        val_len = len(val_files)
        test_len = len(test_files)

        frames_lstm = frames - 1  # we reduce frames by 1 because we count from 0

        # prepare dataset for a cnn/lstm model (4 dimensions)
        stage_discharge_train_values = [
            [stage_discharge_train_values[i]] for i in range(frames_lstm, train_len)]
        stage_discharge_val_values = [
            [stage_discharge_val_values[i]] for i in range(frames_lstm, val_len)]
        stage_discharge_test_values = [
            [stage_discharge_test_values[i]] for i in range(frames_lstm, test_len)]

        # prepare files for a cnn/lstm model (4 dimensions)
        train_files = [[train_files[j] for j in range(
            i - frames_lstm, i + 1) if j < len(train_files)] for i in range(frames_lstm, train_len)]
        val_files = [[val_files[j] for j in range(
            i - frames_lstm, i + 1) if j < len(val_files)] for i in range(frames_lstm, val_len)]
        test_files = [[test_files[j] for j in range(
            i - frames_lstm, i + 1) if j < len(test_files)] for i in range(frames_lstm, test_len)]

        # shuffle lists
        temp = list(zip(stage_discharge_train_values, train_files))
        np.random.shuffle(temp)
        stage_discharge_train_values, train_files = zip(*temp)
        stage_discharge_train_values, train_files = list(
            stage_discharge_train_values), list(train_files)

        temp = list(zip(stage_discharge_val_values, val_files))
        np.random.shuffle(temp)
        stage_discharge_val_values, val_files = zip(*temp)
        stage_discharge_val_values, val_files = list(
            stage_discharge_val_values), list(val_files)

        temp = list(zip(stage_discharge_test_values, test_files))
        np.random.shuffle(temp)
        stage_discharge_test_values, test_files = zip(*temp)
        stage_discharge_test_values, test_files = list(
            stage_discharge_test_values), list(test_files)

    # create images dataset (train, val, test)
    images_train_ds = []
    images_val_ds = []
    images_test_ds = []
    if model == "cnn/lstm":
        images_train_ds = tf.data.Dataset.from_generator(lambda: parse_image_generator(
            train_files), output_signature=(tf.TensorSpec(shape=(frames, img_size, img_size, 3), dtype=tf.float32)))
        images_val_ds = tf.data.Dataset.from_generator(lambda: parse_image_generator(
            val_files), output_signature=(tf.TensorSpec(shape=(frames, img_size, img_size, 3), dtype=tf.float32)))
        images_test_ds = tf.data.Dataset.from_generator(lambda: parse_image_generator(
            test_files), output_signature=(tf.TensorSpec(shape=(frames, img_size, img_size, 3), dtype=tf.float32)))

    else:
        filenames_train_ds = tf.data.Dataset.from_tensor_slices(train_files)
        filenames_val_ds = tf.data.Dataset.from_tensor_slices(val_files)
        filenames_test_ds = tf.data.Dataset.from_tensor_slices(test_files)

        images_train_ds = filenames_train_ds.map(
            parse_image, num_parallel_calls=8)
        images_val_ds = filenames_val_ds.map(parse_image, num_parallel_calls=8)
        images_test_ds = filenames_test_ds.map(
            parse_image, num_parallel_calls=8)

    # create stage and discharge dataset (train, val, test)
    stage_discharge_train_ds = tf.data.Dataset.from_tensor_slices(
        stage_discharge_train_values)
    stage_discharge_val_ds = tf.data.Dataset.from_tensor_slices(
        stage_discharge_val_values)
    stage_discharge_test_ds = tf.data.Dataset.from_tensor_slices(
        stage_discharge_test_values)

    # create tensorflow dataset of images and values (train, val, test)
    train_ds = tf.data.Dataset.zip((images_train_ds, stage_discharge_train_ds))
    train_ds = configure_for_performance(train_ds)
    val_ds = tf.data.Dataset.zip((images_val_ds, stage_discharge_val_ds))
    val_ds = configure_for_performance(val_ds)
    test_ds = tf.data.Dataset.zip((images_test_ds, stage_discharge_test_ds))
    test_ds = configure_for_performance(test_ds)

    return train_ds, len(train_files), val_ds, len(val_files), test_ds, len(test_files)
