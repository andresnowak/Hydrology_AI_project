import os
import numpy as np
import shutil
from PIL import Image, ImageDraw


path = r"dataset/images"
new_path = r"dataset/images_tmp"
new_path_aug = r"dataset/images_tmp_augmented"
new_path_draw = r"dataset/images_tmp_draw"
file_list = os.listdir(path)

files_tmp = np.random.choice(file_list, size=20000)


def random_image_augmentation(image, image_augmentation_rand, probability=0.5):
    if np.random.random() < probability:
        return image_augmentation_rand(image)
    return image


def image_augmentation(files, seed=None):
    import torchvision.transforms as T
    from torch.nn import Sequential
    import torch

    torch.manual_seed(seed)

    transforms = Sequential(T.ColorJitter(.4, .3, .2),
                            T.RandomHorizontalFlip(0.6),
                            T.RandomResizedCrop(
                                512, scale=(0.5, 0.7)),
                            T.RandomRotation(degrees=(0, 25)),
                            )

    for file in files:
        image = Image.open(f"{os.getcwd()}/{path}/{file}")
        image = random_image_augmentation(image, transforms, 0.8)

        image.save(f"{os.getcwd()}/{new_path_aug}/{file}")

    print("finished")


def image_painting(files):
    for file in files:
        image = Image.open(f"{os.getcwd()}/{path}/{file}")
        draw = ImageDraw.Draw(image)

        draw.polygon([(0, 0), (0, 200), (512, 150), (512, 0)], fill=(0, 0, 0)) # upper part
        draw.polygon([(249, 512), (512, 512), (512, 229), (426, 232), (422, 403)], fill=(0, 0, 0)) # bottom part

        image.save(f"{os.getcwd()}/{new_path_draw}/{file}")


def copy(files):
    for file in files:
        shutil.copy(f"{os.getcwd()}/{path}/{file}",
                    f"{os.getcwd()}/{new_path}/{file}")


#image_augmentation(file_list, seed=10)
image_painting(file_list)