import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import os
import cv2
#globbing utility.
import glob

path = "C:\\Users\\lenovo\\PycharmProjects\\face_classification\\train"

seq=iaa.Sequential([
    iaa.Fliplr(1),
    iaa.Affine(scale=2.0),
    iaa.Affine(rotate=(-5,55))
], random_order=True)

#seq=iaa.Affine(scale=0.33)  #zoom out 0.3x
for folder in os.listdir(path):
    i = 0
    for fname in os.listdir(path + '\\' + folder):
        img = imageio.imread(path + '\\' + folder + '\\' + fname)
        print('Original:')
        ia.imshow(img)
        img_aug = seq.augment_image(img)
        print('Augmented:')
        ia.imshow(img_aug)

        imageio.imwrite(os.path.join(path, path + '\\' + folder + '\\' + folder + "%06d.png" % (i,)), img_aug)
        i += 1
