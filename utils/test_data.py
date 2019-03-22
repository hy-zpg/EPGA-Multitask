import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils

def load_image(paths, size):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    return np.array(images, dtype='uint8')


def test_emotion_generation(model,paths,emotion_label,emotion_classes):
    batch_x = load_image(paths, model.input_size)
    X = model.prep_image(batch_x)
    Y = np_utils.to_categorical(emotion_label,emotion_classes)
    print(np.shape(X),np.shape(Y))
    return X, Y

def test_gender_age_generation(model,paths,gender_label,age_label,gender_classes,age_classes):
    batch_x = load_image(paths, model.input_size)
    X = model.prep_image(batch_x)
    Y1 = np_utils.to_categorical(gender_label,gender_classes)
    Y2 = np_utils.to_categorical(age_label,age_classes)
    Y = [Y1,Y2]
    return X, Y