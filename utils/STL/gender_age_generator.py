import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils


def load_image(paths: np.ndarray, size: int,input_shape):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    
    if input_shape[3]==1:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)

    return np.array(images, dtype='uint8')



class DataGenerator(Sequence):
    """
    Custom data generator inherits Keras Sequence class with multiprocessing support
    Parameters
    ----------
    model : Keras Model
        Model to be used in data preprocessing
    db : np.ndarray
        Array of db name
    paths : np.ndarray
        Array of image paths
    age_label : np.ndarray
        Array of age labels
    gender_label : np.ndarray
        Array of gender label
    batch_size : int
        Size of data generated at once
    """

    def __init__(
            self,
            model, 
            paths: np.ndarray,
            age_label: np.ndarray,
            gender_label: np.ndarray,
            batch_size: int):
        self.paths = paths
        self.age_label = age_label
        self.gender_label = gender_label
        self.batch_size = batch_size
        self.gender_classes = model.gender_classes
        self.age_classes = model.age_classes
        self.model = model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.model.input_type == 0:
            batch_x = load_image(paths, self.input_size,self.input_shape[0])
        else:
            batch_x = load_image(paths, self.input_size,self.input_shape) 
        X = self.model.prep_image(batch_x)
        del paths, batch_x

        # print(np.shape(X))
        batch_age = self.age_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        age = batch_age
        if self.categorical:
            age = np_utils.to_categorical(batch_age, self.age_classes)
        del batch_age

        batch_gender = self.gender_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        gender = batch_gender
        if self.categorical:
            gender = np_utils.to_categorical(batch_gender, self.gender_classes)
        del batch_gender
        # print(np.shape(X))
        Y = {'age_prediction': age,
             'gender_prediction': gender}
        # print(np.shape(X))
        return X, Y

        
