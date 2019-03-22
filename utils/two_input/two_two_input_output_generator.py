import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils
from PIL import Image as pil_image
from keras import backend as K


def load_image(paths: np.ndarray, size: int,input_shape):
    """Load image from disk

    Parameters
    ----------
    db : numpy ndarray
        DB's name
    paths : np.ndarray
        Path to imahe
    size : int
        Size of image output

    Returns
    -------
    numpy ndarray

        Array of loaded and processed image
    """

   
    # images = [cv2.imread('data/{}_aligned/{}'.format(db, img_path))
    #           for (db, img_path) in zip(db, paths)]

    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]

    if input_shape[3]==1:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)

    return np.array(images, dtype='uint8')


class DataGenerator(Sequence):
    def __init__(
            self,
            model,
            paths_emotion: np.ndarray,
            paths_gender_age:np.ndarray,
            emotion_label: np.ndarray,
            age_label:np.ndarray,
            emotion_classes:int,
            age_classes:int,
            batch_size: int):
        self.paths_emotion = paths_emotion
        self.paths_gender_age = paths_gender_age
        self.emotion_label = emotion_label
        self.age_label = age_label
        self.batch_size = batch_size
        self.emotion_classes= emotion_classes
        self.age_classes = age_classes
        self.model = model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        
        paths_gender_age = self.paths_gender_age[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_gender_age = load_image( paths_gender_age, self.input_size,self.input_shape)
        X_gender_age = self.model.prep_image(batch_x_gender_age)
        del paths_gender_age, batch_x_gender_age

        # print(np.shape(X))
        batch_age = self.age_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        age = batch_age
        if self.categorical:
            age = np_utils.to_categorical(batch_age, self.age_classes)
        del batch_age


        
        paths_emotion = self.paths_emotion[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_emotion = load_image(paths_emotion, self.input_size)
        X_emotion = self.model.prep_image(batch_x_emotion)
        del paths_emotion, batch_x_emotion
        
        batch_emotion = self.emotion_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        emotion = batch_emotion
        if self.categorical:
            emotion = np_utils.to_categorical(batch_emotion, self.emotion_classes)
        del batch_emotion
        
        if (np.shape(X_emotion) == np.shape(X_gender_age)):
            # print('similar')
            X = [X_emotion,X_gender_age] 
            Y = [emotion,age]
            # Y = {'emotion_prediction': emotion,'gender_prediction':gender,'age_prediction':age}
            return X, Y

        
