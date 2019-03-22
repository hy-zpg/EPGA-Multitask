import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils


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

    if input_shape[3] == 1:
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
            pose_label: np.ndarray,
            batch_size: int):
        self.paths = paths
        self.pose_label = pose_label
        self.batch_size = batch_size
        self.model = model
        self.input_size = model.input_size
        self.input_shape = model.input_shape

    def __len__(self):
        return int(np.ceil(len(self.pose_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.model.input_type == 0:
            batch_x = load_image(paths, self.input_size,self.input_shape[0])
        else:
            batch_x = load_image(paths, self.input_size,self.input_shape)


        X = self.model.prep_image(batch_x)
        del paths, batch_x


        batch_label = self.pose_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        label = batch_label
        # label = np.double(label)*180/np.pi
        del batch_label
        if len(np.shape(X)) == 4: 
            Y = {'pose_prediction': label}
            return X, Y

        
