import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils


def load_image(db: np.ndarray, paths: np.ndarray, size: int):
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

    # images = [cv2.imread('data/imdb_aligned/{}'.format(img_path))
    #           for (db, img_path) in zip(db, paths)]
    images = [cv2.imread('{}'.format(img_path))
              for (db, img_path) in zip(db, paths)]

    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    return np.array(images, dtype='uint8')



# def to_vector(y, num_classes=10):
#     input_shape = y.shape
#     n = y.shape[0]
#     vector = np.zeros((n, num_classes), dtype=np.float32)
#     for i in range(n):
#         for j in range(num_classes):
#             vector[i][j] = y[i][j]
#     return vector


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
            model, db: np.ndarray,
            paths: np.ndarray,
            landmark_label: np.ndarray,
            gender_label: np.ndarray,
            smile_label:np.ndarray,
            pose_label:np.ndarray,
            batch_size: int):
        self.db = db
        self.paths = paths
        self.landmark_label = landmark_label
        self.gender_label = gender_label
        self.smile_label = smile_label
        self.pose_label = pose_label
        self.batch_size = batch_size
        self.model = model
        self.input_size = model.input_size
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.db) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        
        db = self.db[idx * self.batch_size:(idx + 1) * self.batch_size]
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image(db, paths, self.input_size)
        X = self.model.prep_image(batch_x)
        del db, paths, batch_x

        # landmark = np.zeros([self.batch_size,10])
        # batch_landmark = self.landmark_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        # landmark = batch_landmark
        # landmark = to_vector(landmark)
        # print(landmark)
        # print(landmark[0])
        # print(landmark[0][0])
        
    
        # print(landmark)
        # print(type(landmark))
        # print(np.shape(landmark))
        # print(landmark[0])
        # for i in range(self.batch_size):
        #     # print(np.array(landmark[i]))
        #     print(np.array(landmark_[i]))
        #     x = np.array(landmark_[i])
        #     print('x',type(x))
        #     landmark[i] = np.transpose(x)
        # print(np.shape(landmark[0]))
        # print(type(landmark))
        # print('type:',type(landmark))
        # print('landmark:',landmark)
        # landmark = np.zeros([self.batch_size,10])
        # for i in range(np.size(landmark_)):
            # print('ori:',landmark_[i])
            # landmark[i] = np.array(map(float,landmark_[i])).T
            # print('change:',np.array(list(landmark_[i])).T)
            # landmark[i] = np.array(list(landmark_[i])).T
        # # print('shape:',np.size(landmark[0]))
        # print('landmark:',landmark[0],type(landmark))
        # print('type1',type(landmark[31]))
        # print('size',np.size(landmark))
        # del batch_landmark

        batch_gender = self.gender_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        gender = batch_gender
        if self.categorical:
            gender = np_utils.to_categorical(batch_gender, 2)
        del batch_gender
        # print(gender)
        
        batch_smile = self.smile_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        smile = batch_smile
        if self.categorical:
            smile = np_utils.to_categorical(batch_smile, 2)
        del batch_smile

        batch_pose = self.pose_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        pose = batch_pose
        if self.categorical:
            pose = np_utils.to_categorical(batch_pose, 5)
        del batch_pose


        # Y = { 'pose_prediction':pose,
        #      'gender_prediction': gender,
        #      'smile_prediction': smile}

        Y = { 'pose_prediction':pose}


        return X, Y

        
