import numpy as np
import cv2
import os
import tensorflow as tf
from keras.utils import Sequence, np_utils
global graph
graph = tf.get_default_graph()

def load_image(paths: np.ndarray, size: int, input_size):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    
    if input_size[3] == 1:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)

    return np.array(images, dtype='uint8')



class DataGenerator(Sequence):
    def __init__(
            self,
            model, 
            paths: np.ndarray,
            age_label: np.ndarray,
            batch_size: int):
        self.paths = paths
        self.age_label = age_label
        self.batch_size = batch_size
        self.model = model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.is_grey = True if model.name == 'mini_xception' else False
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image( paths, self.input_size,self.input_shape)
        X = self.model.prep_image(batch_x)
        del paths, batch_x

        # print(np.shape(X))
        # print('trainable_model:',self.model.get_weights()[0][0][0][0])
        # with graph.as_default():
        #     print(self.model.predict(X)[0])
        batch_age = self.age_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        age = batch_age
        if self.categorical:
            age = np_utils.to_categorical(batch_age, self.model.age_classes)
        del batch_age

        Y = {'age_prediction': age}
        return X, Y


        
