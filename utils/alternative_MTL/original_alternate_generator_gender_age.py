import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils

import tensorflow as tf
global graph,model
graph = tf.get_default_graph()


def load_image(paths: np.ndarray, size: int,input_shape):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    if input_shape[3] ==1:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)

    return np.array(images, dtype='uint8')

class DataGenerator_gender_age(Sequence):
    def __init__(
            self,
            model, 
            predicted_model,
            paths: np.ndarray,
            gender_label: np.ndarray,
            age_label: np.ndarray,
            batch_size: int,
            is_distilled:bool,
            is_pesudo:bool,
            is_pesudo_selection:bool):
        self.paths = paths
        self.age_label = age_label
        self.gender_label = gender_label
        self.batch_size = batch_size
        self.emotion_classes = model.emotion_classes
        self.gender_classes = model.gender_classes
        self.age_classes = model.age_classes
        self.model = model
        self.predicted_model=predicted_model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.task_type = model.task_type
        self.is_pesudo=is_pesudo
        self.is_distilled=is_distilled
        self.is_pesudo_selection=is_pesudo_selection
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image( paths, self.input_size,self.input_shape)
        X = self.model.prep_image(batch_x)
        del paths, batch_x

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

        if self.predicted_model==None:
            emotion = np.zeros([self.batch_size,self.emotion_classes])
        elif is_distilled:
            emotion = self.predicted_model.predict(X)[0]
        elif is_pesudo:
            emotion_index=np.argmax(emotion, axis=1)
            arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
            emotion = emotion*arg_emotion


        if self.task_type == 4:
            Y = {'emotion_prediction': emotion,
            'gender_prediction':gender,
            'age_prediction':age}
        elif self.task_type == 3:
            Y = {'emotion_prediction': emotion,
            'age_prediction':age}
        else:
            Y = {'emotion_prediction': emotion,
            'gender_prediction':gender,
            'age_prediction':age}
        if np.shape(age)[0] == self.batch_size:
            return X, Y        
