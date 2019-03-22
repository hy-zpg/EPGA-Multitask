import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils

import tensorflow as tf
graph = tf.get_default_graph()


def load_image(paths: np.ndarray, size: int,input_shape):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    if input_shape[3] ==1:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)
    return np.array(images, dtype='uint8')


class DataGenerator_emotion(Sequence):
    def __init__(
            self,
            model, 
            predicted_model,
            paths: np.ndarray,
            emotion_label: np.ndarray,
            batch_size: int,
            is_distilled:bool,
            is_pesudo:bool,
            is_pesudo_selection:bool):
        self.paths = paths
        self.emotion_label = emotion_label
        self.batch_size = batch_size
        self.predicted_model=predicted_model
        self.model = model
        self.emotion_classes = model.emotion_classes
        self.gender_classes = model.gender_classes
        self.age_classes = model.age_classes
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.task_type = model.task_type
        self.is_pesudo=is_pesudo
        self.is_distilled=is_distilled
        self.is_pesudo_selection=is_pesudo_selection
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image(paths, self.input_size,self.input_shape)
        X = self.model.prep_image(batch_x)
        del paths, batch_x

        batch_emotion = self.emotion_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        emotion = batch_emotion
        if self.categorical:
            emotion = np_utils.to_categorical(batch_emotion, self.emotion_classes)
        del batch_emotion
        
        if self.predicted_model==None:
            gender = np.zeros([self.batch_size,self.gender_classes])
            age = np.zeros([self.batch_size,self.age_classes])
        elif is_distilled:
            gender = self.predicted_model.predict(X)[1]
            age = self.predicted_model.predict(X)[2]
        elif is_pesudo:
            gender_index=np.argmax(gender, axis=1)
            arg_gender=np_utils.to_categorical(gender_index, self.model.gender_classes)
            gender = gender*arg_gender

            age_index=np.argmax(age, axis=1)
            arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
            age = age*arg_age

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
        if np.shape(emotion)[0] == self.batch_size:
            return X, Y

