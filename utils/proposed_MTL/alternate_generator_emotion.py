import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils

import tensorflow as tf
# global graph
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
            model_predict,
            paths: np.ndarray,
            emotion_label: np.ndarray,
            batch_size: int):
        self.paths = paths
        self.emotion_label = emotion_label
        self.batch_size = batch_size
        self.model = model
        self.model_predict = model_predict
        self.emotion_classes = model.emotion_classes
        self.gender_classes = model.gender_classes
        self.age_classes = model.age_classes
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.task_type = model.task_type
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
        if self.task_type == 4:
            if self.model_predict == None:
                # print('enmotion none')
                gender = np.zeros([self.batch_size,self.gender_classes])
                age = np.zeros([self.batch_size,self.age_classes])
            else:
                with graph.as_default():
                # with tf.graph().as_default():
                    gender = self.model_predict.predict(X)[1]
                    gender = np.argmax(gender, axis=1)
                    gender = np_utils.to_categorical(gender, self.gender_classes)
                    age = self.model_predict.predict(X)[2]
                    # print('predicted age',age[:3])
                    age = np.argmax(age, axis=1)
                    age = np_utils.to_categorical(age,self.age_classes)
                # print(len(self.model_predict.predict(X)))
                # print('predicted gender',gender[:3])
                    
                
        else:
            if self.model_predict == None:
                print('not none')
                gender = np.zeros([self.batch_size,self.gender_classes])
                age = np.zeros([self.batch_size,self.age_classes])
            else:
                with graph.as_default():
                # with tf.graph().as_default():
                    gender = self.model_predict.predict(X)[1]
                    # print(len(self.model_predict.predict(X)))
                    # print('predicted gender',gender[:3])
                    gender = np.argmax(gender, axis=1)
                    gender = np_utils.to_categorical(gender, self.gender_classes)

                    age = self.model_predict.predict(X)[2]
                    age = np.argmax(age, axis=1)
                    age = np_utils.to_categorical(age,self.age_classes)
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
        if np.shape(X)[0] == self.batch_size:
            return X, Y

