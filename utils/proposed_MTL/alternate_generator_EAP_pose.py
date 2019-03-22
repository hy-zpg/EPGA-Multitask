import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils

import tensorflow as tf
# global graph,model
graph = tf.get_default_graph()
np.seterr(divide='ignore',invalid='ignore')





def load_image(paths: np.ndarray, size: int,input_shape):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    if input_shape[3] ==1:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)
    return np.array(images, dtype='uint8')


class DataGenerator_pose(Sequence):
    def __init__(
            self,
            model, 
            emotion_model,
            age_model,
            paths: np.ndarray,
            pose_label: np.ndarray,
            batch_size: int,
            is_selection:bool,
            selection_threshold:float):
        self.is_selection = is_selection
        self.selection_threshold = selection_threshold
        self.paths = paths
        self.pose_label = pose_label
        self.batch_size = batch_size
        self.model = model
        self.emotion_model = emotion_model
        self.age_model = age_model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.task_type = model.task_type
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.pose_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        K = self.selection_threshold
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image(paths, self.input_size,self.input_shape)
        X = self.model.prep_image(batch_x)
        del paths, batch_x

        batch_pose = self.pose_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        pose = batch_pose
        if self.categorical:
            pose = np_utils.to_categorical(batch_pose, self.model.pose_classes)
        del batch_pose

        with graph.as_default():
        # with tf.graph().as_default():
            if self.emotion_model == None:
                emotion = np.zeros([self.batch_size,self.model.emotion_classes])
                if self.model.task_type==9:
                    age = np.zeros([self.batch_size,self.model.age_classes])
            else:
                emotion = self.emotion_model.predict(X)[0]
                if self.model.task_type==9:
                    age = self.age_model.predict(X)[2]
                if self.is_selection:
                    emotion = np.where(emotion>K,emotion,0)
                    if self.model.task_type==9:
                        age = np.where(age>K,age,0)
                    if idx==2:
                        print('selection emotion:',emotion[0])
                    
                else:                 
                    # max_e = np.max(emotion, axis=1)
                    # emotion = np.where(emotion==max_e,emotion,0)
                    # max_a = np.max(age, axis=1)
                    # age = np.where(age==max_a,age,0)
                    emotion_index=np.argmax(emotion, axis=1)
                    arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                    emotion = emotion*arg_emotion
                    if self.model.task_type==9:
                        age_index=np.argmax(age, axis=1)
                        arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                        age = age*arg_age
                    if idx==2:
                        print('no selection:',emotion[0])
                    
        if self.model.task_type==11:
            Y = {'emotion_prediction': emotion,
            'pose_prediction':pose}
            if np.shape(X)[0] == self.batch_size:
                return X, Y
        elif self.model.task_type==9:
            Y = {'emotion_prediction': emotion,
            'pose_prediction':pose,
            'age_prediction':age}
            if np.shape(X)[0] == self.batch_size:
                return X, Y
