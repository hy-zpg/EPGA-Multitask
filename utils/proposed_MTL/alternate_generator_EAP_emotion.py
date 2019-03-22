import numpy as np
from numpy import *
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


class DataGenerator_emotion(Sequence):
    def __init__(
            self,
            model, 
            pose_model,
            age_model,
            paths: np.ndarray,
            emotion_label: np.ndarray,
            batch_size: int,
            is_selection:bool,
            selection_threshold:float):
        self.is_selection = is_selection
        self.selection_threshold = selection_threshold
        self.paths = paths
        self.emotion_label = emotion_label
        self.batch_size = batch_size
        self.model = model
        self.pose_model = pose_model
        self.age_model = age_model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.task_type = model.task_type
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        K = self.selection_threshold
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image(paths, self.input_size,self.input_shape)
        X = self.model.prep_image(batch_x)
        del paths, batch_x

        batch_emotion = self.emotion_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        emotion = batch_emotion
        if self.categorical:
            emotion = np_utils.to_categorical(batch_emotion, self.model.emotion_classes)
        del batch_emotion

        with graph.as_default():
        # with tf.graph().as_default():
            if self.pose_model == None:
                pose = np.zeros([self.batch_size,self.model.pose_classes])
                if self.model.task_type==9:
                    age = np.zeros([self.batch_size,self.model.age_classes])
            else:
                pose = self.pose_model.predict(X)[1]
                if self.model.task_type==9:
                    age = self.age_model.predict(X)[2]
                if idx==2:
                    print('predicted pose result:',pose[0])
                
                if self.is_selection:
                    pose = np.where(pose>K,pose,0)
                    if self.model.task_type==9: 
                        age = np.where(age>K,age,0) 
                    if idx==2:
                        print('selection pose:',pose[0])
                else:    
                    # max_p = np.max(pose, axis=1)
                    # pose = np.where(pose==max_p,pose,0)
                    # max_a = np.max(age, axis=1)
                    # age = np.where(age==max_a,age,0)
                    pose_index=np.argmax(pose, axis=1)
                    arg_pose=np_utils.to_categorical(pose_index, self.model.pose_classes)
                    pose = pose*arg_pose
                    if self.model.task_type==9:
                        age_index=np.argmax(age, axis=1)
                        arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                        age = age*arg_age
                    if idx==2:
                        print('no pose selection:',pose[0])
                        
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

