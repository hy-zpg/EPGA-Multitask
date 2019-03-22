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


class DataGenerator_age(Sequence):
    def __init__(
            self,
            model, 
            emotion_model,
            pose_model,
            paths: np.ndarray,
            age_label: np.ndarray,
            batch_size: int,
            is_selection:bool,
            selection_threshold:float):
        self.is_selection = is_selection
        self.selection_threshold = selection_threshold
        self.paths = paths
        self.age_label = age_label
        self.batch_size = batch_size
        self.model = model
        self.emotion_model = emotion_model
        self.pose_model = pose_model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.task_type = model.task_type
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.age_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        K = self.selection_threshold
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image(paths, self.input_size,self.input_shape)
        X = self.model.prep_image(batch_x)
        del paths, batch_x

        batch_age = self.age_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        age = batch_age 
        if self.categorical:
            age = np_utils.to_categorical(batch_age, self.model.age_classes)
        del batch_age

        with graph.as_default():
        # with tf.graph().as_default():
            if self.emotion_model == None:
                emotion = np.zeros([self.batch_size,self.model.emotion_classes])
                pose = np.zeros([self.batch_size,self.model.pose_classes])
            else:
                emotion = self.emotion_model.predict(X)[0]
                pose = self.emotion_model.predict(X)[1]
                if self.is_selection:
                    emotion=np.where(emotion>K,emotion,0)
                    pose=np.where(pose>K,pose,0)
                    if idx==2:
                        print('selection emotion pose:',emotion[0],pose[0])
                else:
                    # max_e = np.max(emotion, axis=1)
                    # emotion = np.where(emotion==max_e,emotion,0)
                    # max_p = np.max(pose, axis=1)
                    # pose = np.where(pose==max_p,pose,0)
                    emotion_index=np.argmax(emotion, axis=1)
                    arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                    emotion = emotion*arg_emotion
                    
                    pose_index=np.argmax(pose, axis=1)
                    arg_pose=np_utils.to_categorical(pose_index, self.model.pose_classes)
                    pose = pose*arg_pose

                    
                    if idx==2:
                        print('no selection emotion pose:',emotion[0],pose[0])
                    
        
        Y = {'emotion_prediction': emotion,
        'pose_prediction':pose,
        'age_prediction':age}
        if np.shape(X)[0] == self.batch_size:
            return X, Y

