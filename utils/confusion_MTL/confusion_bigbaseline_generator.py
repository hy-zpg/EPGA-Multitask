import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils
from PIL import Image as pil_image



def load_image(paths: np.ndarray,size: int,input_shape):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    if input_shape[3] ==1:
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
            paths_emotion: np.ndarray,
            paths_pose:np.ndarray,
            paths_attributes:np.ndarray,
            paths_gender:np.ndarray,
            paths_age:np.ndarray,
            emotion_label: np.ndarray,
            pose_label:np.ndarray,
            gender_label:np.ndarray,
            age_label:np.ndarray,
            attribute_label:np.ndarray,
            batch_size: int):
        self.paths_emotion = paths_emotion
        self.paths_attributes = paths_attributes
        self.paths_pose = paths_pose
        self.paths_gender = paths_gender
        self.paths_age = paths_age
        self.emotion_label = emotion_label
        self.pose_label = pose_label
        self.attribute_label = attribute_label
        self.emotion_classes = model.emotion_classes
        self.batch_size = batch_size
        self.model = model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        emotion_length = len(self.paths_emotion)
        pose_length = len(self.paths_pose)
        attribute_length = len(self.paths_attributes)
        gender_length = len(self.paths_gender)
        age_length = len(self.paths_age)
        whole_length = emotion_length+pose_length+attribute_length+gender_length+age_length

        emotion_batch = int(self.batch_size*(emotion_length/whole_length))
        pose_batch = int(self.batch_size*(pose_length/whole_length))
        gender_batch = int(self.batch_size*(gender_length/whole_length))
        age_batch = int(self.batch_size*(age_length/whole_length))
        attributes_batch = self.batch_size - emotion_batch - pose_batch - gender_batch - age_batch
        
        paths_emotion = self.paths_emotion[idx * emotion_batch:(idx + 1) * emotion_batch]
        paths_pose = self.paths_pose[idx * pose_batch:(idx + 1) * pose_batch]
        path_gender = self.paths_gender[idx * gender_batch:(idx + 1) * gender_batch]
        path_age = self.paths_age[idx * age_batch:(idx + 1) * age_batch]
        paths_attributes = self.paths_attributes[idx*attributes_batch:(idx+1)*attributes_batch]
        
        batch_x_emotion = load_image(paths_emotion, self.input_size,self.input_shape)
        X_emotion = self.model.prep_image(batch_x_emotion)
        del paths_emotion, batch_x_emotion
        batch_emotion = self.emotion_label[idx * emotion_batch:(idx + 1) * emotion_batch]
        emotion = batch_emotion
        if self.categorical:
            emotion = np_utils.to_categorical(batch_emotion, self.emotion_classes)
        del batch_emotion
        pose_fake1 = np.zeros([emotion_batch,3])
        attr_fake1=[]
        for i in range(40):
            attr_fake1.append(np.zeros([emotion_batch,2]))


        batch_x_pose = load_image(paths_pose, self.input_size,self.input_shape)
        X_pose = self.model.prep_image(batch_x_pose)
        del paths_pose, batch_x_pose
        batch_pose = self.pose_label[idx * pose_batch:(idx + 1) * pose_batch]
        pose = batch_pose
        pose = np.double(pose)*180/np.pi
        del batch_pose
        emotion_fake2 = np.zeros([pose_batch,self.emotion_classes])
        attr_fake2=[]
        for i in range(40):
            attr_fake2.append(np.zeros([pose_batch,2]))

        batch_x_attr = load_image(paths_attributes, self.input_size,self.input_shape)
        X_attr = self.model.prep_image(batch_x_attr)
        del paths_attributes, batch_x_attr
        batch_attr = self.attribute_label[idx * attributes_batch:(idx + 1) * attributes_batch]
        attr = batch_attr
        del batch_attr
        attr = np.transpose(attr)
        attrs=[]
        if self.categorical:
            for i in range(40):
                attrs.append(np_utils.to_categorical(attr[i], 2))
        emotion_fake3 = np.zeros([attributes_batch,self.emotion_classes])
        pose_fake3 = np.zeros([attributes_batch,3])


        
        

        if self.model.task_type == 7:
            EMOTION = np.concatenate([emotion,emotion_fake3],axis=0)
            ATTR = []
            for i in range(40):
                ATTR.append(np.concatenate([attr_fake1[i],attrs[i]],axis=0))
            X = np.concatenate([X_emotion,X_attr],axis=0)
            
            predcition = []
            predcition.append('emotion_prediction')
            for i in range(40):
                predcition.append('attr{}_predition'.format(i))

            label = []
            label.append(EMOTION)
            for i in range(40):
                label.append(ATTR[i])

            Y=dict(zip(predcition, label))
            if np.shape(X)[0] == self.batch_size - pose_batch:
                return X, Y
        elif self.model.task_type == 8:
            EMOTION = np.concatenate([emotion,emotion_fake2,emotion_fake3],axis=0)
            POSE = np.concatenate([pose_fake1,pose,pose_fake3])
            ATTR = []
            for i in range(40):
                ATTR.append(np.concatenate([attr_fake1[i],attr_fake2[i],attrs[i]],axis=0))

            X = np.concatenate([X_emotion,X_pose,X_attr],axis=0)
            predcition = []
            predcition.append('emotion_prediction')
            predcition.append('pose_prediction')
            for i in range(40):
                predcition.append('attr{}_predition'.format(i))

            label = []
            label.append(EMOTION)
            label.append(POSE)
            for i in range(40):
                label.append(ATTR[i])

            Y=dict(zip(predcition, label))
            if np.shape(X)[0] == self.batch_size:
                return X, Y
        
