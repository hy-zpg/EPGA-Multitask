import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils
from PIL import Image as pil_image



# def load_image(paths: np.ndarray,size: int,input_shape):
#     images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
#     images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
#     if input_shape[3] ==1:
#         images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
#         images = np.expand_dims(images, -1)
#     return np.array(images, dtype='uint8')
    

def load_image(paths: np.ndarray, size: int, input_size,is_augmentation:bool):
    if input_size[3] ==1:
        images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
        images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)
        return np.array(images, dtype='uint8')
    
    else:
        images=[]
        for img_path in paths:
            image = cv2.imread('{}'.format(img_path))
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            # data augmentation
            if is_augmentation:
                image = image_augmentation.img_to_array(image)
                image = image_augmentation.random_rotation(image,rg=10)
                image = image_augmentation.random_shift(image,wrg=0.1, hrg=0.1)
                image = image_augmentation.random_zoom(image,zoom_range=[0.1,0.3])
                image = image_augmentation.flip_axis(image, axis=0)
            images.append(image)
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
            paths_age:np.ndarray,
            emotion_label: np.ndarray,
            pose_label:np.ndarray,
            age_label:np.ndarray,
            batch_size: int,
            is_augmentation:bool):
        self.paths_emotion = paths_emotion
        self.paths_age = paths_age
        self.paths_pose = paths_pose
        self.emotion_label = emotion_label
        self.pose_label = pose_label
        self.age_label = age_label
        self.batch_size = batch_size
        self.model = model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.is_augmentation = is_augmentation
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        emotion_length = len(self.paths_emotion)
        pose_length = len(self.paths_pose)
        age_length = len(self.paths_age)

        emotion_batch = int(self.batch_size*(emotion_length/(emotion_length+pose_length+age_length)))
        pose_batch = int(self.batch_size*(pose_length/(emotion_length+pose_length+age_length)))
        age_batch = self.batch_size - emotion_batch - pose_batch
        
        paths_emotion = self.paths_emotion[idx * emotion_batch:(idx + 1) * emotion_batch]
        paths_pose = self.paths_pose[idx * pose_batch:(idx + 1) * pose_batch]
        paths_age = self.paths_age[idx*age_batch:(idx+1)*age_batch]
        
        batch_x_emotion = load_image(paths_emotion, self.input_size,self.input_shape,self.is_augmentation)
        X_emotion = self.model.prep_image(batch_x_emotion)
        del paths_emotion, batch_x_emotion
        batch_emotion = self.emotion_label[idx * emotion_batch:(idx + 1) * emotion_batch]
        emotion = batch_emotion
        if self.categorical:
            emotion = np_utils.to_categorical(batch_emotion, self.model.emotion_classes)
        del batch_emotion
        pose_fake1 = np.zeros([emotion_batch,self.model.pose_classes])
        age_fake1=np.zeros([emotion_batch,self.model.age_classes])
       

        batch_x_pose = load_image(paths_pose, self.input_size,self.input_shape,self.is_augmentation)
        X_pose = self.model.prep_image(batch_x_pose)
        del paths_pose, batch_x_pose
        batch_pose = self.pose_label[idx * pose_batch:(idx + 1) * pose_batch]
        pose = batch_pose
        if self.categorical:
            pose = np_utils.to_categorical(batch_pose, self.model.pose_classes)

        del batch_pose
        emotion_fake2 = np.zeros([pose_batch,self.model.emotion_classes])
        age_fake2= np.zeros([pose_batch,self.model.age_classes])
        
        batch_x_age = load_image(paths_age, self.input_size,self.input_shape,self.is_augmentation)
        X_age = self.model.prep_image(batch_x_age)
        del paths_age, batch_x_age
        batch_age= self.age_label[idx * age_batch:(idx + 1) * age_batch]
        age = batch_age
        if self.categorical:
            age = np_utils.to_categorical(batch_age, self.model.age_classes)
        del batch_age
        
        emotion_fake3 = np.zeros([age_batch,self.model.emotion_classes])
        pose_fake3 = np.zeros([age_batch,self.model.pose_classes])


        
        

        if self.model.task_type == 3:
            EMOTION = np.concatenate([emotion,emotion_fake3],axis=0)
            AGE = np.concatenate([age_fake1,age],axis=0)
            X = np.concatenate([X_emotion,X_age],axis=0)
            
            predcition = []
            predcition.append('emotion_prediction')
            predcition.append('age_prediction')
            

            label = []
            label.append(EMOTION)
            label.append(AGE)

            Y=dict(zip(predcition, label))
            if np.shape(X)[0] == self.batch_size - pose_batch:
                return X, Y

        elif self.model.task_type == 9:
            EMOTION = np.concatenate([emotion,emotion_fake2,emotion_fake3],axis=0)
            POSE = np.concatenate([pose_fake1,pose,pose_fake3],axis=0)
            AGE = np.concatenate([age_fake1,age_fake2,age],axis=0)

            X = np.concatenate([X_emotion,X_pose,X_age],axis=0)
            predcition = []
            predcition.append('emotion_prediction')
            predcition.append('pose_prediction')
            predcition.append('age_prediction')
            

            label = []
            label.append(EMOTION)
            label.append(POSE)
            label.append(AGE)
            
            Y=dict(zip(predcition, label))
            if np.shape(X)[0] == self.batch_size:
                return X, Y
        elif self.model.task_type == 11:
            EMOTION = np.concatenate([emotion,emotion_fake2],axis=0)
            POSE = np.concatenate([pose_fake1,pose],axis=0)

            X = np.concatenate([X_emotion,X_pose],axis=0)
            predcition = []
            predcition.append('emotion_prediction')
            predcition.append('pose_prediction')
            

            label = []
            label.append(EMOTION)
            label.append(POSE)
            
            Y=dict(zip(predcition, label))
            if np.shape(X)[0] == emotion_batch+pose_batch:
                return X, Y
        
