import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils
from PIL import Image as pil_image

import tensorflow as tf
graph = tf.get_default_graph()

# def load_image(paths: np.ndarray,size: int,input_shape):
#     images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
#     images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
#     if input_shape[3] ==1:
#         images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
#         images = np.expand_dims(images, -1)
#     return np.array(images, dtype='uint8')
def load_image(paths: np.ndarray, size: int, input_size,is_augmentation:bool):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    if input_size[3] ==1:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)
        return np.array(images, dtype='uint8')
    
    else:
        # data augmentation
        if is_augmentation:
            images = [image_augmentation.img_to_array(image) for image in images]
            images = [image_augmentation.random_rotation(image,rg=10) for image in images]
            images = [image_augmentation.random_shift(image,wrg=0.1, hrg=0.1) for image in images]
            images = [image_augmentation.random_zoom(image,zoom_range=[0.1,0.3]) for image in images]
            images = [image_augmentation.flip_axis(image, axis=0) for image in images]
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
            predict_model,
            paths_emotion: np.ndarray,
            paths_pose: np.ndarray,
            paths_gender_age:np.ndarray,
            emotion_label: np.ndarray,
            pose_label: np.ndarray,
            gender_label:np.ndarray,
            age_label:np.ndarray,
            batch_size: int,
            is_distilled:bool,
            is_pesudo:bool,
            is_interpolation:bool,
            pesudo_selection_threshold:int,
            interpolation_weights:int,
            is_augmentation:bool):
        self.predict_model = predict_model
        self.paths_emotion = paths_emotion
        self.paths_pose = paths_pose
        self.paths_gender_age = paths_gender_age
        self.emotion_label = emotion_label
        self.pose_label = pose_label
        self.gender_label = gender_label
        self.age_label = age_label
        self.batch_size = batch_size
        self.is_distilled = is_distilled
        self.is_pesudo = is_pesudo
        self.is_interpolation = is_interpolation
        self.model = model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.is_augmentation = is_augmentation
        self.pesudo_selection_threshold = pesudo_selection_threshold
        self.interpolation_weights = interpolation_weights
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        # return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))
        emotion_length = len(self.paths_emotion)
        gender_age_length = len(self.paths_gender_age)
        pose_length = len(self.paths_pose)
        whole_length = emotion_length + pose_length +  gender_age_length
        emotion_batch = int(self.batch_size*(emotion_length/(whole_length)))
        pose_batch = int(self.batch_size*(pose_length/(whole_length)))
        gender_age_batch = int(self.batch_size*(gender_age_length/(whole_length)))
        length = np.min([int(emotion_length/emotion_batch),int(gender_age_length/gender_age_batch),int(pose_length/pose_batch)])
        # print('batch:',emotion_batch,pose_batch,gender_age_batch)
        return length

    def __getitem__(self, idx: int):
        emotion_length = len(self.paths_emotion)
        gender_age_length = len(self.paths_gender_age)
        pose_length = len(self.paths_pose)
        whole_length = emotion_length + pose_length +  gender_age_length
        emotion_batch = int(self.batch_size*(emotion_length/(whole_length)))
        pose_batch = int(self.batch_size*(pose_length/(whole_length)))
        gender_age_batch = int(self.batch_size*(gender_age_length/(whole_length)))
        # print('batch:',emotion_batch,pose_batch,gender_age_batch)
        
        paths_emotion = self.paths_emotion[idx * emotion_batch:(idx + 1) * emotion_batch]
        paths_pose = self.paths_pose[idx * pose_batch:(idx + 1) * pose_batch]
        paths_gender_age = self.paths_gender_age[idx * gender_age_batch:(idx + 1) * gender_age_batch]
        
        batch_x_emotion = load_image(paths_emotion, self.input_size,self.input_shape,self.is_augmentation)
        X_emotion = self.model.prep_image(batch_x_emotion)
        del paths_emotion, batch_x_emotion
        batch_emotion = self.emotion_label[idx * emotion_batch:(idx + 1) * emotion_batch]
        Emotion = batch_emotion
        if self.categorical:
            Emotion = np_utils.to_categorical(batch_emotion, self.model.emotion_classes)
        del batch_emotion

        if self.predict_model==None:
            pose_fake1 = np.zeros([emotion_batch,self.model.pose_classes])
            gender_fake1 = np.zeros([emotion_batch,self.model.gender_classes])
            age_fake1 = np.zeros([emotion_batch,self.model.age_classes])
        else:
            with graph.as_default():
                gender = self.predict_model.predict(X_emotion)[1]
                age = self.predict_model.predict(X_emotion)[2]
            if self.is_distilled:
                gender_fake1=gender
                age_fake1=age    
            elif self.is_pesudo:
                gender_index=np.argmax(gender, axis=1)
                arg_gender=np_utils.to_categorical(gender_index, self.model.gender_classes)
                gender_fake1 = gender*arg_gender
                age_index=np.argmax(age, axis=1)
                arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                age_fake1 = age*arg_age

            elif self.pesudo_selection_threshold>0:
                gender_fake1 = np.where(gender>self.pesudo_selection_threshold,gender,0)
                age_fake1 = np.where(age>self.pesudo_selection_threshold,age,0)
            elif self.is_interpolation:
                # gender1 = np.where(gender>self.pesudo_selection_threshold,gender,0)
                gender_index=np.argmax(gender, axis=1)
                arg_gender=np_utils.to_categorical(gender_index, self.model.gender_classes)
                gender1 = gender*arg_gender
                gender_fake1 = self.interpolation_weights*gender+(1-self.interpolation_weights)*gender1
                # age1 = np.where(age>self.pesudo_selection_threshold,age,0)
                age_index=np.argmax(age, axis=1)
                arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                age1 = age*arg_age
                age_fake1 = self.interpolation_weights*age+(1-self.interpolation_weights)*age1
        if idx==2:
            print('age',age_fake1[0])


        batch_x_gender_age = load_image(paths_gender_age, self.input_size,self.input_shape,self.is_augmentation)
        X_gender_age = self.model.prep_image(batch_x_gender_age)
        del paths_gender_age, batch_x_gender_age
        batch_gender = self.gender_label[idx * gender_age_batch:(idx + 1) * gender_age_batch]
        Gender = batch_gender
        batch_age = self.age_label[idx * gender_age_batch:(idx + 1) * gender_age_batch]
        Age = batch_age

        if self.categorical:
            Gender = np_utils.to_categorical(batch_gender, self.model.gender_classes)
            Age = np_utils.to_categorical(batch_age, self.model.age_classes)
        del batch_gender,batch_age

        if self.predict_model==None:
            emotion_fake2 = np.zeros([gender_age_batch,self.model.emotion_classes])
            pose_fake2 = np.zeros([gender_age_batch,self.model.pose_classes])
        else:
            with graph.as_default():
                emotion = self.predict_model.predict(X_gender_age)[0]
            if self.is_distilled:
                emotion_fake2=emotion
            elif self.is_pesudo:
                emotion_index=np.argmax(emotion, axis=1)
                arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                emotion_fake2 = emotion*arg_emotion
            elif self.pesudo_selection_threshold>0:
                emotion_fake2 = np.where(emotion>self.pesudo_selection_threshold,emotion,0) 
            elif self.is_interpolation:
                emotion1 = np.where(emotion>self.pesudo_selection_threshold,emotion,0)
                emotion_fake2 = self.interpolation_weights*emotion+(1-self.interpolation_weights)*emotion1
        if idx==2:
            print('emotion',emotion_fake2[0])


        batch_x_pose = load_image(paths_pose, self.input_size,self.input_shape,self.is_augmentation)
        X_pose = self.model.prep_image(batch_x_pose)
        del paths_pose, batch_x_pose
        batch_pose = self.pose_label[idx * pose_batch:(idx + 1) * pose_batch]
        Pose = batch_pose
        if self.categorical:
            Pose = np_utils.to_categorical(batch_pose, self.model.pose_classes)
        del batch_pose

        if self.predict_model==None:
            emotion_fake3 = np.zeros([pose_batch,self.model.emotion_classes])
            gender_fake3 = np.zeros([pose_batch,self.model.gender_classes])
            age_fake3 = np.zeros([pose_batch,self.model.age_classes])
        else:
            with graph.as_default():
                gender = self.predict_model.predict(X_emotion)[1]
                age = self.predict_model.predict(X_emotion)[2]
            if self.is_distilled:
                gender_fake1=gender
                age_fake1=age    
            elif self.is_pesudo:
                gender_index=np.argmax(gender, axis=1)
                arg_gender=np_utils.to_categorical(gender_index, self.model.gender_classes)
                gender_fake1 = gender*arg_gender
                age_index=np.argmax(age, axis=1)
                arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                age_fake1 = age*arg_age

            elif self.pesudo_selection_threshold>0:
                gender_fake1 = np.where(gender>self.pesudo_selection_threshold,gender,0)
                age_fake1 = np.where(age>self.pesudo_selection_threshold,age,0)
            elif self.is_interpolation:
                # gender1 = np.where(gender>self.pesudo_selection_threshold,gender,0)
                gender_index=np.argmax(gender, axis=1)
                arg_gender=np_utils.to_categorical(gender_index, self.model.gender_classes)
                gender1 = gender*arg_gender
                gender_fake1 = self.interpolation_weights*gender+(1-self.interpolation_weights)*gender1
                # age1 = np.where(age>self.pesudo_selection_threshold,age,0)
                age_index=np.argmax(age, axis=1)
                arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                age1 = age*arg_age
                age_fake1 = self.interpolation_weights*age+(1-self.interpolation_weights)*age1
        if idx==2:
            print('age',age_fake1[0])



        if self.model.task_type == 12:
            EMOTION = np.concatenate([Emotion,emotion_fake2,emotion_fake3],axis=0)
            POSE = np.concatenate([pose_fake1,pose_fake2,Pose],axis=0)
            GENDER = np.concatenate([gender_fake1,Gender,gender_fake3],axis=0)
            AGE = np.concatenate([age_fake1,Age,age_fake3],axis=0)
            # print(np.shape(age_fake1),np.shape(Age),np.shape(age_fake3))
            X1 = np.concatenate([X_emotion,X_gender_age],axis=0)
            X = np.concatenate([X1,X_pose],axis=0)
            # print(np.shape(X_emotion),np.shape(X_pose),np.shape(X_gender_age))
            predcition = []
            predcition.append('emotion_prediction')
            predcition.append('gender_prediction')
            predcition.append('age_prediction')
            predcition.append('pose_prediction')
            label = []
            label.append(EMOTION)
            label.append(GENDER)
            label.append(AGE)
            label.append(POSE)

            Y=dict(zip(predcition, label))
            # if idx==2:
            #     print('emotion',np.shape(emotion),np.shape(emotion_fake2))
            #     print('pose',np.shape(pose_fake1),np.shape(pose))
            # if np.shape(X)[0] == self.batch_size:
            if np.shape(EMOTION)[0]==np.shape(GENDER)[0] and np.shape(GENDER)[0]==np.shape(X)[0]:
                return X, Y