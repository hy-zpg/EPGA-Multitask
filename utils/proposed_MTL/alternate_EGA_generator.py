import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils

import tensorflow as tf
graph = tf.get_default_graph()


# def load_image(paths: np.ndarray, size: int,input_shape):
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
    def __init__(
            self,
            model, 
            predicted_model,
            emotion_paths: np.ndarray,
            gender_age_paths:np.ndarray,
            emotion_label: np.ndarray,
            gender_label:np.ndarray,
            age_label:np.ndarray,
            batch_size: int,
            is_emotion:bool,
            is_distilled:bool,
            is_pesudo:bool,
            is_interpolation:bool,
            pesudo_selection_threshold:int,
            interpolation_weights:int,
            is_augmentation:bool):
        self.predicted_model=predicted_model
        self.model = model
        self.emotion_paths=emotion_paths
        self.gender_age_paths=gender_age_paths
        self.emotion_label = emotion_label
        self.gender_label=gender_label
        self.age_label=age_label
        self.batch_size = batch_size
        self.is_pesudo=is_pesudo
        self.is_distilled=is_distilled
        self.is_emotion=is_emotion
        self.pesudo_selection_threshold=pesudo_selection_threshold
        self.is_augmentation = is_augmentation
        self.is_interpolation = is_interpolation
        self.interpolation_weights = interpolation_weights
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        if self.is_emotion:
            return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))
        else:
            return int(np.ceil(len(self.gender_label) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        if self.is_emotion:
            paths = self.emotion_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = load_image(paths, self.model.input_size,self.model.input_shape,self.is_augmentation)
            X = self.model.prep_image(batch_x)
            del paths, batch_x
            batch_emotion = self.emotion_label[idx * self.batch_size:(idx + 1) * self.batch_size]
            emotion = batch_emotion
            if self.categorical:
                emotion = np_utils.to_categorical(batch_emotion, self.model.emotion_classes)
            del batch_emotion
            

            if self.predicted_model==None:
                gender = np.zeros([self.batch_size,self.model.gender_classes])
                age = np.zeros([self.batch_size,self.model.age_classes])
            else:
                with graph.as_default():
                    gender = self.predicted_model.predict(X)[1]
                    age = self.predicted_model.predict(X)[2]
                if self.is_distilled:
                    gender=gender
                    age=age     
                elif self.is_pesudo:
                    
                    gender_index=np.argmax(gender, axis=1)
                    arg_gender=np_utils.to_categorical(gender_index, self.model.gender_classes)
                    gender = gender*arg_gender
                    age_index=np.argmax(age, axis=1)
                    arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                    age = age*arg_age
                elif self.pesudo_selection_threshold>0:
                    gender = np.where(gender>self.pesudo_selection_threshold,gender,0)
                    age = np.where(age>self.pesudo_selection_threshold,age,0)
                elif self.is_interpolation:
                    # gender1 = np.where(gender>self.pesudo_selection_threshold,gender,0)
                    gender_index=np.argmax(gender, axis=1)
                    arg_gender=np_utils.to_categorical(gender_index, self.model.gender_classes)
                    gender1 = gender*arg_gender
                    gender = self.interpolation_weights*gender+(1-self.interpolation_weights)*gender1
                    # age1 = np.where(age>self.pesudo_selection_threshold,age,0)
                    age_index=np.argmax(age, axis=1)
                    arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                    age1 = age*arg_age
                    age = self.interpolation_weights*age+(1-self.interpolation_weights)*age1
            if idx==2:
                print('gender_age',gender[0],age[0])

        else:
            paths = self.gender_age_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = load_image( paths, self.model.input_size,self.model.input_shape,self.is_augmentation)
            X = self.model.prep_image(batch_x)
            del paths, batch_x
            batch_age = self.age_label[idx * self.batch_size:(idx + 1) * self.batch_size]
            age = batch_age
            if self.categorical:
                age = np_utils.to_categorical(batch_age, self.model.age_classes)
            del batch_age
            batch_gender = self.gender_label[idx * self.batch_size:(idx + 1) * self.batch_size]
            gender = batch_gender
            if self.categorical:
                gender = np_utils.to_categorical(batch_gender, self.model.gender_classes)
            del batch_gender

            if self.predicted_model==None:
                emotion = np.zeros([self.batch_size,self.model.emotion_classes])
            else:
                with graph.as_default():
                    emotion = self.predicted_model.predict(X)[0]
                if self.is_distilled:
                    emotion=emotion
                elif self.is_pesudo:
                    emotion_index=np.argmax(emotion, axis=1)
                    arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                    emotion = emotion*arg_emotion
                elif self.pesudo_selection_threshold>0:
                    emotion = np.where(emotion>self.pesudo_selection_threshold,emotion,0)
                elif self.is_interpolation:
                    # emotion1 = np.where(emotion>self.pesudo_selection_threshold,emotion,0)
                    emotion_index=np.argmax(emotion, axis=1)
                    arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                    emotion1 = emotion*arg_emotion
                    emotion = self.interpolation_weights*emotion+(1-self.interpolation_weights)*emotion1
            if idx==2:
                print('emotion',emotion[0])

        if self.model.task_type == 4:
            Y = {'emotion_prediction': emotion,
            'gender_prediction':gender,
            'age_prediction':age}
        elif self.model.task_type == 3:
            Y = {'emotion_prediction': emotion,
            'age_prediction':age}
        else:
            Y = {'emotion_prediction': emotion,
            'gender_prediction':gender,
            'age_prediction':age}
        if self.is_emotion:
            if np.shape(emotion)[0] == self.batch_size:
                return X, Y
        else:
            if np.shape(gender)[0] == self.batch_size:
                return X, Y


