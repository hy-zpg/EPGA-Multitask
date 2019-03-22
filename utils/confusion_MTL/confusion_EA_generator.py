import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils
from PIL import Image as pil_image
import tensorflow as tf
# global graph
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
            paths_age:np.ndarray,
            emotion_label: np.ndarray,
            age_label:np.ndarray,
            batch_size: int,
            is_distilled:bool,
            is_pesudo:bool,
            is_interpolation:bool,
            pesudo_selection_threshold:int,
            interpolation_weights:int,
            is_augmentation:bool,
            pesudo_emotion=np.ndarray,
            pesudo_age=np.ndarray):
        self.predict_model = predict_model
        self.paths_emotion = paths_emotion
        self.paths_age = paths_age
        self.emotion_label = emotion_label
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
        self.pesudo_emotion=pesudo_emotion
        self.pesudo_age=pesudo_age
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        # return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))
        emotion_length = len(self.paths_emotion)
        age_length = len(self.paths_age)
        emotion_batch = int(self.batch_size*(emotion_length/(emotion_length+age_length)))
        age_batch = self.batch_size - emotion_batch
        length = np.min([int(emotion_length/emotion_batch),int(age_length/age_batch)])
        return length

    def __getitem__(self, idx: int):
        emotion_length = len(self.paths_emotion)
        age_length = len(self.paths_age)
        emotion_batch = int(self.batch_size*(emotion_length/(emotion_length+age_length)))
        age_batch = self.batch_size - emotion_batch
        
        paths_emotion = self.paths_emotion[idx * emotion_batch:(idx + 1) * emotion_batch]
        paths_age = self.paths_age[idx * age_batch:(idx + 1) * age_batch]
        
        batch_x_emotion = load_image(paths_emotion, self.input_size,self.input_shape,self.is_augmentation)
        X_emotion = self.model.prep_image(batch_x_emotion)
        del paths_emotion, batch_x_emotion
        batch_emotion = self.emotion_label[idx * emotion_batch:(idx + 1) * emotion_batch]
        Emotion = batch_emotion
        if self.categorical:
            Emotion = np_utils.to_categorical(batch_emotion, self.model.emotion_classes)
        del batch_emotion

        if self.predict_model==None:
            age_fake1 = np.zeros([emotion_batch,self.model.age_classes])
        else:
            with graph.as_default():
                age = self.predict_model.predict(X_emotion)[1]
            if self.is_distilled:
                age_fake1=age     
            elif self.is_pesudo:
                # age_index=np.argmax(age, axis=1)
                # arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                # age_fake1 = age*arg_age
                age_fake1 = self.pesudo_age[idx * emotion_batch:(idx + 1) * emotion_batch]
            elif self.pesudo_selection_threshold>0:
                age_fake1 = np.where(age>self.pesudo_selection_threshold,age,0)
            # elif self.is_interpolation:
            #     age1 = np.where(age>self.pesudo_selection_threshold,age,0)
            elif self.is_interpolation:
                # age1 = np.where(age>self.pesudo_selection_threshold,age,0)
                # age_index=np.argmax(age, axis=1)
                # arg_age=np_utils.to_categorical(age_index, self.model.age_classes)
                # age1 = age*arg_age
                # age1 =  self.pesudo_age[idx * emotion_batch:(idx + 1) * emotion_batch]
                # age_fake1 = self.interpolation_weights*age+(1-self.interpolation_weights)*age1
                age_fake1 = self.pesudo_age[idx * emotion_batch:(idx + 1) * emotion_batch]
        if idx==2:
            if self.predict_model!=None:
                print('age',age_fake1[0])


        batch_x_age = load_image(paths_age, self.input_size,self.input_shape,self.is_augmentation)
        X_age = self.model.prep_image(batch_x_age)
        del paths_age, batch_x_age
        batch_age = self.age_label[idx * age_batch:(idx + 1) * age_batch]
        age = batch_age
        if self.categorical:
            age = np_utils.to_categorical(batch_age, self.model.age_classes)
        del batch_age

        if self.predict_model==None:
            emotion_fake2 = np.zeros([age_batch,self.model.emotion_classes])
        else:
            with graph.as_default():
                emotion = self.predict_model.predict(X_age)[0]
            if self.is_distilled:
                emotion_fake2=emotion
            elif self.is_pesudo:
                # emotion_index=np.argmax(emotion, axis=1)
                # arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                # emotion_fake2 = emotion*arg_emotion
                emotion_fake2 = self.pesudo_emotion[idx * age_batch:(idx + 1) * age_batch]
            elif self.pesudo_selection_threshold>0:
                emotion_fake2 = np.where(emotion>self.pesudo_selection_threshold,emotion,0) 
            elif self.is_interpolation:
                # emotion1 = np.where(emotion>self.pesudo_selection_threshold,emotion,0)
                # emotion_index=np.argmax(emotion, axis=1)
                # arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                # emotion1 = emotion*arg_emotion
                # emotion1 = self.pesudo_emotion[idx * age_batch:(idx + 1) * age_batch]
                # emotion_fake2 = self.interpolation_weights*emotion+(1-self.interpolation_weights)*emotion1
                emotion_fake2 = self.pesudo_emotion[idx * age_batch:(idx + 1) * age_batch]
        if idx==2:
            if self.predict_model!= None:
                print('emotion',emotion_fake2[0])

        
            
            
        if self.model.task_type == 3:
            EMOTION = np.concatenate([Emotion,emotion_fake2],axis=0)
            age = np.concatenate([age_fake1,age],axis=0)
            X = np.concatenate([X_emotion,X_age],axis=0)
            predcition = []
            predcition.append('emotion_prediction')
            predcition.append('age_prediction')
            

            label = []
            label.append(EMOTION)
            label.append(age)

            Y=dict(zip(predcition, label))
            # if idx==2:
            #     print('emotion',np.shape(emotion),np.shape(emotion_fake2))
            #     print('age',np.shape(age_fake1),np.shape(age))
            # if np.shape(X)[0] == self.batch_size:
            if np.shape(EMOTION)[0]==np.shape(age)[0] and np.shape(age)[0]==np.shape(X)[0]:
                return X, Y