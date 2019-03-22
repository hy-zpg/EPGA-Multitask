import numpy as np
import cv2
import os
import tensorflow as tf
from keras.utils import Sequence, np_utils
from keras.preprocessing import image as image_augmentation
global graph
graph = tf.get_default_graph()

# def load_image(paths: np.ndarray, size: int, input_size):
#     images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
#     images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
#     # if is_augmentation:
#         # data augmentation
#     images = image.img_to_array(images)
#     images = image.random_rotation(images,rg=10)
#     images = image.random_shift(images,wrg=0.1, hrg=0.1)
#     images = image.random_zoom(images,zoom_range=0.1)
#     images = flip_axis(images, axis=0)

#     if input_size[3] == 1:
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
            paths: np.ndarray,
            task_label: np.ndarray,
            task_classes:int,
            batch_size: int,
            is_augmentation:bool):
        self.paths = paths
        self.task_label = task_label
        self.batch_size = batch_size
        self.task_classes = task_classes
        self.model = model
        self.input_size = model.input_size
        self.input_shape = model.input_shape
        self.is_augmentation = is_augmentation
        self.is_grey = True if model.name == 'mini_xception' else False
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image( paths, self.input_size,self.input_shape,self.is_augmentation)
        X = self.model.prep_image(batch_x)
        del paths, batch_x

        # print(np.shape(X))
        # print('trainable_model:',self.model.get_weights()[0][0][0][0])
        # with graph.as_default():
        #     print(self.model.predict(X)[0])
        batch_task = self.task_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        task = batch_task
        if self.categorical:
            task = np_utils.to_categorical(batch_task, self.task_classes)
        del batch_task
        if self.model.task_type==0:
            Y = {'emotion_prediction': task}
        elif self.model.task_type == 1:
            Y = {'age_prediction': task}
        elif self.model.task_type == 5:
            Y = {'pose_prediction': task}
        elif self.model.task_type == 10:
            Y = {'gender_prediction': task}
        return X, Y


        
