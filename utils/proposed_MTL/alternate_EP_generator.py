import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils
from keras.preprocessing import image as image_augmentation
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
            # images = [image_augmentation.random_zoom(image,zoom_range=[0.1,0.3]) for image in images]
            # images = [image_augmentation.flip_axis(image, axis=0) for image in images]
        return np.array(images, dtype='uint8')





class DataGenerator(Sequence):
    def __init__(
            self,
            model, 
            predicted_model,
            emotion_paths: np.ndarray,
            pose_paths:np.ndarray,
            emotion_label: np.ndarray,
            pose_label:np.ndarray,
            batch_size: int,
            is_emotion:bool,
            is_distilled:bool,
            is_pesudo:bool,
            is_interpolation:bool,
            pesudo_selection_threshold:int,
            interpolation_weights:int,
            is_augmentation:bool,
            pesudo_data=np.ndarray):
        self.predicted_model=predicted_model
        self.model = model
        self.emotion_paths=emotion_paths
        self.pose_paths=pose_paths
        self.emotion_label = emotion_label
        self.pose_label=pose_label
        self.batch_size = batch_size
        self.is_pesudo=is_pesudo
        self.is_distilled=is_distilled
        self.is_emotion=is_emotion
        self.is_interpolation = is_interpolation
        self.pesudo_selection_threshold=pesudo_selection_threshold
        self.interpolation_weights = interpolation_weights
        self.categorical = True 
        self.is_augmentation = is_augmentation
        self.pesudo_data=pesudo_data

    def __len__(self):
        if self.is_emotion:
            return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))
        else:
            return int(np.ceil(len(self.pose_label) / float(self.batch_size)))

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
                pose = np.zeros([self.batch_size,self.model.pose_classes])
            else:
                # with graph.as_default():
                #     pose = self.predicted_model.predict(X)[1]
                # if self.is_distilled:
                pose = self.pesudo_data[idx*self.batch_size:(idx+1)*self.batch_size]     
                # elif self.is_pesudo:
                #     pose = self.pesudo_data[idx*self.batch_size:(idx+1)*self.batch_size]
                #     # pose_index=np.argmax(pose, axis=1)
                #     # arg_pose=np_utils.to_categorical(pose_index, self.model.pose_classes)
                #     # pose = pose*arg_pose
                # # elif self.pesudo_selection_threshold>0:
                # #     pose = np.where(pose>self.pesudo_selection_threshold,pose,0)
                # elif self.is_interpolation:
                #     # pose_index=np.argmax(pose, axis=1)
                #     # arg_pose=np_utils.to_categorical(pose_index, self.model.pose_classes)
                #     # pose1 = pose*arg_pose
                #     # pose1 = np.where(pose>self.pesudo_selection_threshold,pose,0)
                #     # pose1 = self.pesudo_data[idx*self.batch_size:(idx+1)*self.batch_size]
                #     # pose = self.interpolation_weights*pose+(1-self.interpolation_weights)*pose1
                #     pose = self.pesudo_data[idx*self.batch_size:(idx+1)*self.batch_size]
            if idx==2:
                if self.predicted_model!= None:
                    print('pose',pose[0])

        else:
            paths = self.pose_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = load_image( paths, self.model.input_size,self.model.input_shape,self.is_augmentation)
            X = self.model.prep_image(batch_x)
            del paths, batch_x
            batch_pose = self.pose_label[idx * self.batch_size:(idx + 1) * self.batch_size]
            pose = batch_pose
            if self.categorical:
                pose = np_utils.to_categorical(batch_pose, self.model.pose_classes)
            del batch_pose
            
            if self.predicted_model==None:
                emotion = np.zeros([self.batch_size,self.model.emotion_classes])
            else:
                # with graph.as_default():
                #     emotion = self.predicted_model.predict(X)[0]
                # if self.is_distilled:
                emotion=self.pesudo_data[idx*self.batch_size:(idx+1)*self.batch_size]
                # elif self.is_pesudo:
                    # emotion_index=np.argmax(emotion, axis=1)
                    # arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                    # emotion = emotion*arg_emotion
                    # emotion = self.pesudo_data[idx*self.batch_size:(idx+1)*self.batch_size]
                # elif self.is_interpolation:
                    # emotion_index=np.argmax(emotion, axis=1)
                    # arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
                    # emotion1 = emotion*arg_emotion
                    # emotion1 = np.where(emotion>self.pesudo_selection_threshold,emotion,0)
                    # emotion1 = self.pesudo_data[idx*self.batch_size:(idx+1)*self.batch_size]
                    # emotion = self.interpolation_weights*emotion+(1-self.interpolation_weights)*emotion1
                    # emotion = self.pesudo_data[idx*self.batch_size:(idx+1)*self.batch_size]
            if idx==2:
                if self.predicted_model!= None:
                    print('emotion',emotion[0])

        if self.model.task_type == 11:
            Y = {'emotion_prediction': emotion,
            'pose_prediction':pose}
        return X, Y
        # if self.is_emotion:
        #     if np.shape(emotion)[0] == self.batch_size:
        #         return X, Y
        # else:
        #     if np.shape(pose)[0] == self.batch_size:
        #         return X, Y


