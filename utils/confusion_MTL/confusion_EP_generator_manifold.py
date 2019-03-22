import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils
from PIL import Image as pil_image
from keras.preprocessing import image as image_augmentation
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import laplacian
import tensorflow as tf
import math
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


def manifold_regularize(feature,predict,size):
    alpha = 0.99
    sigma = 0.2
    dm = cdist(feature, feature, 'euclidean')
    matrix = laplacian(dm, normed=False)
    u_matrix = np.diag(matrix)
    s_1 = np.dot(np.dot(np.transpose(predict),matrix),predict)
    s_1_value = np.sum(np.diag(s_1))/size
    manifold_value = np.full(size,s_1_value, dtype=np.float32)



    # rbf = lambda x, sigma: math.exp((-predict)/(2*(math.pow(sigma,2))))
    # vfunc = np.vectorize(rbf)
    # W = vfunc(dm, sigma)
    # np.fill_diagonal(W, 0)
    # def calculate_S(W):
    #     d = np.sum(W, axis=1)
    #     D = np.sqrt(d*d[:, np.newaxis])
    #     return np.divide(W,D,where=D!=0)
    # S = calculate_S(W)
    # F = np.dot(S, predict)*alpha + (1-alpha)*predict
    # n_iter = 400
    # for t in range(n_iter):
    #     F = np.dot(S, F)*alpha + (1-alpha)*predict
    # print('s_1_value',np.shape(manifold_value))
    return manifold_value


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
            paths_pose:np.ndarray,
            emotion_label: np.ndarray,
            pose_label:np.ndarray,
            batch_size: int,
            is_distilled:bool,
            is_pesudo:bool,
            is_interpolation:bool,
            pesudo_selection_threshold:int,
            interpolation_weights:int,
            is_augmentation:bool,
            pesudo_emotion=np.ndarray,
            pesudo_pose=np.ndarray):
        self.predict_model = predict_model
        self.paths_emotion = paths_emotion
        self.paths_pose = paths_pose
        self.emotion_label = emotion_label
        self.pose_label = pose_label
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
        self.pesudo_pose=pesudo_pose
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        # return int(np.ceil(len(self.emotion_label) / float(self.batch_size)))
        emotion_length = len(self.paths_emotion)
        pose_length = len(self.paths_pose)
        emotion_batch = int(self.batch_size*(emotion_length/(emotion_length+pose_length)))
        pose_batch = self.batch_size - emotion_batch
        length = np.min([int(emotion_length/emotion_batch),int(pose_length/pose_batch)])
        return length

    def __getitem__(self, idx: int):
        emotion_length = len(self.paths_emotion)
        pose_length = len(self.paths_pose)
        emotion_batch = int(self.batch_size*(emotion_length/(emotion_length+pose_length)))
        pose_batch = self.batch_size - emotion_batch
        
        paths_emotion = self.paths_emotion[idx * emotion_batch:(idx + 1) * emotion_batch]
        paths_pose = self.paths_pose[idx * pose_batch:(idx + 1) * pose_batch]
        
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
            # with graph.as_default():
            #     pose = self.predict_model.predict(X_emotion)[1]
            # if self.is_distilled:
            #     pose_fake1=pose     
            # elif self.is_pesudo:
            #     # pose_index=np.argmax(pose, axis=1)
            #     # arg_pose=np_utils.to_categorical(pose_index, self.model.pose_classes)
            #     # pose_fake1 = pose*arg_pose
            #     pose_fake1 = self.pesudo_pose[idx * emotion_batch:(idx + 1) * emotion_batch]
            # elif self.pesudo_selection_threshold>0:
            #     pose_fake1 = np.where(pose>self.pesudo_selection_threshold,pose,0)
            # # elif self.is_interpolation:
            # #     pose1 = np.where(pose>self.pesudo_selection_threshold,pose,0)
            # elif self.is_interpolation:
                # pose1 = np.where(pose>self.pesudo_selection_threshold,pose,0)
                # pose_index=np.argmax(pose, axis=1)
                # arg_pose=np_utils.to_categorical(pose_index, self.model.pose_classes)
                # pose1 = pose*arg_pose
                # pose1 =  self.pesudo_pose[idx * emotion_batch:(idx + 1) * emotion_batch]
                # pose_fake1 = self.interpolation_weights*pose+(1-self.interpolation_weights)*pose1
        else:
            pose_fake1 = self.pesudo_pose[idx * emotion_batch:(idx + 1) * emotion_batch]
        if idx==2:
            if self.predict_model!=None:
                print('con_pose',pose_fake1[0])


        batch_x_pose = load_image(paths_pose, self.input_size,self.input_shape,self.is_augmentation)
        X_pose = self.model.prep_image(batch_x_pose)
        del paths_pose, batch_x_pose
        batch_pose = self.pose_label[idx * pose_batch:(idx + 1) * pose_batch]
        Pose = batch_pose
        if self.categorical:
            Pose = np_utils.to_categorical(batch_pose, self.model.pose_classes)
        del batch_pose

        if self.predict_model==None:
            emotion_fake2 = np.zeros([pose_batch,self.model.emotion_classes])
        else:
            # with graph.as_default():
            #     emotion = self.predict_model.predict(X_pose)[0]
            # if self.is_distilled:
            #     emotion_fake2=emotion
            # elif self.is_pesudo:
            #     # emotion_index=np.argmax(emotion, axis=1)
            #     # arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
            #     # emotion_fake2 = emotion*arg_emotion
            #     emotion_fake2 = self.pesudo_emotion[idx * pose_batch:(idx + 1) * pose_batch]
            # elif self.pesudo_selection_threshold>0:
            #     emotion_fake2 = np.where(emotion>self.pesudo_selection_threshold,emotion,0) 
            # elif self.is_interpolation:
            #     # emotion1 = np.where(emotion>self.pesudo_selection_threshold,emotion,0)
            #     # emotion_index=np.argmax(emotion, axis=1)
            #     # arg_emotion=np_utils.to_categorical(emotion_index, self.model.emotion_classes)
            #     # emotion1 = emotion*arg_emotion
            #     # emotion1 = self.pesudo_emotion[idx * pose_batch:(idx + 1) * pose_batch]
            #     # emotion_fake2 = self.interpolation_weights*emotion+(1-self.interpolation_weights)*emotion1
            emotion_fake2 = self.pesudo_emotion[idx * pose_batch:(idx + 1) * pose_batch]
        if idx==2:
            if self.predict_model!= None:
                print('con_emotion',emotion_fake2[0])


        


        
            
            
        if self.model.task_type == 11:

            EMOTION = np.concatenate([Emotion,emotion_fake2],axis=0)
            POSE = np.concatenate([pose_fake1,Pose],axis=0)
            X = np.concatenate([X_emotion,X_pose],axis=0)
            with graph.as_default():
                emotion_feature = self.model.predict(X)[2]
                emotion_predict = self.model.predict(X)[0]
                pose_feature = self.model.predict(X)[3]
                pose_predict = self.model.predict(X)[1]
            manifold_emotion = manifold_regularize(emotion_feature,emotion_predict,self.batch_size)
            manifold_pose = manifold_regularize(pose_feature,pose_predict,self.batch_size)


            predcition = []
            predcition.append('emotion_prediction')
            predcition.append('pose_prediction')
            predcition.append('manifold_emotion')
            predcition.append('manifold_pose')
            

            label = []
            label.append(EMOTION)
            label.append(POSE)
            label.append(manifold_emotion)
            label.append(manifold_pose)

            Y=dict(zip(predcition, label))
            # if idx==2:
            #     print('emotion',np.shape(emotion),np.shape(emotion_fake2))
            #     print('pose',np.shape(pose_fake1),np.shape(pose))
            # if np.shape(X)[0] == self.batch_size:
            if np.shape(EMOTION)[0]==np.shape(POSE)[0] and np.shape(POSE)[0]==np.shape(X)[0]:
                return X, Y