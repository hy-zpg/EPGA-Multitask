import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend as K
import numpy as np
import glob
import pandas as pd
import random
from PIL import Image
import sys
import time
from tqdm import tqdm
import pickle
from random import shuffle
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import Progbar
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Dense, Flatten,Convolution2D
from keras.layers import Reshape, TimeDistributed, Activation,PReLU
from keras.layers.pooling import GlobalAveragePooling2D,MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.layers import Dropout, merge
from keras.regularizers import l2
import keras.losses as losses
from keras.layers import  add, Multiply, Embedding, Lambda
from keras.utils.vis_utils import plot_model 
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model 
from utils.datasets import DataManager 
from utils.two_input.ferplus_adience_confusion import ImageGenerator 
from keras.applications import vgg16
from model.vggface import Multitask_two_input_VGGFacenet
from model.mini_xception import Multitask_two_input_two_output_min_XCEPTION
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','sfew'],
                    default='ferplus',
                    help='Model to be used')
parser.add_argument('--dataset_gender_age',
                    choices=['imdb','adience','sfew'],
                    default='adience',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=300,
                    type=int,
                    help='Num of training epoch')
parser.add_argument('--image_size',
                    default=(64,64,1),
                    type=int,
                    help='Num of training epoch')
parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='Size of data batch to be used')
parser.add_argument('--num_worker',
                    default=4,
                    type=int,
                    help='Number of worker to process data')
parser.add_argument('--patience',
                    default=100,
                    type=int,
                    help='Number of traing epoch')



def split_data(ground_truth_data, split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle == True:
        shuffle(ground_truth_keys)
    length = int(split*len(ground_truth_keys))
    first = ground_truth_keys[:length]
    second = ground_truth_keys[length:2*length]
    third = ground_truth_keys[length*2:length*3]
    fourth = ground_truth_keys[length*3:length*4]
    fifth = ground_truth_keys[length*4:]

    first_train = list(set(ground_truth_keys[:]) - set(first))
    second_train = list(set(ground_truth_keys[:]) - set(second))
    third_train = list(set(ground_truth_keys[:]) - set(third))
    fourth_train = list(set(ground_truth_keys[:]) - set(fourth))
    fifth_train = list(set(ground_truth_keys[:]) - set(fifth))
    return ([first_train,first],[second_train,second],[third_train,third],[fourth_train,fourth],[fifth_train,fifth])


def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:3]:
        layer.trainable = False
    # for layer in model.layers[1:]:
    #     layer.trainable = True
    return model


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)

    args = parser.parse_args()
    MODEL = args.model
    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    NUM_WORKER = args.num_worker
    PATIENCE = args.patience
    IMAGE_SIZE = args.image_size

    model = None
    if MODEL == 'ssrnet':
        model = EmotionNetSSRNet(64, [3, 3, 3], 1.0, 1.0)
    elif MODEL == 'inceptionv3':
        model = EmotionNetInceptionV3()
    elif MODEL == 'mini_xception':
        model = Multitask_two_input_two_output_min_XCEPTION()
    elif MODEL == 'vggFace':
        model = Multitask_two_input_VGGFacenet()
        model = freeze_all_but_mid_and_top(model)
    else:
        model = EmotionNetMobileNetV2()

    losses = {
        "emotion_prediction": "categorical_crossentropy",
        "gender_prediction":"categorical_crossentropy",
        "age_prediction":"categorical_crossentropy"
    }
    metrics = {
        "emotion_prediction": "acc",
        "gender_prediction": "acc",
        "age_prediction": "acc"
    }


    

    dataset_name_E = args.dataset_emotion
    dataset_name_G = args.dataset_gender_age
    data_loader_E = DataManager(dataset_name_E)
    data_loader_G = DataManager(dataset_name_G)
    images_path_E = data_loader_E.dataset_path
    images_path_G = data_loader_G.dataset_path
    ground_truth_data_E = data_loader_E.get_data()
    ground_truth_data_G = data_loader_G.get_data()

    split_result_E = split_data(ground_truth_data_E,0.2)
    split_result_G = split_data(ground_truth_data_G,0.2)

    for i in range(5):
        train_keys_E,test_keys_E = split_result_E[i][:]
        train_keys_G,test_keys_G = split_result_G[i][:]
        
        image_generator=ImageGenerator(
                ground_truth_data_E, ground_truth_data_G,
                images_path_E,images_path_G,
                train_keys_E, test_keys_E,
                train_keys_G, test_keys_G,
                BATCH_SIZE,IMAGE_SIZE,
                grayscale = True)


        model.summary()
        train_len = np.max((len(train_keys_E), len(train_keys_G)))
        test_len = np.min((len(test_keys_E), len(test_keys_G)))

        train_generator = image_generator.flow(mode='train')
        val_generator = image_generator.flow(mode='val')



        weights_path = './train_weights/confusion/augmentation/{}'.format(MODEL)
        logs_path = './train_logs/confusion/augmentation/'
        
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        model_names = weights_path + '.{epoch:02d}-{val_emotion_prediction_acc:.2f} + {val_gender_prediction_acc:.2f} + {val_age_prediction_acc:.2f}.hdf5'
        csv_name = logs_path + '{}.log'
        checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=False)
        csvlogger=CSVLogger(csv_name)
        early_stop = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard = TensorBoard(log_dir='train_log/mixed/{}-{}.log'.format(MODEL,i),batch_size=BATCH_SIZE)
        callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]
        model.compile(optimizer='adam', loss=losses, metrics=metrics)
        model.fit_generator(
            image_generator.flow(mode='train'),
            validation_data=image_generator.flow(mode='val'),
            steps_per_epoch=int(train_len / BATCH_SIZE),
            validation_steps=int(test_len / BATCH_SIZE),
            epochs=EPOCH,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=True,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )
        del train_keys_E,train_keys_G
        del test_keys_E,test_keys_G


if __name__ == '__main__':
    main()





