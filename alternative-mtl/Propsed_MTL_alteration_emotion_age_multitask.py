import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from model.inceptionv3 import EmotionNetInceptionV3
from model.mobilenetv2 import MultitaskMobileNetV2
from model.vggface import AgeEmotionVGGFacenet,MultitaskVGGFacenet,AgenderNetVGGFacenet,EmotionNetVGGFacenet
from model.mini_xception import EmotionNetmin_XCEPTION
from utils.datasets import DataManager 
from utils.confusion_MTL.confusion_emotion_age_generator import DataGenerator as Test_DataGenerator
from utils.proposed_MTL.alternate_generator_emotion import DataGenerator_emotion
from utils.proposed_MTL.alternate_generator_age import DataGenerator_age
from utils.callback import DecayLearningRate
from model.models import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','sfew','kdef','expw'],
                    default='expw',
                    help='Model to be used')
parser.add_argument('--dataset_gender_age',
                    choices=['imdb','adience','sfew','fgnet'],
                    default='fgnet',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='vggFace',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=300,
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


def load_data(dataset_emotion,dataset_gender_age):
    emotion = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_emotion) )
    gender_age = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_gender_age) )

    data_emotion = emotion
    data_gedner_age = gender_age
    del emotion,gender_age
    paths_emotion = data_emotion['full_path'].values
    emotion_label = data_emotion['emotion'].values.astype('uint8')
    paths_gender_age = data_gedner_age['full_path'].values
    age_label = data_gedner_age['age'].values.astype('uint8')
    return paths_emotion, paths_gender_age, emotion_label,age_label

def mae(y_true, y_pred):
    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 70), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 70), dtype='float32') * y_true, axis=1)), axis=-1)


def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:19]:
        layer.trainable = False
    for layer in model.layers[19:]:
        layer.trainable = True
    return model

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)

    args = parser.parse_args()
    MODEL = args.model
    EPOCH = args.epoch
    PATIENCE = args.patience
    BATCH_SIZE = args.batch_size
    NUM_WORKER = args.num_worker
    EMOTION_DATASET = args.dataset_emotion
    GENDER_AGE_DATASET = args.dataset_gender_age

    if EMOTION_DATASET == 'ferplus':
        emotion_classes = 8
    else:
        emotion_classes = 7

    if GENDER_AGE_DATASET == 'imdb':
        gender_classes = 2
        age_classes = 101
    elif GENDER_AGE_DATASET == 'fgnet':
        gender_classes = 2
        age_classes = 70
    
    
    model = None
    if MODEL == 'vggFace':
        model = Net(MODEL,1,4,emotion_classes,gender_classes,age_classes)
        model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        model = Net(MODEL,1,4,emotion_classes,gender_classes,age_classes)
        MODEL = model.name

        

    if GENDER_AGE_DATASET == 'fgnet':
        losses = {
        "emotion_prediction": "categorical_crossentropy",
        "age_prediction":"categorical_crossentropy"
        }
        metrics = {
            "emotion_prediction": "acc",
            "age_prediction": mae
        }

    else:
        losses = {
            "emotion_prediction": "categorical_crossentropy",
            "gender_prediction":"categorical_crossentropy",
            "age_prediction":"categorical_crossentropy"
        }
        metrics = {
            "emotion_prediction": "acc",
            "gender_prediction": "acc",
            "age_prediction": 'acc'
        }
    if MODEL == 'ssrnet':
        losses = {
            "emotion_prediction": "mae",
        }
        metrics = {
            "emotion_prediction": "mae",
        }
        
    model.compile(optimizer='adam', loss=losses, metrics=metrics)
    model.summary()
    


    paths_emotion, paths_gender_age, emotion_label,age_label = load_data(EMOTION_DATASET,GENDER_AGE_DATASET)
    print(len(emotion_label),len(age_label))
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split_gender_age = kf.split(paths_gender_age)
    kf_split_emotion = kf.split(paths_emotion)
    emotion_kf = [[emotion_train_idx,emotion_test_idx] for emotion_train_idx,emotion_test_idx in kf_split_emotion]
    gender_age_kf = [[gender_age_train_idx,gender_age_test_idx] for gender_age_train_idx,gender_age_test_idx in kf_split_gender_age]
    emotion_train_idx,emotion_test_idx = emotion_kf[0]
    gender_age_train_idx,gender_age_test_idx = gender_age_kf[0]
    
    train_emotion_paths = paths_emotion[emotion_train_idx]
    train_emotion = emotion_label[emotion_train_idx]
    test_emotion_paths = paths_emotion[emotion_test_idx]
    test_emotion = emotion_label[emotion_test_idx]

    train_gender_age_paths = paths_gender_age[gender_age_train_idx]
    train_age = age_label[gender_age_train_idx]
    test_gender_age_paths = paths_gender_age[gender_age_test_idx]
    test_age = age_label[gender_age_test_idx]

    


    
    for epoch in  range(EPOCH):
        logs_path_E = './train_log/Proposed_MTL/emotion/{}-{}/{}'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
        weights_path_E = './train_weights/Proposed_MTL/emotion/{}-{}/{}'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
        if not os.path.exists(weights_path_E):
            os.makedirs(weights_path_E)
        if not os.path.exists(logs_path_E):
            os.makedirs(logs_path_E)
        model_names_E = weights_path_E + '{}.hdf5'.format(epoch)
        csv_name_E = logs_path_E + 'emotion.log'
        board_name_E = logs_path_E 
        checkpoint_E = ModelCheckpoint(model_names_E, verbose=1,save_weights_only = True,save_best_only=True)
        csvlogger_E=CSVLogger(csv_name_E)
        early_stop_E = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr_E = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard_E = TensorBoard(log_dir=board_name_E,batch_size=BATCH_SIZE)
        callbacks_E = [checkpoint_E,csvlogger_E,early_stop_E,reduce_lr_E,tensorboard_E]
        
       
        
        if epoch!=0:
            middle_gender_age_weights_path  = weights_path_G + '{}.hdf5'.format(epoch-1)
            middle_gender_age.load_weights(middle_gender_age_weights_path)
        model.fit_generator(
            DataGenerator_emotion(model, middle_gender_age,train_emotion_paths,train_emotion, emotion_classes,BATCH_SIZE),
            validation_data=Test_DataGenerator(model,  test_emotion_paths, test_gender_age_paths,  test_emotion,test_age, emotion_classes,age_classes,BATCH_SIZE),
            epochs=1,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks_E
        )
        print('{}_emotion_train_finished'.format(epoch))

        

        logs_path_G = './train_log/Proposed_MTL/gender_age/{}-{}/{}'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
        weights_path_G = './train_weights/Proposed_MTL/gender_age/{}-{}/{}'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
        if not os.path.exists(weights_path_G):
            os.makedirs(weights_path_G)
        if not os.path.exists(logs_path_G):
            os.makedirs(logs_path_G)
        model_names_G = weights_path_G + '{}.hdf5'.format(epoch)
        csv_name_G = logs_path_G + 'gender_age.log'
        board_name_G = logs_path_G 
        checkpoint_G = ModelCheckpoint(model_names_G, verbose=1,save_weights_only = True,save_best_only=False)
        csvlogger_G=CSVLogger(csv_name_G)
        early_stop_G = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr_G = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard_G = TensorBoard(log_dir=board_name_G,batch_size=BATCH_SIZE)
        callbacks_G = [checkpoint_G,csvlogger_G,early_stop_G,reduce_lr_G,tensorboard_G] 
        
        
        middle_emotion_weights_path = weights_path_E + '{}.hdf5'.format(epoch)
        middle_emotion.load_weights(middle_emotion_weights_path)

        
        model.fit_generator(
            DataGenerator_age(model, middle_emotion,train_gender_age_paths, train_age,age_classes,BATCH_SIZE),
            validation_data=Test_DataGenerator(model,  test_emotion_paths, test_gender_age_paths,  test_emotion,test_age, emotion_classes,age_classes,BATCH_SIZE),
            epochs=1,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks_G
        )
        print('{}_gender_age_train_finished'.format(epoch))


if __name__ == '__main__':
    main()
