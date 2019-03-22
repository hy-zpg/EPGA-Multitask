import argparse
import os 
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from model.inceptionv3 import EmotionNetInceptionV3
from model.mobilenetv2 import MultitaskMobileNetV2
from model.vggface import AgeEmotionVGGFacenet
from model.mini_xception import EmotionNetmin_XCEPTION
from utils.datasets import DataManager 
from utils.confusion_MTL.confusion_emotion_age_generator import DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','sfew'],
                    default='ferplus',
                    help='Model to be used')
parser.add_argument('--dataset_gender_age',
                    choices=['imdb','adience','sfew','fgnet'],
                    default='adience',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='vggFace',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=50,
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
        age_classes = 101

    elif GENDER_AGE_DATASET == 'fgnet':
        age_classes = 70
    else:
        gender_classes = 2
        age_classes = 8



    paths_emotion, paths_gender_age, emotion_label,age_label = load_data(EMOTION_DATASET,GENDER_AGE_DATASET)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split_gender_age = kf.split(paths_gender_age)
    kf_split_emotion = kf.split(paths_emotion)

    emotion_kf = [[emotion_train_idx,emotion_test_idx] for emotion_train_idx,emotion_test_idx in kf_split_emotion]
    gender_age_kf = [[gender_age_train_idx,gender_age_test_idx] for gender_age_train_idx,gender_age_test_idx in kf_split_gender_age]

    # for emotion_train_idx,emotion_test_idx in kf_split_emotion:
    #     for gender_age_train_idx,gender_age_test_idx in kf_split_gender_age:
            # print(emotion_train_idx,emotion_test_idx,gender_age_train_idx,gender_age_test_idx)
    emotion_train_idx,emotion_test_idx = emotion_kf[0]
    gender_age_train_idx,gender_age_test_idx = gender_age_kf[0]

    print(len(emotion_train_idx),len(gender_age_train_idx))
    print(len(emotion_test_idx),len(gender_age_test_idx))


    train_emotion_paths = paths_emotion[emotion_train_idx]
    train_emotion = emotion_label[emotion_train_idx]
    test_emotion_paths = paths_emotion[emotion_test_idx]
    test_emotion = emotion_label[emotion_test_idx]

    train_gender_age_paths = paths_gender_age[gender_age_train_idx]
    train_age = age_label[gender_age_train_idx]
    test_gender_age_paths = paths_gender_age[gender_age_test_idx]
    test_age = age_label[gender_age_test_idx]


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
            "age_prediction":"categorical_crossentropy"
        }
        metrics = {
            "emotion_prediction": "acc",
            "age_prediction": "acc"
        }

    if MODEL == 'ssrnet':
        losses = {
            "emotion_prediction": "mae",
        }
        metrics = {
            "emotion_prediction": "mae",
        }


    model.summary()
    weights_path = './train_weights/confusion/{}-{}/{}'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
    logs_path = './train_log/confusion/{}-{}/{}'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
    
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    model_names = weights_path + '.{epoch:02d}-{val_emotion_prediction_acc:.2f}-{val_age_prediction_acc:.2f}.hdf5'
    csv_name = logs_path + '.log'
    board_name = logs_path 
    checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=True)
    csvlogger=CSVLogger(csv_name)
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(PATIENCE/2), verbose=1)
    tensorboard = TensorBoard(log_dir=board_name,batch_size=BATCH_SIZE)
    callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]

    if MODEL == 'ssrnet':
        callbacks = [
            # ModelCheckpoint(
            #     "train_weight/{}-{val_gender_prediction_binary_accuracy:.4f}-{val_age_prediction_mean_absolute_error:.4f}.h5".format(
            #         MODEL),
            #     verbose=1, save_best_only=True, save_weights_only=True),
            ModelCheckpoint(MODEL, 'val_loss', verbose=1,save_best_only=True),
            CSVLogger('train_log/{}-{}.log'.format(MODEL, n_fold)),
            DecayLearningRate([30, 60])]
    
    model.compile(optimizer='adam', loss=losses, metrics=metrics)
    model.fit_generator(
        DataGenerator(model, train_emotion_paths,train_gender_age_paths, train_emotion, train_age,emotion_classes,age_classes,BATCH_SIZE),
        validation_data=DataGenerator(model,  test_emotion_paths, test_gender_age_paths,  test_emotion,test_age, emotion_classes,age_classes,BATCH_SIZE),
        epochs=EPOCH,
        verbose=2,
        workers=NUM_WORKER,
        use_multiprocessing=False,
        max_queue_size=int(BATCH_SIZE * 2),
        callbacks=callbacks
    )
    del  train_emotion_paths, train_gender_age_paths,train_emotion,train_age
    del  test_emotion_paths, test_gender_age_paths,test_emotion


if __name__ == '__main__':
    main()
