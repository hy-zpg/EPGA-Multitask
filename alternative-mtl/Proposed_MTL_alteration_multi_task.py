import argparse
import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
# from model.inceptionv3 import EmotionNetInceptionV3
# from model.mobilenetv2 import MultitaskMobileNetV2
# from model.vggface import MultitaskVGGFacenet,AgenderNetVGGFacenet,EmotionNetVGGFacenet
# from model.mini_xception import EmotionNetmin_XCEPTION

from utils.datasets import DataManager 
from utils.proposed_MTL.alternate_generator_emotion import DataGenerator_emotion
from utils.proposed_MTL.alternate_generator_gender_age import DataGenerator_gender_age
from utils.confusion_MTL.confusion_generator import DataGenerator as Test_DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','sfew'],
                    default='ferplus',
                    help='Model to be used')
parser.add_argument('--dataset_gender_age',
                    choices=['imdb','adience','sfew',],
                    default='adience',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
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
    gender_label = data_gedner_age['gender'].values.astype('uint8')
    age_label = data_gedner_age['age'].values.astype('uint8')
    return paths_emotion, paths_gender_age, emotion_label,gender_label,age_label

def mae(y_true, y_pred):
    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_true, axis=1)), axis=-1)

def my_mae(y_true,y_pred):
    mask = K.all(K.equal(y_true, 0), axis=-1)
    mask = 1 - K.cast(mask, K.floatx())
    mae = mae(y_true,y_pred)*mask
    return K.sum(mae) / K.sum(mask)


def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:19]:
        layer.trainable = False
    for layer in model.layers[19:]:
        layer.trainable = True
    return model

def freeze_all(model):
    for layer in model.layers[:]:
        layer.trainable = False
    return model

def my_loss(y_true, y_pred):
    mask = K.all(K.equal(y_true, 0), axis=-1)
    mask = 1 - K.cast(mask, K.floatx())
    loss = (K.categorical_crossentropy(y_true, y_pred)) * mask
    return K.sum(loss) / K.sum(mask)

def my_acc(y_true, y_pred):
    mask = K.all(K.equal(y_true, 0), axis=-1)
    mask = 1 - K.cast(mask, K.floatx()) 
    acc = (K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx()))*mask
    return K.sum(acc)/K.sum(mask)



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
    else:
        gender_classes = 2
        age_classes = 8
    
    pose_classes =5
    model = None
    emotion_model = None
    age_model = None
    if MODEL == 'vggFace':
        model = Net(MODEL,1,4,emotion_classes,pose_classes,age_classes,gender_classes)
        model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        model = Net(MODEL,1,4,emotion_classes,pose_classes,age_classes,gender_classes)
        MODEL = model.name

        

    if GENDER_AGE_DATASET == 'imdb':
        losses = {
        "emotion_prediction": my_loss,
        "gender_prediction":my_loss,
        "age_prediction":my_loss
        }
        metrics = {
            "emotion_prediction": my_acc,
            "gender_prediction": my_acc,
            "age_prediction": my_mae
        }

    else:
        losses = {
            "emotion_prediction": my_loss,
            "gender_prediction":my_loss,
            "age_prediction":my_loss
        }
        metrics = {
            "emotion_prediction": my_acc,
            "gender_prediction": my_acc,
            "age_prediction": my_acc
        }
    if MODEL == 'ssrnet':
        losses = {
            "emotion_prediction": "mae",
        }
        metrics = {
            "emotion_prediction": "mae",
        }
        
    
    model.summary()


    paths_emotion, paths_gender_age, emotion_label,gender_label,age_label = load_data(EMOTION_DATASET,GENDER_AGE_DATASET)
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
    train_gender = gender_label[gender_age_train_idx]
    train_age = age_label[gender_age_train_idx]
    test_gender_age_paths = paths_gender_age[gender_age_test_idx]
    test_gender = gender_label[gender_age_test_idx]
    test_age = age_label[gender_age_test_idx]

    model.compile(optimizer='adam', loss=losses, metrics=metrics)
    # model.summary()



    
    for epoch in  range(EPOCH):

        # emotion_model = Model(inputs=model.inputs,outputs=model.outputs)
        # age_model = Model(inputs=model.inputs,outputs=model.outputs)

        logs_path = './train_logs/Proposed_MTL/{}-{}/{}/'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
        weights_path = './train_weights/Proposed_MTL/{}-{}/{}/'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
        board_name_E = './train_log/Proposed_MTL/{}-{}/{}/emotion/'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
        board_name_G = './train_log/Proposed_MTL/{}-{}/{}/age/'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)

        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(board_name_E):
            os.makedirs(board_name_E)
        if not os.path.exists(board_name_G):
            os.makedirs(board_name_G)


        model_names_E = weights_path +'{}_'.format(2*epoch)+'.{epoch:02d}-{val_emotion_prediction_my_acc:.2f}-{val_gender_prediction_my_acc:.2f}-{val_age_prediction_my_acc:.2f}.hdf5'
        # model_names_E = weights_path + '{}_'.format(2*epoch)+'{epoch:02d}.hdf5'
        csv_name_E = logs_path + 'emotion.log'
        checkpoint_E = ModelCheckpoint(model_names_E, verbose=1,save_weights_only = True,save_best_only=True)
        csvlogger_E=CSVLogger(csv_name_E)
        early_stop_E = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr_E = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard_E = TensorBoard(log_dir=board_name_E,batch_size=BATCH_SIZE)
        callbacks_E = [checkpoint_E,csvlogger_E,early_stop_E,reduce_lr_E,tensorboard_E]
        
        if epoch!=0:
            age_model = keras.models.clone_model(model)
            age_model.set_weights(model.get_weights())
            age_model = freeze_all(age_model)
            # age_model.compile(optimizer='adam', loss=losses, metrics=metrics)
            # print('origin')
            # model.summary()
            # print('after')
            # age_model.summary()

        else:
            age_model = model

        # if epoch == 0:
        #     age_model = None
        # else:
        #     age_model = model
        #     previous_age_model = weights_path + '{}_'.format(2*epoch-1)+'05.hdf5'
        #     age_model.load_weights(previous_age_model)
            # age_model.compile(optimizer='adam', loss=losses, metrics=metrics)
            # age_base = model
            # age_model = Model(age_base.input,age_base.output)
            # age_model = freeze_all(age_model)
            # age_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
            # age_model.summary()
        
        model.fit_generator(
            DataGenerator_emotion(model, age_model,train_emotion_paths,train_emotion, BATCH_SIZE),
            validation_data=Test_DataGenerator(model, test_emotion_paths,test_gender_age_paths, test_emotion,test_gender,test_age, BATCH_SIZE),
            epochs=1,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks_E
        )
        print('{}_emotion_train_finished'.format(epoch))
        if epoch!=0:
            print('emoiton model change:',np.array_equal(model.get_weights()[-1],age_model.get_weights()[-1]))

        



        

        # emotion_base = model
        # emotion_model = Model(input= emotion_base.input,output=emotion_base.output)
        
        # emotion_model = model
        # emotion_model = freeze_all(emotion_model)

        # emotion_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        # emotion_model.summary()
        

        model_names_G = weights_path + '{}_'.format(2*epoch+1)+'.{epoch:02d}-{val_emotion_prediction_my_acc:.2f}-{val_gender_prediction_my_acc:.2f}-{val_age_prediction_my_acc:.2f}.hdf5'
        # model_names_G = weights_path + '{}_'.format(2*epoch+1)+'{epoch:02d}.hdf5'
        csv_name_G = logs_path + 'gender_age.log'
        checkpoint_G = ModelCheckpoint(model_names_G, verbose=1,save_weights_only = True,save_best_only=True)
        csvlogger_G=CSVLogger(csv_name_G)
        early_stop_G = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr_G = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard_G = TensorBoard(log_dir=board_name_G,batch_size=BATCH_SIZE)
        callbacks_G = [checkpoint_G,csvlogger_G,early_stop_G,reduce_lr_G,tensorboard_G] 

        # if epoch == 0:
        #     emotion_model = None
        # else:
        # emotion_model = model
        # previous_emotion_model = weights_path + '{}_'.format(2*epoch)+'05.hdf5'
        # emotion_model.load_weights(previous_emotion_model)
        # emotion_model.compile(optimizer='adam', loss=losses, metrics=metrics)
        # print('original weights:',model.get_weights()[0])
        emotion_model = keras.models.clone_model(model)
        emotion_model.set_weights(model.get_weights())
        emotion_model = freeze_all(emotion_model)
        # emotion_model.compile(optimizer='adam', loss=losses, metrics=metrics)
        # print('origin')
        # model.summary()
        # print('after')
        # emotion_model.summary()

        # print('after weights:',emotion_model.get_weights()[0])
        model.fit_generator(
            DataGenerator_gender_age(model, emotion_model,train_gender_age_paths,train_gender, train_age,BATCH_SIZE),
            validation_data=Test_DataGenerator(model, test_emotion_paths,test_gender_age_paths, test_emotion,test_gender,test_age,BATCH_SIZE),
            epochs=1,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks_G
        )

        print('{}_gender_age_train_finished'.format(epoch))
        print('age model change:',np.array_equal(model.get_weights()[-1],emotion_model.get_weights()[-1]))


if __name__ == '__main__':
    main()
