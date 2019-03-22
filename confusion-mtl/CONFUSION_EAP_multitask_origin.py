import argparse
import os 
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
# from model.inceptionv3 import EmotionNetInceptionV3
# from model.mobilenetv2 import MultitaskMobileNetV2
# from model.vggface import MultitaskVGGFacenet
# from model.mini_xception import EmotionNetmin_XCEPTION
from utils.datasets import DataManager 
from utils.confusion_MTL.confusion_EAP_generator import DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','sfew','expw'],
                    default='expw',
                    help='Model to be used')
parser.add_argument('--dataset_pose',
                    choices=['fer2013','ferplus','sfew','expw'],
                    default='aflw',
                    help='Model to be used')
parser.add_argument('--dataset_attr',
                    choices=['imdb','adience','sfew'],
                    default='celeba',
                    help='Model to be used')
parser.add_argument('--dataset_age',
                    choices=['imdb','adience','sfew'],
                    default='adience',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=[ 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=300,
                    type=int,
                    help='Num of training epoch')
parser.add_argument('--batch_size',
                    default=64,
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

def load_data(dataset_emotion,dataset_pose,dataset_age):
    emotion = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_emotion) )
    pose = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_pose) )
    age = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_age) )

    data_emotion = emotion
    data_pose = pose
    data_age = age
    del emotion,pose,age

    paths_emotion = data_emotion['full_path'].values
    emotion_label = data_emotion['emotion'].values.astype('uint8')

    paths_pose = data_pose['full_path'].values
    roll_label = data_pose['roll'].values.astype('float64')
    pitch_label = data_pose['pitch'].values.astype('float64')
    yaw_label = data_pose['yaw'].values.astype('float64')
    pose_label = [roll_label,pitch_label,yaw_label]
    pose_label = np.transpose(pose_label)

    paths_age = data_age['full_path'].values
    age_label = data_age['age']
    return paths_emotion, paths_pose,paths_age, emotion_label,pose_label,age_label

def mae(y_true, y_pred):
    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_true, axis=1)), axis=-1)

def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:19]:
        layer.trainable = False
    for layer in model.layers[19:]:
        layer.trainable = True
    return model


class ATTR_AVG(keras.callbacks.Callback):
    def __init__(self,validation_data,interval=1,attr_avg=0):
        self.interval=interval
        self.x_val,self.y_val=validation_data
        self.attr_avg = attr_avg
    def on_epoch_end(self,epoch, logs={}):
        if epoch % self.interval == 0:
            y_score=self.model.predict(self.x_val,verbose=0)
            for i in range(40):
                self.attr_avg+=y_score[i+2]
            self.attr_avg = self.attr_avg / 40
        print(self.attr_avg)



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
    POSE_DATASET = args.dataset_pose
    AGE_DATASET = args.dataset_age

    if EMOTION_DATASET == 'ferplus':
        emotion_classes = 8
    else:
        emotion_classes = 7

    
    gender_classes = 2
    age_classes = 8
   



    paths_emotion, paths_pose,paths_age, emotion_label,pose_label,age_labels = load_data(EMOTION_DATASET,POSE_DATASET,AGE_DATASET)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split_emotion = kf.split(paths_emotion)
    kf_split_pose = kf.split(paths_pose)
    kf_split_age = kf.split(paths_age)

    emotion_kf = [[emotion_train_idx,emotion_test_idx] for emotion_train_idx,emotion_test_idx in kf_split_emotion]
    pose_kf = [[pose_train_idx,pose_test_idx] for pose_train_idx,pose_test_idx in kf_split_pose]
    age_kf = [[age_train_idx,age_test_idx] for age_train_idx,age_test_idx in kf_split_age]

    # for emotion_train_idx,emotion_test_idx in kf_split_emotion:
    #     for gender_age_train_idx,gender_age_test_idx in kf_split_gender_age:
            # print(emotion_train_idx,emotion_test_idx,gender_age_train_idx,gender_age_test_idx)
    emotion_train_idx,emotion_test_idx = emotion_kf[0]
    pose_train_idx,pose_test_idx = pose_kf[0]
    age_train_idx,age_test_idx = age_kf[0]


    print(len(emotion_train_idx),len(pose_train_idx),len(age_train_idx))
    print(len(emotion_test_idx),len(pose_test_idx),len(age_test_idx))


    train_emotion_paths = paths_emotion[emotion_train_idx]
    train_emotion = emotion_label[emotion_train_idx]
    test_emotion_paths = paths_emotion[emotion_test_idx]
    test_emotion = emotion_label[emotion_test_idx]

    train_pose_paths = paths_pose[pose_train_idx]
    train_pose = pose_label[pose_train_idx]
    test_pose_paths = paths_pose[pose_test_idx]
    test_pose = pose_label[pose_test_idx]


    train_age_paths = paths_age[age_train_idx]
    train_age = age_labels[age_train_idx]
    test_age_paths = paths_age[age_test_idx]
    test_age = age_labels[age_test_idx]



    model = None
    if MODEL == 'vggFace':
        model = Net(MODEL,1,9,emotion_classes,gender_classes,age_classes)
        # model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        model = Net(MODEL,1,9,emotion_classes,gender_classes,age_classes)
        MODEL = model.name


    def my_cross_loss(y_true, y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())
        loss = K.categorical_crossentropy(y_true, y_pred) * mask
        return K.sum(loss) / K.sum(mask)

    def my_bin_loss(y_true, y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())
        loss = K.binary_crossentropy(y_true, y_pred) * mask
        return K.sum(loss) / K.sum(mask)
    def my_mean_square(y_true,y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())
        loss = keras.losses.mean_squared_error(y_true,y_pred)* mask
        return K.sum(loss) / K.sum(mask)

    def my_acc(y_true, y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx()) 
        acc = (K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx()))*mask
        return K.sum(acc)/K.sum(mask)

    def pose_metric(y_true,y_pred):
        # y_true = K.dot(180/np.pi*K.ones(K.shape(y_true)),y_true)
        # y_pred = K.dot(180/np.pi*K.ones(K.shape(y_pred)),y_pred)
        y_true = 180/3.14*y_true
        y_pred = 180/3.14*y_pred
        comp = 15*K.ones(K.shape(y_pred))
        comp = K.cast(comp, K.floatx())
        sub = K.cast(K.abs(y_true-y_pred), K.floatx())
        diff = K.greater(comp,sub)
        diff = K.cast(diff, K.floatx())
        result = K.sum(diff,axis=0)/K.cast(K.shape(y_pred)[0], K.floatx())
        return result[0]
    def my_pose_metric(y_true,y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())
        acc = pose_metric(y_true,y_pred)*mask
        return K.sum(acc) / K.sum(mask)



    if model.task_type == 3:
        loss_dict = {}
        metrics_dict = {}

        prediction = []
        loss =[]
        acc =[]
        loss_weights = []
        
        prediction.append('emotion_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(1)

        prediction.append('age_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(1)
        

        loss_dict = dict(zip(prediction, loss))
        metrics_dict = dict(zip(prediction, acc))
        weights_path = './train_weights/CONFUSION/{}-{}/{}/'.format(EMOTION_DATASET,AGE_DATASET,MODEL)
        logs_path = './train_log/CONFUSION/{}-{}/{}/'.format(EMOTION_DATASET,AGE_DATASET,MODEL)
    
    elif model.task_type == 9:
        loss_dict = {}
        metrics_dict = {}

        prediction = []
        loss =[]
        acc =[]
        loss_weights = []
        
        prediction.append('emotion_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(1)

        prediction.append('pose_prediction')
        loss.append(my_mean_square)
        acc.append(my_pose_metric)
        loss_weights.append(50)

        prediction.append('age_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(1)
        

        loss_dict = dict(zip(prediction, loss))
        metrics_dict = dict(zip(prediction, acc))
        weights_path = './train_weights/CONFUSION/{}-{}-{}/{}/'.format(EMOTION_DATASET,POSE_DATASET,AGE_DATASET,MODEL)
        logs_path = './train_log/CONFUSION/{}-{}-{}/{}/'.format(EMOTION_DATASET,POSE_DATASET,AGE_DATASET,MODEL)
    
    model.summary()
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    if model.task_type == 9:
        model_names = weights_path + '.{epoch:02d}-{val_emotion_prediction_my_acc:.2f}-{val_pose_prediction_my_pose_metric:.2f}-{val_age_prediction_my_acc:.2f}.hdf5'
    elif model.task_type == 3:
        model_names = weights_path + '.{epoch:02d}-{val_emotion_prediction_my_acc:.2f}-{val_age_prediction_my_acc:.2f}.hdf5'
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
    
    
    model.compile(optimizer='adam', loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
    model.fit_generator(
        DataGenerator(model,None, train_emotion_paths,train_pose_paths, train_age_paths,train_emotion, train_pose, train_age,BATCH_SIZE),
        validation_data=DataGenerator(model,None,test_emotion_paths, test_pose_paths,test_age_paths,  test_emotion,test_pose,test_age, BATCH_SIZE),
        epochs=EPOCH,
        verbose=2,
        workers=NUM_WORKER,
        use_multiprocessing=False,
        max_queue_size=int(BATCH_SIZE * 2),
        callbacks=callbacks
    )
    del  train_emotion_paths, train_pose_paths,train_age_paths,train_emotion,train_pose, train_age
    del  test_emotion_paths, test_pose_paths,test_age_paths,test_emotion, test_pose,test_age


if __name__ == '__main__':
    main()
