import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from utils.STL.pose_generator import DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net
from keras.objectives import categorical_crossentropy,mean_squared_error


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='aflw',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='vggFace',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=10,
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


def load_data(dataset):
    file_path = 'data/db/{}_cleaned.csv'.format(dataset) 
    data1 = pd.read_csv(file_path,delimiter=',',encoding='utf-8-sig')
    data = data1
    del data1
    paths = data['full_path'].values
    roll_label = data['roll'].values.astype('float64')
    pitch_label = data['pitch'].values.astype('float64')
    yaw_label = data['yaw'].values.astype('float64')
    label = [roll_label,pitch_label,yaw_label]
    label = np.transpose(label)
    return paths, label


def custom_mse_pose(y_true,y_pred):
    # return K.sign(K.sum(K.abs(y_true),axis=-1))*keras.losses.mean_squared_error(y_true,y_pred)
    return keras.losses.mean_squared_error(y_true,y_pred)

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
    
#'vgg16',     [:19]net layer,      'pool5'
#'resnet50' ,     [:174]net layer,    'avg_pool'
#'senet50' ,    [:286] net layer,        'avg_pool'
def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:19]:
        layer.trainable = False
    return model


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)

    args = parser.parse_args()
    DATASET = args.dataset
    MODEL = args.model
    EPOCH = args.epoch
    PATIENCE = args.patience
    BATCH_SIZE = args.batch_size
    NUM_WORKER = args.num_worker

    paths, pose_label = load_data(DATASET)
    n_fold = 1
 

    

    losses = {"pose_prediction": custom_mse_pose}
    metrics = {"pose_prediction": pose_metric}

    print('[K-FOLD] Started...')
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_split = kf.split(pose_label)
    for train_idx, test_idx in kf_split:

        model = None
        if MODEL == 'vggFace':
            model = Net(MODEL,1,5,8,2,8)
            model = freeze_all_but_mid_and_top(model)
        else:
            model = Net(MODEL,1,5,8,2,8)

        train_paths = paths[train_idx]
        train_label = pose_label[train_idx]

        test_paths = paths[test_idx]
        test_label = pose_label[test_idx]

        print(len(train_paths),len(test_paths))

        

        model.summary()
        weights_path = './train_weights/pose/{}/{}/'.format(DATASET,model.name)
        logs_path = './train_log/pose/{}/{}/'.format(DATASET,model.name)
        
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        model_names = weights_path + '.{epoch:02d}-{val_pose_metric: 0.4f}.hdf5'
        csv_name = logs_path + '{}.log'.format(n_fold)
        board_name = logs_path + '{}'.format(n_fold)

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
        
        model.compile(optimizer='adam', loss=losses, metrics = metrics)
        model.fit_generator(
            DataGenerator(model, train_paths, train_label, BATCH_SIZE),
            validation_data=DataGenerator(model, test_paths, test_label, BATCH_SIZE),
            epochs=EPOCH,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )
        n_fold += 1
        del  train_paths, train_roll,train_pictch,train_yaw
        del  test_paths, test_roll,test_pitch,test_yaw


if __name__ == '__main__':
    main()
