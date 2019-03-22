import argparse
import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from utils.STL.general_generator import DataGenerator
from utils.callback import DecayLearningRate
from keras.callbacks import LearningRateScheduler
from model.models import Net
import ast
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    choices=['expw','adience','aflw','ferplus','fer2013','SFEW','imdb'],
                    default='imdb',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=64,
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
parser.add_argument('--task_type',
                    default=10,
                    type=int,
                    help='emotion 0,age 1, pose 5,gender 13,')
#add training trick
parser.add_argument('--patience',
                    default=20,
                    type=int,
                    help='Number of traing epoch')
parser.add_argument('--is_augmentation',
                     type= ast.literal_eval,
                    help='whether data augmentation')
parser.add_argument('--is_dropout',
                     type= ast.literal_eval,
                    help='whether dropot')
parser.add_argument('--is_bn',
                     type= ast.literal_eval,
                    help='whether bn')
parser.add_argument('--weights_decay',
                    default=0.005,
                     type= float,
                    help='dense layer weights decay')
#add training strategy
parser.add_argument('--is_freezing',
                     type= ast.literal_eval,
                    help='whether pesudo-label selection')
parser.add_argument('--no_freezing_epoch',
                    default=32,
                    type=int,
                    help='starting no freezing')
# parser.add_argument('--lr_steps',
#                     default=10,
#                     type=int,
#                     help='lr step')

def load_data(dataset,task_name):
    file_path = 'data/db/{}_cleaned.csv'.format(dataset) 
    data1 = pd.read_csv(file_path)
    data = data1
    del data1
    paths = data['full_path'].values
    task_label = data[task_name].values.astype('uint8')
    return paths, task_label

#'vgg16',     [:19]net layer,      'pool5'
#'resnet50' ,     [:174]net layer,    'avg_pool'
#'senet50' ,    [:286] net layer,        'avg_pool'
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

def no_freeze_all(model):
    for layer in model.layers[:]:
        layer.trainable = True
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

    if args.task_type == 0:
        task_name = 'emotion'
    elif args.task_type ==1:
        task_name = 'age'
    elif args.task_type == 5:
        task_name = 'pose'
    elif args.task_type == 10:
        task_name = 'gender'

    paths, task_label = load_data(DATASET,task_name)
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split_task = kf.split(task_label)
    task_kf = [[task_train_idx,task_test_idx] for task_train_idx,task_test_idx in kf_split_task]
    task_train_idx,task_test_idx = task_kf[0]
    train_paths = paths[task_train_idx]
    train_task = task_label[task_train_idx]
    test_paths = paths[task_test_idx]
    test_task = task_label[task_test_idx]

    model = None
    if MODEL == 'vggFace':
        if args.dataset=='ferplus':
            model = Net(MODEL,1,args.task_type,8,5,8,2,args.is_dropout,args.is_bn,args.weights_decay)
        else:
            model = Net(MODEL,1,args.task_type,7,5,8,2,args.is_dropout,args.is_bn,args.weights_decay)
        if args.is_freezing:
            model = freeze_all_but_mid_and_top(model)
    else:
        if args.dataset=='ferplus':
            model = Net(MODEL,1,args.task_type,8,5,8,2,args.is_dropout,args.is_bn,args.weights_decay)
        else:
            model = Net(MODEL,1,args.task_type,7,5,8,2,args.is_dropout,args.is_bn,args.weights_decay)
        if args.is_freezing:
            model = freeze_all_but_mid_and_top(model)

    if args.task_type == 0:
        task_classes = model.emotion_classes
        losses = {"emotion_prediction": 'categorical_crossentropy' }
        metrics = { "emotion_prediction": 'acc'   }
    elif args.task_type ==1:
        task_classes = model.age_classes
        losses = {"age_prediction": 'categorical_crossentropy' }
        metrics = { "age_prediction": 'acc'   }
    elif args.task_type == 5:
        task_classes = model.pose_classes
        losses = {"pose_prediction": 'categorical_crossentropy' }
        metrics = { "pose_prediction": 'acc'   }
    elif args.task_type == 10:
        task_classes = model.gender_classes
        losses = {"gender_prediction": 'categorical_crossentropy' }
        metrics = { "gender_prediction": 'acc'   }

    #training trick
    if args.is_augmentation:
        augmentation='augmentation_true'
    else:
        augmentation='augmentation_false'
    if args.is_dropout:
        dropout='dropout_true'
    else:
        dropout='dropout_false'
    if args.is_bn:
        bn = 'bn_true'
    else:
        bn = 'bn_false'
    # key parameter
    if args.is_freezing:
        freezing='freezing_true'
    else:
        freezing='freezing_false'

    weights_path = './train_weights/{}/{}/net-{}_{}_{}_{}-lr-{}_{}'.format(model.name,DATASET,augmentation,dropout,bn,args.weights_decay,freezing,args.no_freezing_epoch)
    logs_path = './train_log/{}/{}/net-{}_{}_{}_{}-lr-{}_{}'.format(model.name,DATASET,augmentation,dropout,bn,args.weights_decay,freezing,args.no_freezing_epoch)
    matrix_path = logs_path+'/matrix/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path)

    model_names = weights_path  + '{epoch:02d}-{val_acc:.4f}.hdf5'
    csv_name = logs_path + '.log'
    board_name = logs_path

    checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=True)
    csvlogger=CSVLogger(filename=csv_name,separator=',',append=True)
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(PATIENCE), verbose=1)
    tensorboard = TensorBoard(log_dir=board_name,batch_size=BATCH_SIZE)
    callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]
    if args.is_freezing:
        model.compile(optimizer='adam', loss=losses, metrics=metrics)
        model.summary()
        model.fit_generator(
            DataGenerator(model, train_paths, train_task, task_classes, BATCH_SIZE,is_augmentation=args.is_augmentation),
            validation_data=DataGenerator(model, test_paths, test_task, task_classes,BATCH_SIZE,is_augmentation=False),
            epochs=args.no_freezing_epoch,
            verbose=2,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )
        if args.epoch>args.no_freezing_epoch:
            adam = keras.optimizers.Adam(lr=1*1e-5)
            print('lr changing to:',1*1e-5)
            model = no_freeze_all(model)
            model.compile(optimizer=adam, loss=losses, metrics=metrics) 
            model.summary()
            model.fit_generator(
            DataGenerator(model, train_paths, train_task, task_classes, BATCH_SIZE,is_augmentation=args.is_augmentation),
            validation_data=DataGenerator(model, test_paths, test_task, task_classes,BATCH_SIZE,is_augmentation=False),
            epochs=args.epoch-args.no_freezing_epoch,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks)

        predicted_result = model.predict_generator(DataGenerator(model, test_paths, test_task, task_classes,BATCH_SIZE,is_augmentation=False))
        task_reuslt=np.argmax(np.array(predicted_result),axis=1)
        matrix=confusion_matrix(test_task,task_reuslt)
        print('confusion matrix:',matrix)
        np.savetxt(matrix_path+'.txt',matrix)
        

    else:
        adam = keras.optimizers.Adam(lr=0.00001)
        model.compile(optimizer=adam, loss=losses, metrics=metrics)
        model.summary()
        model.fit_generator(
            DataGenerator(model, train_paths, train_task, task_classes, BATCH_SIZE,is_augmentation=args.is_augmentation),
            validation_data=DataGenerator(model, test_paths, test_task, task_classes,BATCH_SIZE,is_augmentation=False),
            epochs=args.epoch,
            verbose=2,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks)
        
        predicted_result = model.predict_generator(DataGenerator(model, test_paths, test_task, task_classes,BATCH_SIZE,is_augmentation=False))
        task_reuslt=np.argmax(np.array(predicted_result),axis=1)
        matrix=confusion_matrix(test_task,task_reuslt)
        print('confusion matrix:',matrix)
        np.savetxt(matrix_path+'.txt',matrix)
        
    del  train_paths, train_task
    del  test_paths, test_task


if __name__ == '__main__':
    main()
