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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    choices=['expw','adience','aflw','ferplus','fer2013'],
                    default='aflw',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=4,
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
parser.add_argument('--is_freezing',
                     type= ast.literal_eval,
                    help='whether pesudo-label selection')
parser.add_argument('--is_dropout',
                     type= ast.literal_eval,
                    help='whether need dropout')
parser.add_argument('--no_freezing_epoch',
                    default=2,
                    type=int,
                    help='starting no freezing')
parser.add_argument('--task_type',
                    default=5,
                    type=int,
                    help='emotion 0,age 1, pose 5,')

# def scheduler(epoch):
#     if epoch==5:
#         K.set_value(model.optimizer.lr, lr * 0.0001)
#         print("lr changed to {}".format(lr * 0.0001))
#     elif epoch==9:
#         K.set_value(model.optimizer.lr, lr * 0.001)
#         print("lr changed to {}".format(lr * 0.001))
#     elif epoch==11:
#         K.set_value(model.optimizer.lr, lr * 0.01)
#         print("lr changed to {}".format(lr * 0.01))
#     elif epoch==13:
#         K.set_value(model.optimizer.lr, lr * 0.1)
#         print("lr changed to {}".format(lr * 0.1))
#     return K.get_value(model.optimizer.lr)
 

def scheduler(epoch):
    lr= 0.001
    if epoch==0:
        lr=lr*1e-12
        print("lr changed to {}".format(lr))
    elif epoch==4:
        lr=lr*1e-9
        print("lr changed to {}".format(lr))
    elif epoch==8:
        lr=lr*1e-6
        print("lr changed to {}".format(lr))
    elif epoch==12:
        lr=lr*1e-3
        print("lr changed to {}".format(lr))
    elif epoch==16:
        lr=lr
        print("lr changed to {}".format(lr))
    return lr


# def scheduler(epoch):
#     lr= 0.001
#     if epoch>=0:
#         lr=lr*0.0001
#         print("lr changed to {}".format(lr))
#     return lr
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

def model_factorization(model):
    middle_layer = model.get_layer('pool5').output
    model_1=Model(inputs=model.input,outputs=middle_layer)
    model_2 = Model(inputs=model_1.output,outputs=model.output)
    return model_1,model_2
def final_model(model_1,model_2):
    model = Model(inputs=model_1.input,outputs=model_2.output)
    return model

# def updateTargetmodel(model,target_model):
#     modelweights = model.trainable_weights
#     targetmodelweights = target_model.trainable_weights
#     for i in range(len(targetmodelweights)):
#         targetmodelweights[i].assign(modelweights[i])
#     return target_model

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
            model = Net(MODEL,1,args.task_type,args.is_dropout,8,5,8,2)
        else:
            model = Net(MODEL,1,args.task_type,args.is_dropout,7,5,8,2)
        if args.is_freezing:
            model = freeze_all_but_mid_and_top(model)
    else:
        if args.dataset=='ferplus':
            model = Net(MODEL,1,args.task_type,args.is_dropout,8,5,8,2)
        else:
            model = Net(MODEL,1,args.task_type,args.is_dropout,7,5,8,2)
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


    if args.is_freezing:
        freezing='freezing_true'
    else:
        freezing='freezing_false'
    if args.is_dropout:
        dropout='dropout_true'
    else:
        dropout='drouout_false'
    weights_path = './train_weights/{}/{}/{}-{}_'.format(model.name,DATASET,freezing,dropout)
    logs_path = './train_log/{}/{}/{}-{}_'.format(model.name,DATASET,freezing,dropout)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    model_names = weights_path  + '{epoch:02d}-{val_acc:.2f}.hdf5'
    csv_name = logs_path + '.log'
    board_name = logs_path

    checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=True)
    csvlogger=CSVLogger(filename=csv_name,separator=',',append=True)
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(PATIENCE/2), verbose=1)
    tensorboard = TensorBoard(log_dir=board_name,batch_size=BATCH_SIZE)
    callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]
    adam = keras.optimizer.adam(lr=0.0001)
    model.compile(optimizer=adam, loss=losses, metrics=metrics)
    print('starting model')
    model.summary()
    model.fit_generator(
        DataGenerator(model, train_paths, train_task, task_classes, BATCH_SIZE),
        validation_data=DataGenerator(model, test_paths, test_task, task_classes,BATCH_SIZE),
        epochs=args.no_freezing_epoch,
        verbose=2,
        workers=NUM_WORKER,
        use_multiprocessing=False,
        max_queue_size=int(BATCH_SIZE * 2),
        callbacks=callbacks
    )
    if args.is_freezing:
        reduce_lr = LearningRateScheduler(scheduler)
    else:
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(PATIENCE/2), verbose=1)
    if args.is_freezing:
        before_freezing_weights=model.get_weights()[1]
        model = no_freeze_all(model)
        model.compile(optimizer=adam, loss=losses, metrics=metrics)
        callbacks_new = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]
        no_freezing_weights=model.get_weights()[1]
        print('combile whether change',np.equal(before_freezing_weights,no_freezing_weights))
        model.summary()
    model.fit_generator(
        DataGenerator(model, train_paths, train_task, task_classes, BATCH_SIZE),
        validation_data=DataGenerator(model, test_paths, test_task, task_classes,BATCH_SIZE),
        epochs=EPOCH-args.no_freezing_epoch,
        verbose=2,
        workers=NUM_WORKER,
        use_multiprocessing=False,
        max_queue_size=int(BATCH_SIZE * 2),
        callbacks=callbacks_new
    )
    train_no_freezing_weights=model.get_weights()[1]
    print('second stage training change',np.equal(no_freezing_weights,train_no_freezing_weights))
    del  train_paths, train_task
    del  test_paths, test_task


if __name__ == '__main__':
    main()
