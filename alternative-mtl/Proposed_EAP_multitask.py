import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
import ast
from utils.datasets import DataManager 
from utils.proposed_MTL.alternate_generator_EAP_emotion import DataGenerator_emotion
from utils.proposed_MTL.alternate_generator_EAP_pose import DataGenerator_pose
from utils.proposed_MTL.alternate_generator_EAP_age import DataGenerator_age
from utils.confusion_MTL.confusion_EAP_generator import DataGenerator as Test_DataGenerator
from utils.callback import DecayLearningRate
from keras.callbacks import LearningRateScheduler
from model.models import Net
tf_session = K.get_session() 
tf_graph = tf.get_default_graph()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','sfew'],
                    default='expw',
                    help='Model to be used')
parser.add_argument('--dataset_age',
                    choices=['imdb','adience','sfew',],
                    default='adience',
                    help='Model to be used')
parser.add_argument('--dataset_pose',
                    choices=['imdb','adience','sfew',],
                    default='aflw',
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
                    default=10,
                    type=int,
                    help='Number of traing epoch')
parser.add_argument('--E_loss_weights',
                    default= 1,
                    type= float,
                    help='emotion')
parser.add_argument('--P_loss_weights',
                    default= 1,
                    type= float,
                    help='pose')
parser.add_argument('--A_loss_weights',
                    default= 1,
                    type= float,
                    help='age')
parser.add_argument('--no_freezing_epoch',
                    default=5,
                    type=int,
                    help='starting no freezing')
parser.add_argument('--freezing',
                    type= ast.literal_eval,
                    help='is freeze')
parser.add_argument('--is_based_STL',
                    type=ast.literal_eval,
                    help='whether need  pre-trained on STL')
parser.add_argument('--is_pesudo',
                    type=ast.literal_eval,
                    help='whether need psudo-label')
parser.add_argument('--pesudo_selection',
                    type= ast.literal_eval,
                    help='whether pesudo-label selection')
parser.add_argument('--is_dropout',
                     type= ast.literal_eval,
                    help='whether need dropout')
parser.add_argument('--selection_threshold',
                    default= 0.9,
                    type= float,
                    help='pesudo-label selection threshold')
parser.add_argument('--emotion_path',
                    type=str,
                    help='emotion model')
parser.add_argument('--pose_path',
                    type=str,
                    help='pose model')
parser.add_argument('--age_path',
                    type=str,
                    help='age model')
parser.add_argument('--each_model_epoch',
                    default=1,
                    type=int,
                    help='each model epoch')
parser.add_argument('--is_EP',
                    type=ast.literal_eval,
                    help='is emotion-pose model')
parser.add_argument('--lr_step',
                    default= 10,
                    type=int,
                    help='lr step')

# def scheduler(epoch):
#     lr= 0.001
#     lr_steps=10
#     if epoch==0:
#         lr=lr*1e-12
#         print("lr changed to {}".format(lr))
#     elif epoch==lr_steps:
#         lr=lr*1e-9
#         print("lr changed to {}".format(lr))
#     elif epoch==lr_steps*2:
#         lr=lr*1e-6
#         print("lr changed to {}".format(lr))
#     elif epoch==lr_steps*3:
#         lr=lr*1e-3
#         print("lr changed to {}".format(lr))
#     elif epoch==lr_steps*4:
#         lr=lr*1e-2
#         print("lr changed to {}".format(lr))
#     elif epoch==lr_steps*5:
#         lr=lr*1e-1
#         print("lr changed to {}".format(lr))
#     return lr


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
    pose_label = data_pose['pose'].values.astype('uint8')
    
    paths_age = data_age['full_path'].values
    age_label = data_age['age'].values.astype('uint8')
    return paths_emotion, paths_pose, paths_age, emotion_label,pose_label,age_label


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

def my_cross_loss(y_true, y_pred):
    mask = K.all(K.equal(y_true, 0), axis=-1)
    mask = 1 - K.cast(mask, K.floatx())
    loss = K.categorical_crossentropy(y_true, y_pred) * mask+1e-10
    #loss=(-tf.reduce_sum(y_true*tf.log(tf.clip_by_value(y_pred,1e-10,1.0))))*mask
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

def base_STL_model(MODEL,is_EP,is_dropout,emotion_path,pose_path,age_path,emotion_classes,pose_classes,age_classes,gender_classes):
    emotion_model=Net(MODEL,1,0,is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)
    pose_model=Net(MODEL,1,5,is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)
    age_model=Net(MODEL,1,1,is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)
    emotion_model.load_weights(emotion_path)
    pose_model.load_weights(pose_path)
    age_model.load_weights(age_path)
    if is_EP:
        model= Net(MODEL,1,11,is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)
    else:
        model= Net(MODEL,1,9,is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)

    if is_EP:
            #flatten common_fc
        model.layers[19].set_weights(emotion_model.layers[19].get_weights())
        model.layers[20].set_weights(emotion_model.layers[20].get_weights())

        # emotion_FC pose_fc age_FC
        model.layers[21].set_weights(emotion_model.layers[21].get_weights())
        model.layers[22].set_weights(pose_model.layers[21].get_weights())

        # emotion_prediction pose_prediction age_prediction
        model.layers[23].set_weights(emotion_model.layers[22].get_weights())
        model.layers[24].set_weights(pose_model.layers[22].get_weights())
        return model
    else:

        #flatten common_fc
        model.layers[19].set_weights(emotion_model.layers[19].get_weights())
        model.layers[20].set_weights(emotion_model.layers[20].get_weights())

        # emotion_FC pose_fc age_FC
        model.layers[21].set_weights(emotion_model.layers[21].get_weights())
        model.layers[22].set_weights(pose_model.layers[21].get_weights())
        model.layers[23].set_weights(age_model.layers[21].get_weights())

        # emotion_prediction pose_prediction age_prediction
        model.layers[24].set_weights(emotion_model.layers[22].get_weights())
        model.layers[25].set_weights(pose_model.layers[22].get_weights())
        model.layers[26].set_weights(age_model.layers[22].get_weights())
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
    POSE_DATASET = args.dataset_pose
    AGE_DATASET = args.dataset_age
    threshold_K = args.selection_threshold
    IS_seletion = args.pesudo_selection

    emotion_classes = 7
    gender_classes = 2
    age_classes  = 8
    pose_classes =5
    
    
    model = None
    emotion_model = None
    age_model = None
    pose_model = None
    model = None
    if MODEL == 'vggFace':
        if args.is_based_STL:
            print('based on single task')
            model = base_STL_model(MODEL,args.is_EP,args.is_dropout,args.emotion_path,args.pose_path,args.age_path,emotion_classes,pose_classes,age_classes,gender_classes)
        else:
            print('based on vggface')
            if args.is_EP:
                model = Net(MODEL,1,11,args.is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)
            else:
                model = Net(MODEL,1,9,args.is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)
        
        
        if args.freezing:
            model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        if args.is_EP:
            model = Net(MODEL,1,11,args.is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)
        else:
            model = Net(MODEL,1,9,args.is_dropout,emotion_classes,pose_classes,age_classes,gender_classes)
        if args.freezing:
            model = freeze_all_but_mid_and_top(model)
        MODEL = model.name

    if model.task_type==9:
        loss_dict = {}
        metrics_dict = {}

        prediction = []
        loss =[]
        acc =[]
        loss_weights = []
        
        prediction.append('emotion_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(args.E_loss_weights)

        prediction.append('pose_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(args.P_loss_weights)

        prediction.append('age_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(args.A_loss_weights)
        loss_dict = dict(zip(prediction, loss))
        metrics_dict = dict(zip(prediction, acc))
    elif model.task_type==11:
        loss_dict = {}
        metrics_dict = {}

        prediction = []
        loss =[]
        acc =[]
        loss_weights = []
        
        prediction.append('emotion_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(args.E_loss_weights)

        prediction.append('pose_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(args.P_loss_weights)

        loss_dict = dict(zip(prediction, loss))
        metrics_dict = dict(zip(prediction, acc))

    paths_emotion, paths_pose,paths_age, emotion_label,pose_label,age_label = load_data(EMOTION_DATASET,POSE_DATASET,AGE_DATASET)
    print(len(emotion_label),len(pose_label),len(age_label))
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split_age = kf.split(paths_age)
    kf_split_emotion = kf.split(paths_emotion)
    kf_split_pose = kf.split(paths_pose)

    emotion_kf = [[emotion_train_idx,emotion_test_idx] for emotion_train_idx,emotion_test_idx in kf_split_emotion]
    pose_kf = [[pose_train_idx,pose_test_idx] for pose_train_idx,pose_test_idx in kf_split_pose]
    age_kf = [[age_train_idx,age_test_idx] for age_train_idx,age_test_idx in kf_split_age]
    
    emotion_train_idx,emotion_test_idx = emotion_kf[0]
    train_emotion_paths = paths_emotion[emotion_train_idx]
    train_emotion = emotion_label[emotion_train_idx]
    test_emotion_paths = paths_emotion[emotion_test_idx]
    test_emotion = emotion_label[emotion_test_idx]

    pose_train_idx,pose_test_idx = pose_kf[0]
    train_pose_paths = paths_pose[pose_train_idx]
    train_pose = pose_label[pose_train_idx]
    test_pose_paths = paths_pose[pose_test_idx]
    test_pose = pose_label[pose_test_idx]

    age_train_idx,age_test_idx = age_kf[0]
    train_age_paths = paths_age[age_train_idx]
    train_age = age_label[age_train_idx]
    test_age_paths = paths_age[age_test_idx]
    test_age = age_label[age_test_idx]

    print('whether freezing:',args.freezing)
    if not args.freezing:
        adam=keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
    else:
        model.compile(optimizer='adam', loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
    model.summary()

    if args.is_pesudo:
        is_pesudo='pesudo_true'
    else:
        is_pesudo='pesudo_false'
    if args.is_based_STL:
        is_based_STL='based_STL_true'
    else:
        is_based_STL='based_STL_false'
    if args.freezing:
        is_freezing='freezing_true'
    else:
        is_freezing='freezing_false'
    if args.pesudo_selection:
        is_pesudo_selection='pesudo_selection_true'
    else:
        is_pesudo_selection='pesudo_selection_false'
    if args.is_dropout:
        dropout='dropout_true'
    else:
        dropout='drouout_false'

    if model.task_type==9:
        main_path='{}-{}-{}/{}/'.format(EMOTION_DATASET,POSE_DATASET,AGE_DATASET,MODEL)
        appendex_path='{}-{}-{}-{}-{}-{}-{}_{}_{}-threshold_{}'.format(dropout,is_based_STL,is_freezing,is_pesudo,is_pesudo_selection,args.lr_step,args.E_loss_weights,args.P_loss_weights,args.A_loss_weights,args.selection_threshold)
        weights_path = './train_weights/ITERATION/'+main_path+appendex_path
        logs_path = './train_log/ITERATION/'+main_path+appendex_path
    elif model.task_type==11:
        main_path='{}-{}/{}/'.format(EMOTION_DATASET,POSE_DATASET,MODEL)
        appendex_path='{}-{}-{}-{}-{}-{}-{}_{}-threshold_{}'.format(dropout,is_based_STL,is_freezing,is_pesudo,is_pesudo_selection,args.lr_step,args.E_loss_weights,args.P_loss_weights,args.selection_threshold)
        weights_path = './train_weights/ITERATION/'+main_path+appendex_path
        logs_path = './train_log/ITERATION/'+main_path+appendex_path

    if not os.path.exists(weights_path):
            os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    if model.task_type==9:
        model_names = weights_path +'.{epoch:02d}-{val_emotion_prediction_my_acc:.2f}-{val_pose_prediction_my_acc:.2f}-{val_age_prediction_my_acc:.2f}.hdf5'
    elif model.task_type==11:
        model_names = weights_path +'.{epoch:02d}-{val_emotion_prediction_my_acc:.2f}-{val_pose_prediction_my_acc:.2f}.hdf5'
    csv_name = logs_path + '.log'
    checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=True)
    csvlogger=CSVLogger(filename=csv_name,append=True)
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(PATIENCE), verbose=1)
    tensorboard = TensorBoard(log_dir=logs_path,batch_size=BATCH_SIZE)
    callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]


    for epoch in  range(EPOCH):
        if not args.is_pesudo:
            pose_model=None
            if model.task_type==9:
                age_model=None
        else:
            if epoch==0:
                pose_model = keras.models.clone_model(model)
                pose_model.set_weights(model.get_weights())
                pose_model = freeze_all(pose_model)
                if model.task_type==9:
                    age_model = keras.models.clone_model(model)
                    age_model.set_weights(model.get_weights())
                    age_model = freeze_all(age_model)
            else:
                if model.task_type==9:
                    age_model = keras.models.clone_model(model)
                    age_model.set_weights(model.get_weights())
                    age_model = freeze_all(age_model)
        if model.task_type==9:
            model.fit_generator(
            DataGenerator_emotion(model, age_model,age_model,train_emotion_paths,train_emotion, BATCH_SIZE,IS_seletion,threshold_K),
            validation_data=Test_DataGenerator(model, test_emotion_paths,test_pose_paths,test_age_paths, test_emotion,test_pose,test_age, BATCH_SIZE),
            epochs=args.each_model_epoch,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks)
        elif model.task_type==11:
            model.fit_generator(
                DataGenerator_emotion(model, pose_model,pose_model,train_emotion_paths,train_emotion, BATCH_SIZE,IS_seletion,threshold_K),
                validation_data=Test_DataGenerator(model, test_emotion_paths,test_pose_paths,test_age_paths, test_emotion,test_pose,test_age, BATCH_SIZE),
                epochs=args.each_model_epoch,
                verbose=2,
                workers=NUM_WORKER,
                use_multiprocessing=False,
                max_queue_size=int(BATCH_SIZE * 2),
                callbacks=callbacks)

        # if epoch!=0:
        #     print('emotion-pose',np.equal(model.get_weights()[-1],pose_model.get_weights()[-1]))
        #     print('emotion-age',np.equal(model.get_weights()[-1],age_model.get_weights()[-1]))
        print('{}_emotion_train_finished'.format(epoch))

        

        # with tf_session.as_default():
        #     with tf_graph.as_default():
        emotion_model = keras.models.clone_model(model)
        emotion_model.set_weights(model.get_weights())
        emotion_model = freeze_all(emotion_model)
        if not args.is_pesudo:
            emotion_model=None 
        if model.task_type==9:
            model.fit_generator(
                DataGenerator_pose(model, emotion_model,emotion_model,train_pose_paths,train_pose,BATCH_SIZE,IS_seletion,threshold_K),
                validation_data=Test_DataGenerator(model, test_emotion_paths,test_pose_paths,test_age_paths, test_emotion,test_pose,test_age,BATCH_SIZE),
                epochs=args.each_model_epoch,
                verbose=2,
                workers=NUM_WORKER,
                use_multiprocessing=False,
                max_queue_size=int(BATCH_SIZE * 2),
                callbacks=callbacks)
        elif model.task_type==11:
            model.fit_generator(
                    DataGenerator_pose(model, emotion_model,emotion_model,train_pose_paths,train_pose,BATCH_SIZE,IS_seletion,threshold_K),
                    validation_data=Test_DataGenerator(model, test_emotion_paths,test_pose_paths,test_age_paths, test_emotion,test_pose,test_age,BATCH_SIZE),
                    epochs=args.each_model_epoch,
                    verbose=2,
                    workers=NUM_WORKER,
                    use_multiprocessing=False,
                    max_queue_size=int(BATCH_SIZE * 2),
                    callbacks=callbacks)
        # if epoch!=0:
        #     print('pose-emotion',np.equal(model.get_weights()[-1],emotion_model.get_weights()[-1]))
        #     print('pose-age',np.equal(model.get_weights()[-1],age_model.get_weights()[-1]))
        print('{}_pose_train_finished'.format(epoch))


        if model.task_type==9:
            pose_model = keras.models.clone_model(model)
            pose_model.set_weights(model.get_weights())
            pose_model = freeze_all(pose_model)
            if not args.is_pesudo:
                pose_model=None
            model.fit_generator(
                DataGenerator_age(model, pose_model,pose_model,train_age_paths,train_age, BATCH_SIZE,IS_seletion,threshold_K),
                validation_data=Test_DataGenerator(model, test_emotion_paths,test_pose_paths,test_age_paths, test_emotion,test_pose,test_age,BATCH_SIZE),
                epochs=args.each_model_epoch,
                verbose=2,
                workers=NUM_WORKER,
                use_multiprocessing=False,
                max_queue_size=int(BATCH_SIZE * 2),
                callbacks=callbacks)
            # print('age-emotion',np.equal(model.get_weights()[-1],emotion_model.get_weights()[-1]))
            # print('age-pose',np.equal(model.get_weights()[-1],pose_model.get_weights()[-1]))
            print('{}_age_train_finished'.format(epoch))
        if args.freezing:
            if epoch==args.no_freezing_epoch:
                print('no freezing and start to changing learning rate')
                model = no_freeze_all(model)
                adam = keras.optimizers.Adam(lr=1*1e-9)
                print('lr changing to:',1*1e-9)
                model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
                print('epoch is{}'.format(epoch))
                model.summary()
            elif epoch == args.no_freezing_epoch+args.lr_step:
                adam = keras.optimizers.Adam(lr=1*1e-8)
                print('lr changing to:',1*1e-8)
                model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
                print('epoch is{}'.format(epoch))
                model.summary()
            elif epoch == args.no_freezing_epoch+args.lr_step*2:
                adam = keras.optimizers.Adam(lr=1*1e-7)
                print('lr changing to:',1*1e-7)
                model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
                print('epoch is{}'.format(epoch))
                model.summary()
            elif epoch == args.no_freezing_epoch+args.lr_step*3:
                adam = keras.optimizers.Adam(lr=1*1e-6)
                print('lr changing to:',1*1e-6)
                model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
                print('epoch is{}'.format(epoch))
                model.summary()
            elif epoch == args.no_freezing_epoch+args.lr_step*4:
                adam = keras.optimizers.Adam(lr=1*1e-5)
                print('lr changing to:',1*1e-5)
                model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
                print('epoch is{}'.format(epoch))
                model.summary()
            elif epoch == args.no_freezing_epoch+args.lr_step*5:
                adam = keras.optimizers.Adam(lr=1*1e-4)
                print('lr changing to:',1*1e-4)
                model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
                print('epoch is{}'.format(epoch))
                model.summary()
        
if __name__ == '__main__':
    main()
