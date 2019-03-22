import argparse
import os 
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import ast
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from utils.datasets import DataManager 
from utils.STL.emotion_generator import DataGenerator as emotion_DataGenerator
from utils.STL.age_generator import DataGenerator as age_DataGenerator
from utils.confusion_MTL.confusion_EPGA_generator import DataGenerator
from utils.callback import DecayLearningRate
from keras.callbacks import LearningRateScheduler
from model.models import Net
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','sfew','expw'],
                    default='fer2013',
                    help='Model to be used')
parser.add_argument('--dataset_pose',
                    choices=['aflw','ferplus','sfew','expw'],
                    default='aflw',
                    help='Model to be used')
parser.add_argument('--dataset_gender_age',
                    choices=['adience'],
                    default='adience',
                    help='gender age datasets')
parser.add_argument('--model',
                    choices=[ 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=64,
                    type=int,
                    help='Num of training epoch')
parser.add_argument('--batch_size',
                    default=128,
                    type=int,
                    help='Size of data batch to be used')
parser.add_argument('--num_worker',
                    default=1,
                    type=int,
                    help='Number of worker to process data')

#add training trick
parser.add_argument('--patience',
                    default=10,
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
parser.add_argument('--E_loss_weights',
                    default= 5,
                    type= float,
                    help='emotion')
parser.add_argument('--P_loss_weights',
                    default= 5,
                    type= float,
                    help='pose')
parser.add_argument('--G_loss_weights',
                    default= 1,
                    type= float,
                    help='gender')
parser.add_argument('--A_loss_weights',
                    default= 1,
                    type= float,
                    help='age')
# parser.add_argument('--is_based_STL',
#                     type=ast.literal_eval,
#                     help='whether need  pre-trained on STL')
# parser.add_argument('--emotion_path',
#                     type=str,
#                     help='emotion model')
# parser.add_argument('--pose_path',
#                     type=str,
#                     help='pose model')


#pesudo label strategy
parser.add_argument('--is_naive',
                    type=ast.literal_eval,
                    help='whether naive')
parser.add_argument('--is_distilled',
                    type= ast.literal_eval,
                    help='whether distilled')
parser.add_argument('--is_pesudo',
                    type=ast.literal_eval,
                    help='whether need psudo-label')
parser.add_argument('--is_interpolation',
                     type= ast.literal_eval,
                    help='whether need dropout')
parser.add_argument('--selection_threshold',
                    default= 0.9,
                    type= float,
                    help='pesudo-label selection threshold')
parser.add_argument('--interpolation_weights',
                    default= 0.5,
                    type= float,
                    help='interpolation weights')



def load_data(dataset_emotion,dataset_gender_age,dataset_pose):
    emotion = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_emotion) )
    pose = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_pose) )
    gender_age = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_gender_age) )
    data_emotion = emotion
    data_pose = pose
    data_gedner_age = gender_age
    del emotion,gender_age,pose
    paths_emotion = data_emotion['full_path'].values
    emotion_label = data_emotion['emotion'].values.astype('uint8')

    paths_pose = data_pose['full_path'].values
    pose_label = data_pose['pose'].values.astype('uint8')


    paths_gender_age = data_gedner_age['full_path'].values
    gender_label = data_gedner_age['gender'].values.astype('uint8')
    age_label = data_gedner_age['age'].values.astype('uint8')
    return paths_emotion, paths_pose, paths_gender_age, emotion_label, pose_label, gender_label,age_label



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

# def base_STL_model(MODEL,is_EP,emotion_path,pose_path,age_path,emotion_classes,pose_classes,age_classes,gender_classes):
#     emotion_model=Net(MODEL,1,0,emotion_classes,pose_classes,age_classes,gender_classes)
#     pose_model=Net(MODEL,1,5,emotion_classes,pose_classes,age_classes,gender_classes)
#     age_model=Net(MODEL,1,1,emotion_classes,pose_classes,age_classes,gender_classes)
#     emotion_model.load_weights(emotion_path)
#     pose_model.load_weights(pose_path)
#     age_model.load_weights(age_path)
#     if is_EP:
#         model= Net(MODEL,1,11,emotion_classes,pose_classes,age_classes,gender_classes)
#     else:
#         model= Net(MODEL,1,9,emotion_classes,pose_classes,age_classes,gender_classes)

#     if is_EP:
#             #flatten common_fc
#         model.layers[19].set_weights(emotion_model.layers[19].get_weights())
#         model.layers[20].set_weights(emotion_model.layers[20].get_weights())

#         # emotion_FC pose_fc age_FC
#         model.layers[21].set_weights(emotion_model.layers[21].get_weights())
#         model.layers[22].set_weights(pose_model.layers[21].get_weights())

#         # emotion_prediction pose_prediction age_prediction
#         model.layers[23].set_weights(emotion_model.layers[22].get_weights())
#         model.layers[24].set_weights(pose_model.layers[22].get_weights())
#         return model
#     else:

#         #flatten common_fc
#         model.layers[19].set_weights(emotion_model.layers[19].get_weights())
#         model.layers[20].set_weights(emotion_model.layers[20].get_weights())

#         # emotion_FC pose_fc age_FC
#         model.layers[21].set_weights(emotion_model.layers[21].get_weights())
#         model.layers[22].set_weights(pose_model.layers[21].get_weights())
#         model.layers[23].set_weights(age_model.layers[21].get_weights())

#         # emotion_prediction pose_prediction age_prediction
#         model.layers[24].set_weights(emotion_model.layers[22].get_weights())
#         model.layers[25].set_weights(pose_model.layers[22].get_weights())
#         model.layers[26].set_weights(age_model.layers[22].get_weights())
#     return model

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

    
    gender_classes = 2
    age_classes = 8
    pose_classes = 5
   
    # emotion_path='train_weights/EmotionNetVGGFace_vgg16/expw/2__01-0.64.hdf5'
    # pose_path = 'train_weights/PoseNetVGGFace_vgg16/aflw/1__01-0.77.hdf5'
    # age_path = 'train_weights/AgeNetVGGFace_vgg16/adience/1__01-0.67.hdf5'

    paths_emotion, paths_pose, paths_gender_age, emotion_label, pose_label, gender_label,age_label = load_data(EMOTION_DATASET,GENDER_AGE_DATASET,args.dataset_pose)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split_gender_age = kf.split(paths_gender_age)
    kf_split_emotion = kf.split(paths_emotion)
    kf_split_pose = kf.split(paths_pose)

    emotion_kf = [[emotion_train_idx,emotion_test_idx] for emotion_train_idx,emotion_test_idx in kf_split_emotion]
    gender_age_kf = [[gender_age_train_idx,gender_age_test_idx] for gender_age_train_idx,gender_age_test_idx in kf_split_gender_age]
    pose_kf = [[pose_train_idx,pose_test_idx] for pose_train_idx,pose_test_idx in kf_split_pose]

    emotion_train_idx,emotion_test_idx = emotion_kf[0]
    pose_train_idx,pose_test_idx = pose_kf[0]
    gender_age_train_idx,gender_age_test_idx = gender_age_kf[0]


    print(len(emotion_train_idx),len(gender_age_train_idx),len(pose_train_idx))
    print(len(emotion_test_idx),len(gender_age_test_idx),len(pose_test_idx))


    train_emotion_paths = paths_emotion[emotion_train_idx]
    train_emotion = emotion_label[emotion_train_idx]
    test_emotion_paths = paths_emotion[emotion_test_idx]
    test_emotion = emotion_label[emotion_test_idx]

    train_pose_paths = paths_pose[pose_train_idx]
    train_pose = pose_label[pose_train_idx]
    test_pose_paths = paths_pose[pose_test_idx]
    test_pose = pose_label[pose_test_idx]

    train_gender_age_paths = paths_gender_age[gender_age_train_idx]
    train_gender = gender_label[gender_age_train_idx]
    train_age = age_label[gender_age_train_idx]
    test_gender_age_paths = paths_gender_age[gender_age_test_idx]
    test_gender = gender_label[gender_age_test_idx]
    test_age = age_label[gender_age_test_idx]






    model = None
    if MODEL == 'vggFace':
        print('based on vggface')
        model = Net(MODEL,1,12,emotion_classes,pose_classes,age_classes,gender_classes,args.is_dropout,args.is_bn,args.weights_decay)
        MODEL = model.name
        if args.is_freezing:
            model = freeze_all_but_mid_and_top(model)
    else:
        model = Net(MODEL,1,12,emotion_classes,pose_classes,age_classes,gender_classes,args.is_dropout,args.is_bn,args.weights_decay)
        MODEL = model.name
        if args.is_freezing:
            model = freeze_all_but_mid_and_top(model)

    
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

    prediction.append('gender_prediction')
    loss.append(my_cross_loss)
    acc.append(my_acc)
    loss_weights.append(args.G_loss_weights)

    prediction.append('age_prediction')
    loss.append(my_cross_loss)
    acc.append(my_acc)
    loss_weights.append(args.A_loss_weights)

    loss_dict = dict(zip(prediction, loss))
    metrics_dict = dict(zip(prediction, acc))

    

     #pesudo label
    if args.is_naive:
        is_naive='naive_true'
        print('this is naive method')
    else:
        is_naive='naive_false'
    if args.is_distilled:
        is_distilled='distilled_true'
        print('this is distiiled method')
    else:
        is_distilled='distilled_false'
    if args.is_pesudo:
        is_pesudo='pesudo_true'
        print('this is pesudo method')
    else:
        is_pesudo='pesudo_false'
    if args.is_interpolation:
        is_interpolation = 'interpolation_true'
        print('this is interpolation method')
    else:
        is_interpolation = 'interpolation_false'
    if args.selection_threshold>0:
        print('this is pesudo selection method')

    #training trich
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
    
    #import parameter
    if args.is_freezing:
        freezing='freezing_true'
    else:
        freezing='freezing_false'


    main_path='{}-{}-{}/{}/'.format(EMOTION_DATASET,GENDER_AGE_DATASET,args.dataset_pose,MODEL)
    appendex_path='net-{}_{}_{}_{}-pesudo-{}_{}_{}_{}_{}_{}-lr-{}_{}_{}_{}_{}_'.format(augmentation,dropout,bn,args.weights_decay,
        is_naive,is_distilled,is_pesudo,is_interpolation,args.selection_threshold,args.interpolation_weights,
        args.E_loss_weights,args.G_loss_weights,args.A_loss_weights,freezing,args.no_freezing_epoch)
    weights_path = './train_weights/CONFUSION/'+main_path+appendex_path
    logs_path = './train_log/CONFUSION/'+main_path+appendex_path
    matrix_path = './train_log/CONFUSION/'+main_path+appendex_path+'/matrix/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path)




    model_names = weights_path +'.{epoch:02d}-{val_emotion_prediction_my_acc:.4f}-{val_pose_prediction_my_acc:.4f}-{val_gender_prediction_my_acc:.4f}-{val_age_prediction_my_acc:.4f}.hdf5'
    csv_name = logs_path + '.log'
    checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=False)
    csvlogger=CSVLogger(filename=csv_name,append=True)
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(PATIENCE), verbose=1)
    tensorboard = TensorBoard(log_dir=logs_path,batch_size=BATCH_SIZE)
    callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]

    print('whether freezing:',args.is_freezing)
    if not args.is_freezing:
        adam=keras.optimizers.Adam(lr=0.0001)
        model = no_freeze_all(model)
        model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
        model.summary()
        for epoch in  range(EPOCH):
            if epoch==0 or args.is_naive:
                model_pre = None
            else:
                model_pre = keras.models.clone_model(model)
                model_pre.set_weights(model.get_weights())
            
            model.fit_generator(
                DataGenerator(model,model_pre,train_emotion_paths,train_pose_paths,train_gender_age_paths,train_emotion,train_pose,train_gender,train_age,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,is_augmentation=args.is_augmentation),
                validation_data = DataGenerator(model, None,test_emotion_paths,test_pose_paths,test_gender_age_paths, test_emotion,test_pose,test_gender,test_age,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,is_augmentation=False),
                epochs=1,
                verbose=2,
                workers=1,
                use_multiprocessing=False,
                max_queue_size=int(BATCH_SIZE * 2),
                callbacks=callbacks)
            
            print('{}_train_finished'.format(epoch))
            
            emotion_predict = model.predict_generator(emotion_DataGenerator(model,test_emotion_paths,test_emotion,args.batch_size))
            emotion_reuslt=np.argmax(emotion_predict[0],axis=1)
            emotion_matrix=confusion_matrix(test_emotion,emotion_reuslt)
            age_predict = model.predict_generator(age_DataGenerator(model,test_gender_age_paths,test_age,args.batch_size))
            age_reuslt=np.argmax(age_predict[2],axis=1)
            age_matrix=confusion_matrix(test_age,age_reuslt)
            print('emotion confusion matrix:',emotion_matrix)
            print('age confusion matrix:',age_matrix)
            np.savetxt(matrix_path+'{}_emotion_matrix.txt'.format(epoch),emotion_matrix)
            np.savetxt(matrix_path+'{}_age_matrix.txt'.format(epoch),age_matrix) 
    else:
        model.compile(optimizer='adam', loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
        model.summary()
        for epoch in  range(EPOCH):
            if epoch == 0 or args.is_naive:
                model_pre = None
            else:
                model_pre = keras.models.clone_model(model)
                model_pre.set_weights(model.get_weights())
            model.fit_generator(
            DataGenerator(model,model_pre,train_emotion_paths,train_pose_paths,train_gender_age_paths,train_emotion,train_pose,train_gender,train_age,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,is_augmentation=args.is_augmentation),
            validation_data = DataGenerator(model, None,test_emotion_paths,test_pose_paths,test_gender_age_paths, test_emotion,test_pose,test_gender,test_age,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,is_augmentation=False),
            epochs=1,
            verbose=2,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks)    

            if epoch==args.no_freezing_epoch:
                print('no freezing and start to changing learning rate')
                model = no_freeze_all(model)
                adam = keras.optimizers.Adam(lr=1*1e-4)
                print('lr changing to:',1*1e-4)
                model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
                print('epoch is{}'.format(epoch))
                model.summary()   

            print('{}_train_finished'.format(epoch))

            emotion_predict = model.predict_generator(emotion_DataGenerator(model,test_emotion_paths,test_emotion,args.batch_size))
            emotion_reuslt=np.argmax(emotion_predict[0],axis=1)
            emotion_matrix=confusion_matrix(test_emotion,emotion_reuslt)
            age_predict = model.predict_generator(age_DataGenerator(model,test_gender_age_paths,test_age,args.batch_size))
            age_reuslt=np.argmax(age_predict[2],axis=1)
            age_matrix=confusion_matrix(test_age,age_reuslt)
            print('emotion confusion matrix:',emotion_matrix)
            print('age confusion matrix:',age_matrix)
            np.savetxt(matrix_path+'{}_emotion_matrix.txt'.format(epoch),emotion_matrix)
            np.savetxt(matrix_path+'{}_age_matrix.txt'.format(epoch),age_matrix) 
if __name__ == '__main__':
    main()
