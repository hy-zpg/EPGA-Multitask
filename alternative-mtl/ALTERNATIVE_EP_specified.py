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
from utils.STL.emotion_generator import DataGenerator as emotion_DataGenerator
from utils.STL.pose_generator import DataGenerator as pose_DataGenerator
from utils.proposed_MTL.alternate_EP_generator import DataGenerator
from utils.confusion_MTL.confusion_EP_generator import DataGenerator as Test_DataGenerator
from utils.callback import DecayLearningRate
from utils.pseudo_density_distribution.pseudo_label_weights import assign_weights
from keras.callbacks import LearningRateScheduler
from model.models import Net
from sklearn.metrics import confusion_matrix

tf_session = K.get_session() 
tf_graph = tf.get_default_graph()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','SFEW','expw'],
                    default='ferplus',
                    help='Model to be used')
parser.add_argument('--dataset_pose',
                    choices=['fer2013','aflw','sfew','expw'],
                    default='aflw',
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
                    default= 1,
                    type= float,
                    help='pose')
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
parser.add_argument('--is_pesudo_confidence',
                    type=ast.literal_eval,
                    help='whether need confidence')
parser.add_argument('--is_pesudo_density',
                    type=ast.literal_eval,
                    help='whether need density')
parser.add_argument('--is_pesudo_distribution',
                    type=ast.literal_eval,
                    help='whether need distribution')
parser.add_argument('--is_interpolation',
                     type= ast.literal_eval,
                    help='is soft weights')
parser.add_argument('--density_t',
                    default= 0.6,
                    type= float,
                    help='densty_t')
parser.add_argument('--distill_t',
                    default= 2,
                    type= int,
                    help='distill_t')
parser.add_argument('--selection_threshold',
                    default= 0.9,
                    type= float,
                    help='pesudo-label selection threshold')
parser.add_argument('--interpolation_weights',
                    default= 0.9,
                    type= float,
                    help='interpolation weights')
parser.add_argument('--cluster_k',
                    default= 4,
                    type= int,
                    help='the k value of the gmm model')



# def load_data(dataset_emotion,dataset_pose):
#     emotion = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_emotion) )
#     pose = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_pose) )
#     data_emotion = emotion
#     data_pose = pose
#     del emotion,pose
#     paths_emotion = data_emotion['full_path'].values
#     emotion_label = data_emotion['emotion'].values.astype('uint8')
#     paths_pose = data_pose['full_path'].values
#     pose_label = data_pose['pose'].values.astype('uint8')
#     return paths_emotion, paths_pose, emotion_label,pose_label
#     # return paths_emotion[:5000], paths_pose[:5000], emotion_label[:5000],pose_label[:5000]

def load_data(dataset_emotion,dataset_pose):
    emotion_train = pd.read_csv('data/db/{}_train_cleaned.csv'.format(dataset_emotion) )
    emotion_train.sample(frac=1).reset_index(drop=True)
    emotion_valid = pd.read_csv('data/db/{}_valid_cleaned.csv'.format(dataset_emotion) )
    pose = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_pose) )
    data_emotion_train = emotion_train
    data_emotion_test = emotion_valid
    data_pose = pose
    del emotion_train,emotion_valid,pose
    train_emotion_paths = data_emotion_train['full_path'].values
    train_emotion = data_emotion_train['emotion'].values.astype('uint8')
    test_emotion_paths = data_emotion_test['full_path'].values
    test_emotion = data_emotion_test['emotion'].values.astype('uint8')

    paths_pose = data_pose['full_path'].values
    pose_label = data_pose['pose'].values.astype('uint8')
    return train_emotion_paths,test_emotion_paths,paths_pose, train_emotion,test_emotion,pose_label
    # return paths_emotion[:5000], paths_pose[:5000], emotion_label[:5000],pose_label[:5000]

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
    POSE_DATASET = args.dataset_pose
   

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

    train_emotion_paths,test_emotion_paths,paths_pose, train_emotion,test_emotion,pose_label = load_data(EMOTION_DATASET,POSE_DATASET)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split_pose = kf.split(paths_pose)
    pose_kf = [[pose_train_idx,pose_test_idx] for pose_train_idx,pose_test_idx in kf_split_pose]
    pose_train_idx,pose_test_idx = pose_kf[0]


    print(len(train_emotion_paths),len(pose_train_idx))
    print(len(test_emotion_paths),len(pose_test_idx))

    train_pose_paths = paths_pose[pose_train_idx]
    train_pose = pose_label[pose_train_idx]
    test_pose_paths = paths_pose[pose_test_idx]
    test_pose = pose_label[pose_test_idx]




    model = None
    if MODEL == 'vggFace':
        print('based on vggface')
        model = Net(MODEL,1,11,emotion_classes,pose_classes,age_classes,gender_classes,args.is_dropout,args.is_bn,args.weights_decay)
        MODEL = model.name
        if args.is_freezing:
            model = freeze_all_but_mid_and_top(model)
    else:
        model = Net(MODEL,1,11,emotion_classes,pose_classes,age_classes,gender_classes,args.is_dropout,args.is_bn,args.weights_decay)
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
    if args.is_pesudo_confidence:
        is_pesudo_confidence='pesudo_confidence_true'
    else:
        is_pesudo_confidence='pesudo_confidence_false'
    if args.is_pesudo_density:
        is_pesudo_density='pesudo_density_true'
    else:
        is_pesudo_density='pesudo_density_false'
    if args.is_pesudo_distribution:
        is_pesudo_distribution='pesudo_distribution_true'
    else:
        is_pesudo_distribution='pesudo_distribution_false'
    
    if args.is_interpolation:
        is_soft_weight = 'soft_weight'
        print('this is overall soft weight')
    else:
        is_soft_weight = 'hard_weight'
        print('this is overall hard weight')


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


    main_path='{}-{}/{}/'.format(EMOTION_DATASET,POSE_DATASET,MODEL)
    # appendex_path='Net-{}_{}_{}_{}-Pesudo-{}_{}_{}_{}_{}_{}_{}_distill_t-{}_threshold_d-{}_interpolation_w-{}_cluster_k-{}_density_t-{}-LR-{}_{}_{}_{}'.format(augmentation,dropout,bn,args.weights_decay,
    #     is_naive,is_distilled,is_soft_weight,is_pesudo,is_pesudo_confidence,is_pesudo_density,is_pesudo_distribution,
    #     args.distill_t,args.selection_threshold,args.interpolation_weights,args.cluster_k,args.density_t,
    #     args.E_loss_weights,args.P_loss_weights,freezing,args.no_freezing_epoch)
    appendex_path='Pesudo-{}_{}_{}_{}_{}_{}_{}_distill_t-{}_threshold_d-{}_interpolation_w-{}_cluster_k-{}_density_t-{}-LR-{}_{}_{}_{}'.format(
        is_naive,is_distilled,is_soft_weight,is_pesudo,is_pesudo_confidence,is_pesudo_density,is_pesudo_distribution,
        args.distill_t,args.selection_threshold,args.interpolation_weights,args.cluster_k,args.density_t,
        args.E_loss_weights,args.P_loss_weights,freezing,args.no_freezing_epoch)
    weights_path = './train_weights/ITERATION/'+main_path+appendex_path
    logs_path = './train_log/ITERATION/'+main_path+appendex_path
    matrix_path = './train_log/ITERATION/'+main_path+appendex_path+'/matrix/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path)




    model_names = weights_path +'.{val_emotion_prediction_my_acc:.4f}-{val_pose_prediction_my_acc:.4f}.hdf5'
    csv_name = logs_path + '.log'
    checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=True)
    csvlogger=CSVLogger(filename=csv_name,append=True)
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(PATIENCE), verbose=1)
    tensorboard = TensorBoard(log_dir=logs_path,batch_size=BATCH_SIZE)
    callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]
    print('whether freezing:',args.is_freezing)
    if args.is_pesudo or args.is_interpolation or args.is_distilled:
        each_term_epoch = 2
    else:
        each_term_epoch = 1
    if not args.is_freezing:
        adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # adam = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        model = no_freeze_all(model)
        model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
        model.summary()
        for epoch in  range(EPOCH):
            if epoch==0:
                print('initializing')
                model.fit_generator(
                Test_DataGenerator(model,None,train_emotion_paths,train_pose_paths,train_emotion,train_pose,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,args.is_augmentation,None,None),
                validation_data = Test_DataGenerator(model, None,test_emotion_paths,test_pose_paths, test_emotion,test_pose,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,False,None,None),
                epochs=6,
                verbose=2,
                workers=1,
                use_multiprocessing=False,
                max_queue_size=int(BATCH_SIZE * 2),
                callbacks=callbacks)
                print('initializing finished')
            else:
               # model_pre = keras.models.clone_model(model)
                model_pre = Model(inputs=model.input,outputs=model.output)
                model_pre.set_weights(model.get_weights())
                if epoch%2==0:
                    is_emotion=True
                    if args.is_naive:
                        model.compile(optimizer=adam, loss=loss_dict,loss_weights=[1,0],metrics=metrics_dict)
                else:
                    is_emotion=False
                    if args.is_naive:
                        model.compile(optimizer=adam, loss=loss_dict,loss_weights=[0,1],metrics=metrics_dict)
                if args.is_pesudo or args.is_interpolation or args.is_distilled:
                    is_hard_weight = not args.is_interpolation
                    pesudo_data = assign_weights(model,is_emotion,train_emotion_paths,train_emotion,train_pose_paths,train_pose,args.cluster_k,args.is_augmentation,args.selection_threshold,args.is_distilled,args.is_pesudo_confidence,args.is_pesudo_density,args.is_pesudo_distribution,is_hard_weight,args.interpolation_weights,args.distill_t,args.density_t)
                else:
                    pesudo_data=None
                model.fit_generator(
                    DataGenerator(model,model_pre,train_emotion_paths,train_pose_paths,train_emotion,train_pose,args.batch_size,is_emotion,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,args.is_augmentation,pesudo_data),
                    validation_data = Test_DataGenerator(model, None,test_emotion_paths,test_pose_paths, test_emotion,test_pose,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,False,None,None),
                    epochs=each_term_epoch,
                    verbose=2,
                    workers=1,
                    use_multiprocessing=False,
                    max_queue_size=int(BATCH_SIZE * 2),
                    callbacks=callbacks
                )
                if epoch%2==0:
                    print('{}_emotion_train_finished'.format(epoch))
                else:
                    print('{}_pose_train_finished'.format(epoch))
            emotion_predict = model.predict_generator(emotion_DataGenerator(model,test_emotion_paths,test_emotion,args.batch_size))
            emotion_reuslt=np.argmax(emotion_predict[0],axis=1)
            emotion_matrix=confusion_matrix(test_emotion,emotion_reuslt)
            pose_predict = model.predict_generator(pose_DataGenerator(model,test_pose_paths,test_pose,args.batch_size))
            pose_reuslt=np.argmax(pose_predict[1],axis=1)
            pose_matrix=confusion_matrix(test_pose,pose_reuslt)
            print('emotion confusion matrix:',emotion_matrix)
            print('pose confusion matrix:',pose_matrix)
            np.savetxt(matrix_path+'{}_emotion_matrix.txt'.format(epoch),emotion_matrix)
            np.savetxt(matrix_path+'{}_pose_matrix.txt'.format(epoch),pose_matrix) 

    else:
        model.compile(optimizer='adam', loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
        model.summary()
        for epoch in  range(EPOCH):
            if epoch==0:
                print('initializing')
                model.fit_generator(
                Test_DataGenerator(model,None,train_emotion_paths,train_pose_paths,train_emotion,train_pose,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,args.is_augmentation,None,None),
                validation_data = Test_DataGenerator(model, None,test_emotion_paths,test_pose_paths, test_emotion,test_pose,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,False,None,None),
                epochs=3,
                verbose=2,
                workers=1,
                use_multiprocessing=False,
                max_queue_size=int(BATCH_SIZE * 2),
                callbacks=callbacks)
                print('initializing finished')
            else:
               # model_pre = keras.models.clone_model(model)
                model_pre = Model(inputs=model.input,outputs=model.output)
                model_pre.set_weights(model.get_weights())
                if epoch%2==0:
                    is_emotion=True
                    if args.is_naive:
                        if epoch < args.no_freezing_epoch:
                            model.compile(optimizer='adam', loss=loss_dict,loss_weights=[1,0],metrics=metrics_dict)
                        else:
                            adam = keras.optimizers.Adam(lr=1*1e-4)
                            model.compile(optimizer=adam, loss=loss_dict,loss_weights=[1,0],metrics=metrics_dict)

                else:
                    is_emotion=False
                    if args.is_naive:
                        if epoch<args.no_freezing_epoch:
                            model.compile(optimizer='adam', loss=loss_dict,loss_weights=[0,1],metrics=metrics_dict)
                        else:
                            adam = keras.optimizers.Adam(lr=1*1e-4)
                            model.compile(optimizer=adam, loss=loss_dict,loss_weights=[0,1],metrics=metrics_dict)
                if args.is_pesudo or args.is_interpolation or args.is_distilled:
                    is_hard_weight = not args.is_interpolation
                    pesudo_data = assign_weights(model,is_emotion,train_emotion_paths,train_emotion,train_pose_paths,train_pose,args.cluster_k,args.is_augmentation,args.selection_threshold,args.is_distilled,args.is_pesudo_confidence,args.is_pesudo_density,args.is_pesudo_distribution,is_hard_weight,args.interpolation_weights,args.distill_t,args.density_t)
                else:
                    pesudo_data = None
                model.fit_generator(
                    DataGenerator(model,model_pre,train_emotion_paths,train_pose_paths,train_emotion,train_pose,args.batch_size,is_emotion,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,args.is_augmentation,pesudo_data),
                    validation_data = Test_DataGenerator(model, None,test_emotion_paths,test_pose_paths, test_emotion,test_pose,args.batch_size,args.is_distilled,args.is_pesudo,args.is_interpolation,args.selection_threshold,args.interpolation_weights,False,None,None),
                    epochs=each_term_epoch,
                    verbose=2,
                    workers=1,
                    use_multiprocessing=False,
                    max_queue_size=int(BATCH_SIZE * 2),
                    callbacks=callbacks
                )
                if epoch%2==0:
                    print('{}_emotion_train_finished'.format(epoch))
                else:
                    print('{}_pose_train_finished'.format(epoch))
                
                if epoch==args.no_freezing_epoch:
                    print('no freezing and start to changing learning rate')
                    model = no_freeze_all(model)
                    adam=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                    print('lr changing to:',1*1e-5)
                    model.compile(optimizer=adam, loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
                    print('epoch is{}'.format(epoch))
                    model.summary() 
            
            emotion_predict = model.predict_generator(emotion_DataGenerator(model,test_emotion_paths,test_emotion,args.batch_size))
            emotion_reuslt=np.argmax(emotion_predict[0],axis=1)
            emotion_matrix=confusion_matrix(test_emotion,emotion_reuslt)
            pose_predict = model.predict_generator(pose_DataGenerator(model,test_pose_paths,test_pose,args.batch_size))
            pose_reuslt=np.argmax(pose_predict[1],axis=1)
            pose_matrix=confusion_matrix(test_pose,pose_reuslt)
            print('emotion confusion matrix:',emotion_matrix)
            print('pose confusion matrix:',pose_matrix)
            np.savetxt(matrix_path+'{}_emotion_matrix.txt'.format(epoch),emotion_matrix)
            np.savetxt(matrix_path+'{}_pose_matrix.txt'.format(epoch),pose_matrix) 
if __name__ == '__main__':
    main()
