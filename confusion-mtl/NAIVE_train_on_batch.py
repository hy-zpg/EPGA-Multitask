import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import tqdm
from keras import backend as K
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from model.inceptionv3 import EmotionNetInceptionV3
from model.mobilenetv2 import MultitaskMobileNetV2
from model.vggface import MultitaskVGGFacenet,AgenderNetVGGFacenet,EmotionNetVGGFacenet
from model.mini_xception import EmotionNetmin_XCEPTION
from utils.datasets import DataManager 
from utils.naive_MTL.original_alternate_generator_emotion import DataGenerator_emotion
from utils.naive_MTL.original_alternate_generator_gender_age import DataGenerator_gender_age
# from utils.confusion_MTL.confusion_generator import DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['ferplus','ferplus','sfew'],
                    default='ferplus',
                    help='Model to be used')
parser.add_argument('--dataset_gender_age',
                    choices=['imdb','adience','fgnet'],
                    default='adience',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
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
parser.add_argument('--lr',
                    default=0.01,
                    type=float,
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


def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:19]:
        layer.trainable = False
    for layer in model.layers[19:]:
        layer.trainable = True
    return model

def freeze_gender_age_branch(model,model_name):
    if model_name == 'vggFace':
        model.layers[22].trainable = False
        model.layers[23].trainable = False
        model.layers[25].trainable = False
        model.layers[26].trainable = False
        model.layers[21].trainable = True
        model.layers[24].trainable = True
    else:
        model.layers[44].trainable = False
        model.layers[45].trainable = False
        model.layers[43].trainable = True
    return model

def freeze_emotion_branch(model,model_name):
    if model_name == 'vggFace':
        model.layers[21].trainable = False
        model.layers[24].trainable = False
        model.layers[22].trainable = True
        model.layers[23].trainable = True
        model.layers[25].trainable = True
        model.layers[26].trainable = True
    else:
        model.layers[43].trainable = False
        model.layers[44].trainable = True
        model.layers[45].trainable = True
    return model

def write_log(callback,names,logs,batch_no):
    for name,value in zip(names,logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary,batch_no)
        callback.writer.flush()

def path_file(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL):
    logs_path = './train_logs/NAIVE/{}-{}/{}/'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
    weights_path = './train_weights/NAIVE/{}-{}/{}/'.format(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    return logs_path,weights_path

def store_model_choice(model,EMOTION,GENDER,AGE,emotion_acc,gender_acc,age_acc,store_path,epoch):
    dis1 = 1 if EMOTION < emotion_acc else 0
    dis2 = 1 if GENDER < gender_acc else 0
    dis3 = 1 if AGE < age_acc else 0
    if (dis1+dis2+dis3)<2:
        return EMOTION,GENDER,AGE
    else:
        acc = str(emotion_acc)+'-'+str(gender_acc)+'-'+str(age_acc)
        model.save(store_path+str(epoch)+'_'+acc+'.hdf5')
        return emotion_acc,gender_acc,age_acc

def test_model(model,callback,emotion_valid_name,gender_valid_name,age_valid_name,emotion_generater,gender_age_generater,BATCH_NB_TEST_1,BATCH_NB_TEST_2,EMOTION_ACC,GENDER_ACC,AGE_ACC,stored_model_path,epoch_nb):
    Losses_test_emotion = []
    Losses_test_gender_age = []
    for batch_nb in range(BATCH_NB_TEST_1):
        Image_data_1, Labels_1 = emotion_generater.__getitem__(batch_nb)
        emotion_loss = model.test_on_batch(Image_data_1,
            [Labels_1['emotion_prediction'],Labels_1['gender_prediction'],Labels_1['age_prediction']])
        loss = [emotion_loss[1],emotion_loss[4]]
        # write_log(callback,emotion_valid_name,loss,batch_nb)
        Losses_test_emotion.append(emotion_loss)
    for batch_nb in range(BATCH_NB_TEST_2):
        Image_data_2, Labels_2 = gender_age_generater.__getitem__(batch_nb)
        gender_age_loss = model.test_on_batch(Image_data_2,
            [Labels_2['emotion_prediction'],Labels_2['gender_prediction'],Labels_2['age_prediction']])
        loss1 = [gender_age_loss[2],gender_age_loss[5]]
        # write_log(callback,gender_valid_name,loss1,batch_nb)
        loss2 = [gender_age_loss[3],gender_age_loss[6]]
        # write_log(callback,age_valid_name,loss2,batch_nb)
        Losses_test_gender_age.append(gender_age_loss) 
    emotion_loss = np.array(Losses_test_emotion).mean(axis=0)[1]
    gender_loss = np.array(Losses_test_gender_age).mean(axis=0)[2]
    age_loss = np.array(Losses_test_gender_age).mean(axis=0)[3]
    total_loss = emotion_loss + gender_loss + age_loss
    emotion_acc = np.array(Losses_test_emotion).mean(axis=0)[4]
    gender_acc = np.array(Losses_test_gender_age).mean(axis=0)[5]
    age_acc = np.array(Losses_test_gender_age).mean(axis=0)[6]
    valid_result = [total_loss,emotion_loss,gender_loss,age_loss,emotion_acc,gender_acc,age_acc]
    
    valid_average_emotion_acc= round(emotion_acc,2)
    valid_average_gender_acc= round(gender_acc,2)
    valid_average_age_acc= round(age_acc,2)
    EMOTION_ACC,GENDER_ACC,AGE_ACC = store_model_choice(model,EMOTION_ACC,GENDER_ACC,AGE_ACC,valid_average_emotion_acc,valid_average_gender_acc,valid_average_age_acc,stored_model_path,epoch_nb)
    return valid_result,EMOTION_ACC,GENDER_ACC,AGE_ACC

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

    train_names_1=['train_total_loss_1','train_emotion_loss_1','train_gender_loss_1','train_age_loss_1',
    'train_emotion_acc_1','train_gender_acc_1','train_age_acc_1']
    train_names_2=['train_total_loss_2','train_emotion_loss_2','train_gender_loss_2','train_age_loss_2',
    'train_emotion_acc_2','train_gender_acc_2','train_age_acc_2']
    valid_names_1 = ['valid_total_loss','valid_emotion_loss_1','valid_gender_loss_1','valid_age_loss_1', 
    'valid_emotion_acc_1','valid_gender_acc_1','valid_age_acc_1']
    valid_names_2 = ['valid_total_loss','valid_emotion_loss_2','valid_gender_loss_2','valid_age_loss_2',
    'valid_emotion_acc_2','valid_gender_acc_2','valid_age_acc_2']
    emotion_valid_name = ['valid_emotion_loss','valid_emotion_loss']
    gender_valid_name = ['valid_gender_loss','valid_gender_loss']
    age_valid_name = ['valid_age_loss','valid_age_loss']
    train_names=['train_total_loss','train_emotion_loss','train_gender_loss','train_age_loss',
    'train_emotion_acc','train_gender_acc','train_age_acc']
    valid_names = ['valid_total_loss','valid_emotion_loss','valid_gender_loss','valid_age_loss', 
    'valid_emotion_acc','valid_gender_acc','valid_age_acc']
   


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
    else:
        gender_classes = 2
        age_classes = 8
    
    model = None
    if MODEL == 'vggFace':
        model = Net(MODEL,1,4,emotion_classes,gender_classes,age_classes)
        model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        model = Net(MODEL,1,4,emotion_classes,gender_classes,age_classes)
        MODEL = model.name


    if GENDER_AGE_DATASET == 'imdb':
        losses = {
        "emotion_prediction": "categorical_crossentropy",
        "gender_prediction":"categorical_crossentropy",
        "age_prediction":"categorical_crossentropy"
        }
        metrics = {
            "emotion_prediction": "acc",
            "gender_prediction": "acc",
            "age_prediction": mae
        }

    else:
        losses = {
            "emotion_prediction": "categorical_crossentropy",
            "gender_prediction":"categorical_crossentropy",
            "age_prediction":"categorical_crossentropy"
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
    
    # emotion_trained_model = freeze_gender_age_branch(model,args.model)
    # gender_age_trained_model = freeze_emotion_branch(model,args.model)
    # emotion_trained_model.compile(optimizer='adam', loss=losses, loss_weights = [1,0,0],metrics=metrics)
    # gender_age_trained_model.compile(optimizer='adam', loss=losses, loss_weights = [0,1,1],metrics=metrics)
    if GENDER_AGE_DATASET == 'imdb':
        model.compile(optimizer='adam', loss=losses,loss_weights = [1,1,0.5],metrics=metrics)
    else:   
        model.compile(optimizer='adam', loss=losses,metrics=metrics)


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

    # model.summary()
    # model = freeze_gender_age_branch(model,args.model)
    # model.summary()
    # model = freeze_emotion_branch(model,args.model)
    # model.summary()
    emotion_generater = DataGenerator_emotion(model,train_emotion_paths,train_emotion, BATCH_SIZE)
    emotion_test_generater = DataGenerator_emotion(model,test_emotion_paths,test_emotion, BATCH_SIZE)
    gender_age_generater = DataGenerator_gender_age(model,train_gender_age_paths,train_gender,train_age,BATCH_SIZE)
    gender_age_test_generater = DataGenerator_gender_age(model,test_gender_age_paths,test_gender,test_age,BATCH_SIZE)

    
    #prepare the  parameter of network
    base_lr = args.lr
    rmsprop=keras.optimizers.Adam(lr=base_lr)
    EPOCH_NB = EPOCH
    BATCH_NB_1 = int(len(train_emotion_paths)/BATCH_SIZE)
    BATCH_NB_2 = int(len(train_gender_age_paths)/BATCH_SIZE)
    BATCH_NB_TEST_1 =  int(len(test_emotion_paths)/BATCH_SIZE)
    BATCH_NB_TEST_2 =  int(len(test_gender_age_paths)/BATCH_SIZE)
    print (BATCH_NB_1,BATCH_NB_2,BATCH_NB_TEST_1,BATCH_NB_TEST_2)
    lr_schedule = [base_lr]*50 + [base_lr*0.1]*40 + [base_lr*0.01]*30+[base_lr*0.001]*20+[base_lr*0.0001]*10+[base_lr*0.00001]*10000
    
    log_path,stored_model_path = path_file(EMOTION_DATASET,GENDER_AGE_DATASET,MODEL)
    callback = TensorBoard(log_path)
    callback.set_model(model)

    


    EMOTION_ACC = 0
    GENDER_ACC = 0
    AGE_ACC = 0
    for epoch_nb in range(EPOCH):
        if epoch_nb>1 and lr_schedule[epoch_nb] != lr_schedule[epoch_nb-1]:
            K.get_session().run(model.optimizer.lr.assign(lr_schedule[epoch_nb])) 
        

        Losses_1=[]
        Losses_2 = []
        
        # model = freeze_gender_age_branch(model,args.model)
        # model.compile(optimizer='adam', loss=losses,loss_weights = [1,0,0],metrics=metrics)
        # model.summary()
        for batch_nb in range(BATCH_NB_1):  
            [Image_data_1, Labels_1] = emotion_generater.__getitem__(batch_nb)
            losses_1 = model.train_on_batch(Image_data_1,
                [Labels_1['emotion_prediction'],Labels_1['gender_prediction'],Labels_1['age_prediction']])
            Losses_1.append(losses_1)
        #test
        valid_result,EMOTION_ACC,GENDER_ACC,AGE_ACC = test_model(model,callback,emotion_valid_name,gender_valid_name,age_valid_name,
            emotion_test_generater,gender_age_test_generater,BATCH_NB_TEST_1,BATCH_NB_TEST_2,EMOTION_ACC,GENDER_ACC,AGE_ACC,stored_model_path,epoch_nb)
        loss_epoch = np.array(Losses_1).mean(axis=0)
        write_log(callback,train_names,loss_epoch,2*epoch_nb)
        write_log(callback,valid_names,valid_result,2*epoch_nb)
        print  ('Done for Epoch %d.'% epoch_nb)
        print  ('data1_train_result:',np.array(Losses_1).mean(axis=0))
        print  ('valid_result:',valid_result)
        

        # model = freeze_emotion_branch(model,args.model)
        # model.compile(optimizer='adam', loss=losses,loss_weights = [0,1,1],metrics=metrics)
        # model .summary()
        for batch_nb in range(BATCH_NB_2):
            [Image_data_2, Labels_2] = gender_age_generater.__getitem__(batch_nb)
            losses_2 = model.train_on_batch(Image_data_2,
                    [Labels_2['emotion_prediction'],Labels_2['gender_prediction'],Labels_2['age_prediction']])
            # write_log(callback,train_names_2,losses_2,batch_nb)
            Losses_2.append(losses_2)
        #test
        valid_result,EMOTION_ACC,GENDER_ACC,AGE_ACC = test_model(model,callback,emotion_valid_name,gender_valid_name,age_valid_name,
            emotion_test_generater,gender_age_test_generater,BATCH_NB_TEST_1,BATCH_NB_TEST_2,EMOTION_ACC,GENDER_ACC,AGE_ACC,stored_model_path,epoch_nb)
        loss_epoch = np.array(Losses_2).mean(axis=0)
        write_log(callback,train_names,loss_epoch,2*epoch_nb+1)
        write_log(callback,valid_names,valid_result,2*epoch_nb+1)

        print  ('Done for Epoch %d.'% epoch_nb)
        print  ('data2_train_result:',np.array(Losses_2).mean(axis=0))
        print  ('valid_result:',valid_result)
        

        

if __name__ == '__main__':
    main()
