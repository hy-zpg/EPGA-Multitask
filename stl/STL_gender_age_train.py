import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from model.inceptionv3 import AgenderNetInceptionV3
from model.mobilenetv2 import AgenderNetMobileNetV2
from model.ssrnet import AgenderSSRNet
from model.vggface import AgenderNetVGGFacenet
from model.mini_xception import AgenderNetmin_XCEPTION
from utils.STL.gender_age_generator import DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    choices=['imdb','adience','sfew'],
                    default='adience',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=100,
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


def load_data(dataset):
    file_path = 'data/db/{}_cleaned.csv'.format(dataset) 
    data1 = pd.read_csv(file_path)
    data = data1
    del data1
    paths = data['full_path'].values
    age_label = data['age'].values.astype('uint8')
    gender_label = data['gender'].values.astype('uint8')
    return paths, age_label, gender_label

def mae(y_true, y_pred):
    """Custom MAE for 101 age class, apply softmax regression

    Parameters
    ----------
    y_true : tensor
        ground truth
    y_pred : tensor
        prediction from model

    Returns
    -------
    float
        MAE score
    """

    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_true, axis=1)), axis=-1)


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

    paths, age_label, gender_label = load_data(DATASET)
    n_fold = 1

    if DATASET == 'imdb':
        gender_classes = 2
        age_classes = 101
        losses = {
            "age_prediction": "categorical_crossentropy",
            "gender_prediction": "categorical_crossentropy",
        }
        metrics = {
            "age_prediction": mae,
            "gender_prediction": "acc",
        }
    else:
        gender_classes = 2
        age_classes = 8
        losses = {
            "age_prediction": 'categorical_crossentropy',
            "gender_prediction": "categorical_crossentropy",
        }
        metrics = {
            "age_prediction": 'acc',
            "gender_prediction": "acc",
        }

    model = None
    if MODEL == 'vggFace':
        model = Net(MODEL,1,2,8,gender_classes,age_classes)
        model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        model = Net(MODEL,1,2,8,gender_classes,age_classes)
        MODEL = model.name

    print('[K-FOLD] Started...')
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_split = kf.split(age_label)
    for train_idx, test_idx in kf_split:

        train_paths = paths[train_idx]
        train_age = age_label[train_idx]
        train_gender = gender_label[train_idx]

        test_paths = paths[test_idx]
        test_age = age_label[test_idx]
        test_gender = gender_label[test_idx]

        print(len(train_paths),len(test_paths))

        

        if MODEL == 'ssrnet':
            losses = {
                "age_prediction": "mae",
                "gender_prediction": "mae",
            }
            metrics = {
                "age_prediction": "mae",
                "gender_prediction": "binary_accuracy",
            }

        model.summary()
        weights_path = './train_weights/gender_age/{}/{}/'.format(DATASET,MODEL)
        logs_path = './train_log/gender_age/{}/{}/'.format(DATASET,MODEL)
        
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        model_names = weights_path + '.{epoch:02d}-{val_gender_prediction_acc: 0.4f} - {val_age_prediction_acc: 0.4f}.hdf5'
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
        
        model.compile(optimizer='adam', loss=losses,loss_weights=[0.1,1], metrics=metrics)
        model.fit_generator(
            DataGenerator(model, train_paths, train_age, train_gender, BATCH_SIZE),
            validation_data=DataGenerator(model, test_paths, test_age, test_gender, BATCH_SIZE),
            epochs=EPOCH,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )
        n_fold += 1
        del  train_paths, train_age, train_gender
        del  test_paths, test_age, test_gender


if __name__ == '__main__':
    main()
