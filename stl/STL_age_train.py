import argparse
import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
# from model.inceptionv3 import AgenderNetInceptionV3
# from model.mobilenetv2 import AgenderNetMobileNetV2
# from model.ssrnet import AgenderSSRNet
# from model.vggface import AgeNetVGGFacenet
# from model.mini_xception import AgenderNetmin_XCEPTION, AgeNetmin_XCEPTION
from utils.STL.age_generator import DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    choices=['imdb','adience','fgnet','megaage_asian'],
                    default='adience',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=1,
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
parser.add_argument('--no_freezing_epoch',
                    default=5,
                    type=int,
                    help='starting no freezing')


def load_data(dataset):
    file_path = 'data/db/{}.csv'.format(dataset) 
    data1 = pd.read_csv(file_path)
    data = data1
    del data1
    paths = data['full_path'].values
    age_label = data['age'].values.astype('uint8')
    # gender_label = data['gender'].values.astype('uint8')
    return paths, age_label

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

    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 70), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 70), dtype='float32') * y_true, axis=1)), axis=-1)


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

def updateTargetmodel(model,target_model):
    modelweights = model.trainable_weights
    targetmodelweights = target_model.trainable_weights
    for i in range(len(targetmodelweights)):
        targetmodelweights[i].assign(modelweights[i])
    return target_model

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


    if DATASET == 'fgnet':
        age_classes = 70
        losses = {
            "age_prediction": mae
        }
        metrics = {
            "age_prediction": mae
        }
    elif DATASET == 'megaage_asian':
        age_classes = 71
        losses = {
            "age_prediction": mae
        }
        metrics = {
            "age_prediction": mae
        }
    else:
        gender_classes = 2
        age_classes = 8
        losses = {
            "age_prediction": 'categorical_crossentropy'        }
        metrics = {
            "age_prediction": 'acc'        }


    paths, age_label = load_data(DATASET)
    n_fold = 1
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_split = kf.split(age_label)

    for train_idx, test_idx in kf_split:
        model = None
        if MODEL == 'vggFace':
            model = Net(MODEL,1,1,8,2,age_classes)
            if args.is_freezing:
                model = freeze_all_but_mid_and_top(model)
        else:
            model = Net(MODEL,1,1,8,2,age_classes)
        
        train_paths = paths[train_idx]
        train_age = age_label[train_idx]

        test_paths = paths[test_idx]
        test_age = age_label[test_idx]

        if MODEL == 'ssrnet':
            losses = {
                "age_prediction": "mae",
                "gender_prediction": "mae",
            }
            metrics = {
                "age_prediction": "mae",
                "gender_prediction": "binary_accuracy",
            }
        weights_path = './train_weights/age/{}/{}/'.format(DATASET,model.name)
        logs_path = './train_log/age/{}/{}/'.format(DATASET,model.name)
        
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        model_names = weights_path + '{}__'.format(n_fold) + '{epoch:02d}-{val_acc:.2f}.hdf5'
        csv_name = logs_path + '{}.log'.format(n_fold)
        board_name = logs_path + '{}'.format(n_fold)

        checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=True)
        early_stop = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard = TensorBoard(log_dir=board_name,batch_size=BATCH_SIZE)
        callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]
        model.compile(optimizer='adam', loss=losses, metrics=metrics)

        # emotion_model = Net(MODEL,1,1,8,2,age_classes)
        # emotion_model = updateTargetmodel(model,emotion_model)
        # emotion_model = keras.models.clone_model(model)
        # emotion_model.set_weights(model.get_weights())
        # emotion_model = freeze_all(emotion_model)
        # emotion_model.compile(optimizer='adam', loss=losses, metrics=metrics)
        print('freezing model')
        model.summary()
        model.fit_generator(
            DataGenerator(model, train_paths, train_age, age_classes, BATCH_SIZE),
            validation_data=DataGenerator(model, test_paths, test_age, age_classes,BATCH_SIZE),
            epochs=args.no_freezing_epoch,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )

        model = no_freeze_all(model)
        print('no freezing')
        model.summary()
        model.fit_generator(
            DataGenerator(model, train_paths, train_age, age_classes, BATCH_SIZE),
            validation_data=DataGenerator(model, test_paths, test_age, age_classes,BATCH_SIZE),
            epochs=EPOCH-args.no_freezing_epoch,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )

        n_fold += 1
        print(np.array_equal(model.get_weights()[0][0][0][0],emotion_model.get_weights()[0][0][0][0]))
        if np.array_equal(model.get_weights()[0],emotion_model.get_weights()[0]):
                print('please check code!')
        del  train_paths, train_age
        del  test_paths, test_age


if __name__ == '__main__':
    main()
