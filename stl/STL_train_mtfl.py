import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from model.mobilenetv2 import MTFLMobileNetV2
from utils.STL.mtfl_generator import DataGenerator
from utils.callback import DecayLearningRate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    choices=['imdb','adience','fgnet','megaage_asian'],
                    default='mtfl',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mobilenetv2',
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
                    default=100,
                    type=int,
                    help='Number of traing epoch')


# def load_data():
#     imdb = pd.read_csv('data/db/mtfl_cleaned.csv')
#     data = imdb
#     del imdb
#     db = data['db_name'].values
#     paths = data['full_path'].values
#     smile_label = data['smile'].values.astype('uint8')
#     gender_label = data['gender'].values.astype('uint8')
#     landmark_label = data['landmark'].values
#     return db, paths, landmark_label, gender_label, smile_label


def landmark_deal(list):
    length=len(list)
    landmark = np.zeros([10,1])
    for i in range(length):
        landmark[i] = list[i]
    return landmark

def load_data(dataset):
    file_path = 'data/db/{}_cleaned.csv'.format(dataset) 
    data1 = pd.read_csv(file_path)
    data = data1

    db = data['db_name'].values
    paths = data['full_path'].values
    smile_label = data['smile'].values.astype('uint8')
    gender_label = data['gender'].values.astype('uint8')
    landmark_label = data['landmark'].values
    pose_label = data['pose'].values
    return db, paths, landmark_label, gender_label, smile_label,pose_label



#     return K.mean(K.abs(K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_pred, axis=1) -
#                         K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_true, axis=1)), axis=-1)


def custom_mse_lm(y_true,y_pred):
    return K.sign(K.sum(K.abs(y_true),axis=-1))*K.sum(K.square(tf.multiply((K.sign(y_true)+1)*0.5, y_true-y_pred)),axis=-1)/K.sum((K.sign(y_true)+1)*0.5,axis=-1)

#'vgg16',     [:19]net layer,      'pool5'
#'resnet50' ,     [:174]net layer,    'avg_pool'
#'senet50' ,    [:286] net layer,        'avg_pool'
def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:147]:
        layer.trainable = False
    for layer in model.layers[147:]:
        layer.trainable = True
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

    db, paths, landmark_label,gender_label, smile_label,pose_label = load_data(args.dataset)
    n_fold = 1
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_split = kf.split(gender_label)
    for train_idx, test_idx in kf_split:
        model = None
        if MODEL == 'ssrnet':
            model = AgenderSSRNet(64, [3, 3, 3], 1.0, 1.0)
        elif MODEL == 'inceptionv3':
            model = AgenderNetInceptionV3()
        elif MODEL == 'mini_xcpetion':
            model = AgenderNetmin_XCEPTION()
        elif MODEL == 'vggFace':
            model = AgenderNetVGGFacenet()
            # model = freeze_all_but_mid_and_top(model)
            MODEL = model.name
        else:
            model = MTFLMobileNetV2()
            MODEL = model.name
        train_db = db[train_idx]
        train_paths = paths[train_idx]
        train_landmark = landmark_label[train_idx]
        train_gender = gender_label[train_idx]
        train_smile = smile_label[train_idx]
        train_pose = pose_label[train_idx]

        test_db = db[test_idx]
        test_paths = paths[test_idx]
        test_landmark = landmark_label[test_idx]
        test_gender = gender_label[test_idx]
        test_smile = smile_label[test_idx]
        test_pose = pose_label[test_idx]

        losses = {
            # 'landmark_prediction':custom_mse_lm,
            "gender_prediction": "categorical_crossentropy",
            "smile_prediction": "categorical_crossentropy",
            'pose_prediction':'categorical_crossentropy'
        }
        metrics = {
            # "landmark_prediction":"mse",
            "gender_prediction": 'acc',
            "smile_prediction": "acc",
            'pose_prediction':'acc'
        }
        if MODEL == 'ssrnet':
            losses = {
                "age_prediction": "mae",
                "gender_prediction": "mae",
            }
            metrics = {
                "age_prediction": "mae",
                "gender_prediction": "binary_accuracy",
            }

# -{epoch:02d}-{val_loss:.4f}-{val_gender_prediction_binary_accuracy:.4f}-{val_age_prediction_mean_absolute_error:.4f}
        # model.summary()
        weights_path = './train_weight/mtfl/'+ MODEL
        logs_path = 'train_log/mtfl/'

        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        model_names = weights_path + '.{epoch:02d}-{val_pose_prediction_acc: 0.4f} - {val_gender_prediction_acc: 0.4f} - {val_smile_prediction_acc: 0.4f}- {val_pose_prediction_acc: 0.4f}.hdf5'
        csv_name = logs_path + '{}-{}.log'.format(MODEL,n_fold)
        board_name = logs_path + '{}'.format(n_fold)

        checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only=True,save_best_only=True)
        csvlogger=CSVLogger(csv_name)
        early_stop = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard = TensorBoard(board_name,batch_size=BATCH_SIZE)
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
        
        model.compile(optimizer='adam', loss=losses, loss_weights=[5,2,2],metrics=metrics)
        model.fit_generator(
            DataGenerator(model, train_db, train_paths, train_landmark,train_gender, train_smile,train_pose, BATCH_SIZE),
            validation_data=DataGenerator(model, test_db, test_paths, test_landmark, test_gender,test_smile,test_pose, BATCH_SIZE),
            epochs=EPOCH,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )
        n_fold += 1
        del train_db, train_paths, train_landmark, train_gender,train_smile
        del test_db, test_paths, test_landmark, test_gender,test_smile


if __name__ == '__main__':
    main()
