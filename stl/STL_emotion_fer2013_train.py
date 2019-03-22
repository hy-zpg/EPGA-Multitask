import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
from model.inceptionv3 import EmotionNetInceptionV3
from model.mobilenetv2 import EmotionNetMobileNetV2
from model.vggface import EmotionNetVGGFacenet
from model.mini_xception import EmotionNetmin_XCEPTION
from utils.datasets import DataManager 
from utils.fer2013_generator import DataGenerator
from utils.callback import DecayLearningRate

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mobilenetv2',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=50,
                    type=int,
                    help='Num of training epoch')
parser.add_argument('--batch_size',
                    default=128,
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


def load_data():
    sfew = pd.read_csv('data/db/FER2013_cleaned.csv')
    data = sfew
    del sfew
    paths = data['fullpath'].values
    emotion_label = data['emotion'].values.astype('uint8')
    return paths, emotion_label

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

    paths, emotion_label = load_data()
    n_fold = 1
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split = kf.split(emotion_label)
    for train_idx, test_idx in kf_split:
        model = None
        if MODEL == 'ssrnet':
            model = EmotionNetSSRNet(64, [3, 3, 3], 1.0, 1.0)
        elif MODEL == 'inceptionv3':
            model = EmotionNetInceptionV3()
        elif MODEL == 'mini_xcpetion':
            model = EmotionNetmin_XCEPTION()
        elif MODEL == 'vggFace':
            model = EmotionNetVGGFacenet()
        else:
            model = EmotionNetMobileNetV2()
        train_paths = paths[train_idx]
        train_emotion = emotion_label[train_idx]

        test_paths = paths[test_idx]
        test_emotion = emotion_label[test_idx]


        losses = {
            "emotion_prediction": "categorical_crossentropy",
        }
        metrics = {
            "emotion_prediction": "acc",
        }
        if MODEL == 'ssrnet':
            losses = {
                "emotion_prediction": "mae",
            }
            metrics = {
                "emotion_prediction": "mae",
            }

# -{epoch:02d}-{val_loss:.4f}-{val_gender_prediction_binary_accuracy:.4f}-{val_age_prediction_mean_absolute_error:.4f}
        model.summary()
        model_names = './train_weight/fer2013_emotion/'+ MODEL + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only=True,save_best_only=True)
        csvlogger=CSVLogger('train_log/fer2013_emotion/{}-{}.log'.format(MODEL,n_fold))
        early_stop = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard = TensorBoard(log_dir='train_log/fer2013_emotion/{}-{}'.format(MODEL,n_fold),batch_size=BATCH_SIZE)
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
        
        model.compile(optimizer='adam', loss=losses, metrics=metrics)
        model.fit_generator(
            DataGenerator(model, train_paths, train_emotion, BATCH_SIZE),
            validation_data=DataGenerator(model,  test_paths,  test_emotion, BATCH_SIZE),
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
