import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
# from model.inceptionv3 import EmotionNetInceptionV3
# from model.xception import EmotionNetXception
# from model.mobilenetv2 import EmotionNetMobileNetV2
# from model.vggface import EmotionNetVGGFacenet
# from model.mini_xception import EmotionNetmin_XCEPTION
from model.models import Net
import ast
from utils.datasets import DataManager 
from utils.STL.emotion_generator import DataGenerator
from utils.callback import DecayLearningRate
from keras.utils.vis_utils import plot_model
from model.models import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    choices=['fer2013','ferplus','sfew','kdef','expw'],
                    default='expw',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception','xception'],
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
    file_path = 'data/db/{}_cleaned.csv'.format(dataset) 
    data1 = pd.read_csv(file_path)
    data = data1
    del data1
    paths = data['full_path'].values
    emotion_label = data['emotion'].values.astype('uint8')
    return paths, emotion_label

def mae(y_true, y_pred):
    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_true, axis=1)), axis=-1)


#'vgg16',     [:19]net layer,      'pool5'
#'resnet50' ,     [:174]net layer,    'avg_pool'
#'senet50' ,    [:286] net layer,        'avg_pool'
def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:19]:
        layer.trainable = False
    for layer in model.layers[19:]:
        layer.trainable = True
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
    MODEL = args.model
    EPOCH = args.epoch
    PATIENCE = args.patience
    BATCH_SIZE = args.batch_size
    NUM_WORKER = args.num_worker
    DATASET = args.dataset

    if DATASET == 'ferplus':
        emotion_classes = 8
    else:
        emotion_classes = 7


    paths, emotion_label = load_data(DATASET)
    n_fold = 1
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split = kf.split(emotion_label)
    for train_idx, test_idx in kf_split:

        model = None
        if MODEL == 'vggFace':
            model = Net(MODEL,1,0,emotion_classes,2,2)
            if args.is_freezing:
                model = freeze_all_but_mid_and_top(model)
        else:
            model = Net(MODEL,1,0,emotion_classes,2,2)



        train_paths = paths[train_idx]
        train_emotion = emotion_label[train_idx]

        test_paths = paths[test_idx]
        test_emotion = emotion_label[test_idx]

        print(len(train_paths),len(test_paths))


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
        weights_path = './train_weights/emotion/{}/{}/'.format(DATASET,model.name)
        logs_path = './train_log/emotion/{}/{}/'.format(DATASET,model.name)
        # plot_model(model,to_file= logs_path+'.jpg',show_shapes=True)
        
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        model_names = weights_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
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
                ModelCheckpoint(MODEL, 'val_loss', verbose=1,save_best_only=True),
                CSVLogger('train_log/{}-{}.log'.format(MODEL, n_fold)),
                DecayLearningRate([30, 60])]
        
        model.compile(optimizer='adam', loss=losses, metrics=metrics)
        model.fit_generator(
            DataGenerator(model, train_paths, train_emotion, BATCH_SIZE),
            validation_data=DataGenerator(model,  test_paths,  test_emotion, BATCH_SIZE),
            epochs=args.no_freezing_epoch,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )
        n_fold += 1
        del  train_paths, train_emotion
        del  test_paths, test_emotion


if __name__ == '__main__':
    main()
