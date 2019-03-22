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
from model.vggface import AgeNetVGGFacenet
from model.mini_xception import AgenderNetmin_XCEPTION, AgeNetmin_XCEPTION
from utils.STL.attributes_generator import DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='celeba',
                    help='Model to be used')
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='vggFace',
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


def load_data(dataset):
    file_path = 'data/db/{}_cleaned.csv'.format(dataset) 
    data1 = pd.read_csv(file_path)
    data = data1
    del data1
    paths = data['full_path'].values
    attr_label = []
    for i in range(40):
        attr_label.append(data['{}'.format(i)].values.astype('uint8'))
    attr_labels = [attr_label[i] for i in range(40)]
    attr_labels = np.transpose(attr_labels)
    return paths, attr_labels

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

    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 71), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 71), dtype='float32') * y_true, axis=1)), axis=-1)


#'vgg16',     [:19]net layer,      'pool5'
#'resnet50' ,     [:174]net layer,    'avg_pool'
#'senet50' ,    [:286] net layer,        'avg_pool'
def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:19]:
        layer.trainable = False
    for layer in model.layers[19:]:
        layer.trainable = True
    return model


# class Average_ACC(keras.callbacks.Callback):
#     def __init__(self,validation_data,interval=1):
#         self.interval=interval
#         self.x_val,self.y_val=validation_data
#     def on_epoch_end(self,epoch, logs={}):
#         if epoch % self.interval == 0:
#             self.
            

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


    loss_dict = {}
    metrics_dict = {}
    attr_predcition=[]
    loss =[]
    acc =[]
    for i in range(40):
        attr_predcition.append('attr{}_predition'.format(i))
        loss.append('binary_crossentropy')
        acc.append('acc') 
    # print(attr_predcition,loss,acc)  
    loss_dict = dict(zip(attr_predcition, loss))
    metrics_dict = dict(zip(attr_predcition, acc))




    

    losses = {
        "age_prediction": mae
    }
    metrics = {
        "age_prediction": mae
    }
   


    model = None
    if MODEL == 'vggFace':
        model = Net(MODEL,1,6,8,2,8)
        model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        model = Net(MODEL,1,6,8,2,8)
        MODEL = model.name

    paths, attr_label = load_data(DATASET)
    # model.summary()
    n_fold = 1
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_split = kf.split(attr_label)
    for train_idx, test_idx in kf_split:
        print('length:train test:',len(train_idx),len(test_idx))
        train_paths = paths[train_idx]
        train_attr = attr_label[train_idx]


        test_paths = paths[test_idx]
        test_attr = attr_label[test_idx]

        if MODEL == 'ssrnet':
            losses = {
                "age_prediction": "mae",
                "gender_prediction": "mae",
            }
            metrics = {
                "age_prediction": "mae",
                "gender_prediction": "binary_accuracy",
            }
        weights_path = './train_weights/attr/{}/{}/'.format(DATASET,MODEL)
        logs_path = './train_log/attr/{}/{}/'.format(DATASET,MODEL)
        
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        # model_names = weights_path + '{epoch:02d}-{val_acc:.2f}.hdf5'
        model_names = weights_path + '{epoch:02d}.hdf5'
        csv_name = logs_path + '{}.log'.format(n_fold)
        board_name = logs_path + '{}'.format(n_fold)

        checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=True)
        csvlogger=CSVLogger(csv_name)
        early_stop = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(PATIENCE/2), verbose=1)
        tensorboard = TensorBoard(log_dir=board_name,batch_size=BATCH_SIZE)
        callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]

        
        model.compile(optimizer='adam', loss=loss_dict, metrics=metrics_dict)
        model.fit_generator( 
            DataGenerator(model, train_paths, train_attr, BATCH_SIZE),
            validation_data=DataGenerator(model, test_paths, test_attr, BATCH_SIZE),
            epochs=EPOCH,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=False,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )
        n_fold += 1
        del  train_paths, train_attr
        del  test_paths, test_attr


if __name__ == '__main__':
    main()
