import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def STL_analysis(file_path,store_path):
    data1 = pd.read_csv(file_path)
    data = data1
    del data1
    epoch = data['epoch'].values
    train_loss=data['loss'].values.astype('float64')
    valid_loss=data['val_loss'].values.astype('float64')
    train_acc=data['acc'].values.astype('float64')
    valid_acc=data['val_acc'].values.astype('float64')

    fig=plt.figure(num=2)
    fig.suptitle('training result')
    ax1=fig.add_subplot(2,1,1)
    # ax1.set_title('loss',fontsize=12)
    ax1.plot(np.arange(0,len(epoch),1),train_loss,color='g',label='train_loss')
    ax1.plot(np.arange(0,len(epoch),1),valid_loss,color='r',label='valid_loss')
    ax1.legend()
    ax2=fig.add_subplot(2,1,2)
    # ax2.set_title('accuracy',fontsize=12)
    ax2.plot(np.arange(0,len(epoch),1),train_acc,color='g',label='train_acc')
    ax2.plot(np.arange(0,len(epoch),1),valid_acc,color='r',label='valid_acc')
    ax2.legend()
    plt.savefig(store_path+'stl_result_figure',dpi=2048)
    plt.close(1)
    plt.close(2)

def MTL_analysis(file_path,store_path):
    data1 = pd.read_csv(file_path)
    data = data1
    del data1
    epoch = data['epoch'].values
    train_loss=data['loss'].values.astype('float64')
    train_emotion_loss=data['emotion_prediction_loss'].values.astype('float64')
    train_pose_loss=data['pose_prediction_loss'].values.astype('float64')
    train_age_loss=data['age_prediction_loss'].values.astype('float64')
    
    valid_loss=data['val_loss'].values.astype('float64')
    valid_emotion_loss=data['val_emotion_prediction_loss'].values.astype('float64')
    valid_pose_loss=data['val_pose_prediction_loss'].values.astype('float64')
    valid_age_loss=data['val_age_prediction_loss'].values.astype('float64')

    
    train_emotion_acc=data['emotion_prediction_my_acc'].values.astype('float64')
    train_pose_acc=data['pose_prediction_my_acc'].values.astype('float64')
    train_age_acc=data['age_prediction_my_acc'].values.astype('float64')
    valid_emotion_acc=data['val_emotion_prediction_my_acc'].values.astype('float64')
    valid_pose_acc=data['val_pose_prediction_my_acc'].values.astype('float64')
    valid_age_acc=data['val_age_prediction_my_acc'].values.astype('float64')


    fig=plt.figure(num=3)
    fig.suptitle('training result')
    ax0=fig.add_subplot(3,1,1)
    ax0.plot(np.arange(0,len(epoch),1),train_loss,color='g',linestyle='-',label='total_train_loss')
    ax0.plot(np.arange(0,len(epoch),1),valid_loss,color='g',linestyle='-',label='total_valid_loss')
    ax0.legend(fontsize=2)
    ax1=fig.add_subplot(3,1,2)
    ax1.plot(np.arange(0,len(epoch),1),train_emotion_loss,color='r',linestyle='-.',label='emotion_train_loss')
    ax1.plot(np.arange(0,len(epoch),1),valid_emotion_loss,color='r',linestyle='-.',label='emotion_valid_loss')
    ax1.plot(np.arange(0,len(epoch),1),train_pose_loss,color='b',linestyle='--',label='pose_train_loss')
    ax1.plot(np.arange(0,len(epoch),1),valid_pose_loss,color='b',linestyle='--',label='pose_valid_loss')
    ax1.plot(np.arange(0,len(epoch),1),train_age_loss,color='y',linestyle=':',label='age_train_loss')
    ax1.plot(np.arange(0,len(epoch),1),valid_age_loss,color='y',linestyle=':',label='age_valid_loss')
    ax1.legend(fontsize=2)

    ax2=fig.add_subplot(3,1,3)
    # ax2.plot(np.arange(0,len(epoch),1),train_acc,color='g',linestyle='-',label='total_train_acc')
    # ax2.plot(np.arange(0,len(epoch),1),valid_acc,color='g',linestyle='-',label='total_valid_acc')
    ax2.plot(np.arange(0,len(epoch),1),train_emotion_acc,color='r',linestyle='-.',label='emotion_train_acc')
    ax2.plot(np.arange(0,len(epoch),1),valid_emotion_acc,color='r',linestyle='-.',label='emotion_valid_acc')
    ax2.plot(np.arange(0,len(epoch),1),train_pose_acc,color='b',linestyle='--',label='pose_train_acc')
    ax2.plot(np.arange(0,len(epoch),1),valid_pose_acc,color='b',linestyle='--',label='pose_valid_acc')
    ax2.plot(np.arange(0,len(epoch),1),train_age_acc,color='y',linestyle=':',label='age_train_acc')
    ax2.plot(np.arange(0,len(epoch),1),valid_age_acc,color='y',linestyle=':',label='age_valid_acc')
    ax2.legend(fontsize=2)
    plt.savefig(store_path+'mtl_result_figure',dpi=2048)
    plt.close(1)
    plt.close(2)


stl_file_path='/home/yanhong/Downloads/next_step/HY_MTL/train_log/AgeNetEmotionNetminixception/adience/1.log'
mtl_file_path='/home/yanhong/Downloads/next_step/HY_MTL/train_log/CONFUSION/expw-aflw-adience/EPA_VGGFace_vgg16/based_STL_true-freezing_true-pesudo_false-pesudo_selection_false-threshold_0.9.log'
store_path='/home/yanhong/Downloads/next_step/HY_MTL/result_analysis/'
STL_analysis(stl_file_path,store_path)
MTL_analysis(mtl_file_path,store_path)