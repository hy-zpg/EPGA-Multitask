# import numpy as np
import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.optimize
import cv2
from keras.models import Model
from keras.utils import Sequence, np_utils
from utils.STL.emotion_generator import DataGenerator as emotion_DataGenerator
from utils.STL.pose_generator import DataGenerator as pose_DataGenerator
from utils.pseudo_density_distribution.pseudo_density import CurriculumClustering
from utils.pseudo_density_distribution.pseudo_distribution import weights_gmm_dataset_distribution
from utils.pseudo_density_distribution.pseudo_distribution import weights_density_dataset_distribution
from utils.pseudo_density_distribution.density_gmm_distribution import Density_gmm_distribution_weights
# from utils.pseudo_gmm_distribution import Data_distribution
import shutil
import os


def load_image(paths: np.ndarray, size: int, input_size,is_augmentation:bool):
    images = [cv2.imread('{}'.format(img_path)) for img_path in paths]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    if input_size[3] ==1:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
        images = np.expand_dims(images, -1)
        return np.array(images, dtype='uint8')
    
    else:
        if is_augmentation:
            images = [image_augmentation.img_to_array(image) for image in images]
            images = [image_augmentation.random_rotation(image,rg=10) for image in images]
            images = [image_augmentation.random_shift(image,wrg=0.1, hrg=0.1) for image in images]
            images = [image_augmentation.random_zoom(image,zoom_range=[0.1,0.3]) for image in images]
            images = [image_augmentation.flip_axis(image, axis=0) for image in images]
        return np.array(images, dtype='uint8')

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x




## assign weights to different samples
'''
@method1: combining hard label(pseudo_label) and soft label(distill_label),selection
@method2: combining hard label(pseudo_label) and soft label(distill_label), interpolation
1. confidence score && distilled knowledge
2. confidence score & density(purity)  && distilled knowledge
3. confidence score & density(purity) & distribution(distance)  && distilled knowledge
'''


'''
pseudo_label_possibilty: label weights
pseudo_label_index: label category
distilled_knowledge: distilled knowledge matrix
'''
def Weight_Strategy(model,is_hard_weight,is_emotion,interpolation_weight,pseudo_label_possibilty,pseudo_label_index,distilled_knowledge,selection_threshold):
    # pseudo_label_possibilty = np.array(pseudo_label_possibilty)
    # pseudo_label_possibilty = pseudo_label_possibilty/np.max(pseudo_label_possibilty)
    # pseudo_label_possibilty = np.nan_to_num(pseudo_label_possibilty)
    # nan_index = np.isnan(pseudo_label_possibilty)
    # inf_index = np.isinf(pseudo_label_possibilty)
    # print(np.shape(nan_index))
    # print(np.shape(inf_index))
    # print(nan_index[:10])
    # print(inf_index[:10])
    # if np.shape(nan_index)[0]!= 0:
    #     pseudo_label_possibilty[nan_index]=0
    # if np.shape(inf_index)[0]!=0:
    #     pseudo_label_possibilty[inf_index]=0
    # pseudo_label_possibilty[np.isinf(pseudo_label_possibilty)] = 0
    # pseudo_label_possibilty[np.isnan(pseudo_label_possibilty)] = 0
    ##density -1
    # pseudo_label_possibilty = np.where(pseudo_label_possibilty > 0.00005, pseudo_label_possibilty, 0)
    
    print('before normalization weights:',pseudo_label_possibilty[:10])
    pseudo_label_possibilty = pseudo_label_possibilty/np.max(pseudo_label_possibilty)
    sorted_result = np.array(sorted(pseudo_label_possibilty))
    threshold_index = int(selection_threshold*np.shape(pseudo_label_possibilty)[0])
    selection_threshold = sorted_result[threshold_index]
    ##normalization
    
    # print('after normalization weights:',pseudo_label_possibilty[:10])
    if is_emotion:
        classes = model.emotion_classes
    else:
        classes = model.pose_classes
    if is_hard_weight:
        # pseudo_label_possibilty = pseudo_label_possibilty/np.max(pseudo_label_possibilty)
        # pseudo_label_possibilty = np.where(pseudo_label_possibilty > selection_threshold, pseudo_label_possibilty, 0)
        # pseudo = np_utils.to_categorical(pseudo_label_index, classes)
        # pseudo_label_possibilty = pseudo_label_possibilty/np.max(pseudo_label_possibilty)
        # print('final selected weights:',pseudo_label_possibilty[:10])
        # print('pseudo ratio:',np.shape(np.nonzero(pseudo_label_possibilty))[1]/np.shape(pseudo_label_possibilty)[0])
        # pseudo = pseudo * [np.full(classes,value) for value in pseudo_label_possibilty]
        # for i in range(len(pseudo_label_possibilty)):
        #     if pseudo_label_possibilty[i]==0:
        #         pseudo[i] = distilled_knowledge[i]
        print('final weights:',pseudo_label_possibilty[:10])
        pseudo_label_possibilty = np.where(pseudo_label_possibilty > selection_threshold, pseudo_label_possibilty, 0)
        pseudo = np_utils.to_categorical(pseudo_label_index, classes)
        pseudo = pseudo * [np.full(classes,value) for value in pseudo_label_possibilty]
        for i in range(len(pseudo_label_possibilty)):
            if pseudo_label_possibilty[i]==0:
                # pseudo[i] = distilled_knowledge[i]
                pseudo[i] = np.zeros([classes])
        return pseudo
    else:
        print('final weights:',pseudo_label_possibilty[:10])
        pseudo_1 = np_utils.to_categorical(pseudo_label_index, classes)
        pseudo_1 = pseudo_1*[np.full(classes,value) for value in pseudo_label_possibilty]
        pseudo_weights = pseudo_label_possibilty
        distill_weights = np.ones(len(pseudo_label_possibilty)) - pseudo_weights
        distilled_knowledge = distilled_knowledge*[np.full(classes,value) for value in distill_weights]
        pseudo = pseudo_1 + distilled_knowledge


        


        return pseudo

'''
emotion_index: pseudo data category
pose_index: pseudo data category
'''
def Density_Distribution(model,pre_model,is_emotion,is_density,is_distribution,cluster_k,density_t,emotion_path,emotion_label,emotion_index,pose_path,pose_label,pose_index):
    # emotion pseudo weight
    if is_emotion:
        print('feature extracting')
        emotion_feature_model = Model(inputs=pre_model.inputs,outputs=pre_model.get_layer('emotion_fc').output)
        # emotion_feature_model.set_weights(pre_model.get_weights())
        emotion_pseudo_feature = emotion_feature_model.predict_generator(pose_DataGenerator(model,pose_path,pose_label,32))
        emotion_feature = emotion_feature_model.predict_generator(emotion_DataGenerator(model,emotion_path,emotion_label,32))
        print('density-distribution calculation')
        for i in range(model.emotion_classes):
            print('{}-categoty-num-{}'.format(i,np.shape(np.where(emotion_index==i))))
        cc = Density_gmm_distribution_weights(cluster_k,density_t,is_density,is_distribution)
        cc.fit(emotion_feature,emotion_label,emotion_pseudo_feature, emotion_index)
        density_distribution_weights = cc.output_labels
        return density_distribution_weights
    # pose pseudo weight
    else:
        print('feature extracting')
        pose_feature_model = Model(inputs=pre_model.inputs,outputs=pre_model.get_layer('pose_fc').output) 
        # pose_feature_model.set_weights(pre_model.get_weights())
        pose_pseudo_feature = pose_feature_model.predict_generator(emotion_DataGenerator(model,emotion_path,emotion_label,32))
        pose_feature = pose_feature_model.predict_generator(pose_DataGenerator(model,pose_path,pose_label,32))
        print('density-distribution calculation')
        for i in range(model.pose_classes):
            print('{}-categoty-num-{}'.format(i,np.shape(np.where(pose_index==i))))
        cc = Density_gmm_distribution_weights(cluster_k,density_t,is_density,is_distribution)
        cc.fit(pose_feature,pose_label,pose_pseudo_feature, pose_index)
        density_distribution_weights = cc.output_labels
        return density_distribution_weights
 




def assign_weights(model,pre_model,is_emotion,emotion_path,emotion_label,pose_path,pose_label,k,is_augmentation,selection_threshold,is_distill,is_confidence,is_density,is_distribution,is_hard_weight,interpolation_weight,distill_t,density_t):
    print('starting assigning weights')
    vggface=True
    confidence = (is_confidence and not is_density and not is_distribution)
    density = (is_density and not is_confidence and not is_distribution)
    distribution = (is_distribution and not is_confidence and not is_density)
    confidence_density = (is_confidence and is_density and not is_distribution)
    confidence_distribution = (is_confidence and is_distribution and not is_density)
    density_distribution = (is_density and is_distribution and not is_confidence)
    confidence_density_distribution = (is_density and is_distribution and is_confidence)
    pure = (not is_confidence and not is_density and not is_distribution)
    if is_distill:
        print('distill method')   
    else: 
        if pure:
            print('pseudo generation with pure')
        if confidence:
            print('pseudo generation with confidence')
        if density:
            print('pseudo generation with density')
        if distribution:
            print('pseudo generation with distribution')
        if confidence_density:
            print('pseudo generatio with confidence_density')
        if confidence_distribution:
            print('pseudo generation with confidence_distribution')
        if density_distribution:
            print('pseudo generation with density_distribution')
        if confidence_density_distribution:
            print('pseudo generation with confidence_density_distribution')


    
    # generating emotion pseudo
    if not is_emotion:
        # adding temperature to the distilled knowledge
        emotion_distill_model = Model(inputs=pre_model.inputs,outputs=pre_model.get_layer('emotion').output)
        # emotion_distill_model.set_weights(pre_model.get_weights())
        emotion_distill = emotion_distill_model.predict_generator(pose_DataGenerator(model,pose_path,pose_label,32))
        # distilled knowledge, generating array
        emotion_distill = softmax(emotion_distill/distill_t)
        # pseudo confidence from model prediction, generating vector
        emotion_confidence_score = pre_model.predict_generator(pose_DataGenerator(model,pose_path,pose_label,32))[0]
        emotion_index=np.argmax(emotion_confidence_score, axis=1)
        emotion_possibility = [emotion_confidence_score[i][emotion_index[i]] for i in range(np.shape(emotion_index)[0])]
        print('confidence:',emotion_possibility[:10])
        # weights combination strategy
        is_emotion_classes = True
        if is_distill:
            return emotion_distill
        if pure:
            emotion_pseudo_1 = np_utils.to_categorical(emotion_index, model.emotion_classes)
            emotion_pseudo = 0.5*emotion_pseudo_1 + 0.5*emotion_distill
        elif confidence:
            emotion_pseudo = Weight_Strategy(model,is_hard_weight,is_emotion_classes,interpolation_weight,emotion_possibility,emotion_index,emotion_distill,selection_threshold)
        elif density or distribution or density_distribution:
            density_distribution_weights = Density_Distribution(model,pre_model,is_emotion_classes,is_density,is_distribution,k,density_t,emotion_path,emotion_label,emotion_index,pose_path,pose_label,None)
            emotion_pseudo = Weight_Strategy(model,is_hard_weight,is_emotion_classes,interpolation_weight,density_distribution_weights,emotion_index,emotion_distill,selection_threshold)
        elif confidence_density or confidence_distribution or confidence_density_distribution:
            density_distribution_weights = Density_Distribution(model,pre_model,is_emotion_classes,is_density,is_distribution,k,density_t,emotion_path,emotion_label,emotion_index,pose_path,pose_label,None)
            combined_weights = emotion_possibility * density_distribution_weights

            # print('select samples to visualze:')
            # x = combined_weights/np.max(combined_weights)
            # good_samples = np.where(x>0.9)
            # stored_path = './selected_images/'
            # # os.mkdir(stored_path)
            # print(pose_path[good_samples])
            # for i in range(np.shape(good_samples)[1]):
            #     shutil.copyfile(pose_path[good_samples][0][i],stored_path+'{}_pose_good.jpg'.format(emotion_index[good_samples[0][i]]))
            # bad_samples = np.where(x<0.001)
            # print(pose_path[bad_samples])
            # for i in range(np.shape(bad_samples)[1]):
            #     shutil.copyfile(pose_path[bad_samples][0][i],stored_path+'{}_pose_bad.jpg'.format(emotion_index[bad_samples[0][i]]))
 


            emotion_pseudo = Weight_Strategy(model,is_hard_weight,is_emotion_classes,interpolation_weight,combined_weights,emotion_index,emotion_distill,selection_threshold)
        return emotion_pseudo

    else:
        # adding temperature to the distilled knowledge
        pose_distill_model = Model(inputs=pre_model.inputs,outputs=pre_model.get_layer('pose').output)
        # pose_distill_model.set_weights(pre_model.get_weights())
        pose_distill = pose_distill_model.predict_generator(emotion_DataGenerator(model,emotion_path,emotion_label,32))
        # distilled knowledge, generating array
        pose_distill = softmax(pose_distill/distill_t)
        # pseudo confidence from model prediction, generating vector
        pose_confidence_score = pre_model.predict_generator(emotion_DataGenerator(model,emotion_path,emotion_label,32))[1]
        pose_index=np.argmax(pose_confidence_score, axis=1)
        pose_possibility = [pose_confidence_score[i][pose_index[i]] for i in range(np.shape(pose_index)[0])]
        print('confidence:',pose_possibility[:20])
        # weights combination strategy
        is_emotion_classes = False
        if is_distill:
            return pose_distill
        if pure:
            pose_pseudo_1 = np_utils.to_categorical(pose_index, model.pose_classes)
            pose_pseudo = 0.5*pose_pseudo_1 + 0.5*pose_distill
        elif confidence:
            pose_pseudo = Weight_Strategy(model,is_hard_weight,is_emotion_classes,interpolation_weight,pose_possibility,pose_index,pose_distill,selection_threshold)
        elif density or distribution or density_distribution:
            density_distribution_weights = Density_Distribution(model,pre_model,is_emotion_classes,is_density,is_distribution,k,density_t,emotion_path,emotion_label,None,pose_path,pose_label,pose_index)
            pose_pseudo = Weight_Strategy(model,is_hard_weight,is_emotion_classes,interpolation_weight,density_distribution_weights,pose_index,pose_distill,selection_threshold)
        elif confidence_density or confidence_distribution or confidence_density_distribution:
            density_distribution_weights = Density_Distribution(model,pre_model,is_emotion_classes,is_density,is_distribution,k,density_t,emotion_path,emotion_label,None,pose_path,pose_label,pose_index)
            combined_weights = pose_possibility * density_distribution_weights
            
            # print('select samples to visualze:')
            # x = combined_weights/np.max(combined_weights)
            # good_samples = np.where(x>0.9)
            # stored_path = './selected_images/'
            # # os.mkdir(stored_path)
            # print(emotion_path[good_samples])
            # for i in range(np.shape(good_samples)[1]):
            #     shutil.copyfile(emotion_path[good_samples[0][i]],stored_path+'{}_emotion_good.jpg'.format(pose_index[good_samples[0][i]]))
            # bad_samples = np.where(x<0.001)
            # print(emotion_path[bad_samples])
            # for i in range(np.shape(bad_samples)[1]):
            #     shutil.copyfile(emotion_path[bad_samples[0][i]],stored_path+'{}_emotion_bad.jpg'.format(pose_index[bad_samples[0][i]]))
 

            # print('final weights before normalization:',combined_weights[:30])
            # combined_weights = combined_weights/np.max(combined_weights)
            pose_pseudo = Weight_Strategy(model,is_hard_weight,is_emotion_classes,interpolation_weight,combined_weights,pose_index,pose_distill,selection_threshold)
        return pose_pseudo






       
        

        




