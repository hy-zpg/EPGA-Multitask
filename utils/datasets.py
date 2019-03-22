from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2
import keras
import math


class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(64, 64)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = '/home/yanhong/Downloads/next_step/Xception/datasets/imdb_crop/'
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '/home/yanhong/Downloads/next_step/Xception/datasets/fer2013/fer2013.csv'
        elif self.dataset_name == 'KDEF':
            self.dataset_path = '/home/yanhong/Downloads/next_step/Xception/datasets/KDEF/'
        elif self.dataset_name == 'ferplus':
            self.dataset_path = '/home/yanhong/Downloads/next_step/Xception/datasets/ferplus/ferplus_whole/'
        elif self.dataset_name =='wiki':
            self.dataset_path = '/home/yanhong/Downloads/next_step/Xception/datasets/wiki_crop/'
        elif self.dataset_name == 'adience':
            self.dataset_path = '/home/yanhong/Downloads/next_step/Xception/datasets/adience_crop/'
        elif self.dataset_name == 'lap':
            self.dataset_path = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/LAP_data/train/'
        elif self.dataset_name == 'mtfl':
            self.dataset_path = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/MTFL/'
        elif self.dataset_name=='stanford40':
            self.dataset_path = '/home/yanhong/Downloads/next_step/action/stanford_40_action_cnn/src/data/raw/JPEGImages'
        else:
            raise Exception('Incorrect dataset name, please input imdb or fer2013')

    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        elif self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()
        elif self.dataset_name == 'KDEF':
            ground_truth_data = self._load_KDEF()
        elif self.dataset_name == 'ferplus':
            emotion_label,ground_truth_data = self._load_ferplus()
        elif self.dataset_name =='wiki':
            ground_truth_data = self._load_wiki()
        elif self.dataset_name == 'adience':
            ground_truth_data = self._load_adience()
        elif self.dataset_name == 'lap':
            ground_truth_data = self._load_lap()
        elif self.dataset_name == 'mtfl':
            ground_truth_data = self._load_mtfl()
        return ground_truth_data

    def _load_imdb(self):
        face_score_treshold = 3
        root_dir = self.dataset_path
        data_path = root_dir + 'imdb.mat'
        dataset = loadmat(data_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        age_classes = dataset['imdb']['age'][0,0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        unknown_age_mask = np.logical_not(np.isnan(age_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        mask = np.logical_and(mask,unknown_age_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        age_classes = age_classes[mask].tolist()
        targets = [np.array(gender_classes),np.array(age_classes)]
        targets = np.transpose(targets)
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names,targets))

    def _load_wiki(self):
        face_score_treshold = 3
        root_dir = self.dataset_path
        data_path = root_dir + 'wiki.mat'
        dataset = loadmat(data_path)
        image_names_array = dataset['wiki']['full_path'][0, 0][0]
        gender_classes = dataset['wiki']['gender'][0, 0][0]
        age_classes = dataset['wiki']['age'][0,0][0]
        face_score = dataset['wiki']['face_score'][0, 0][0]
        second_face_score = dataset['wiki']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        unknown_age_mask = np.logical_not(np.isnan(age_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        mask = np.logical_and(mask,unknown_age_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        age_classes = age_classes[mask].tolist()
        targets = [np.array(gender_classes),np.array(age_classes)]
        targets = np.transpose(targets)
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names,targets))


    def _load_ferplus(self):
        img_names=[]
        targets = []
        label_file = self.dataset_path + 'labels_list/'
        for i in range(8):
            fname = (label_file +'train-aug-c%d.txt' % (i))
            f = open(fname)
            lines = f.readlines()
            f.close()
            for line in lines:
                items = line.split()
                if items[0].split('-')[0] in img_names:
                    continue
                else:
                    img_names.append(items[0].split('-')[0])
                    targets.append(items[1])
        fname = (label_file+ 'validation.txt')
        f=open(fname)
        lines = f.readlines()
        f.close()
        for lines in lines:
            items = line.split()
            if items[0].split('-')[0] in img_names:
                continue
            else:
                img_names.append(items[0].split('-')[0])
                targets.append(items[1])
        return targets,dict(zip(img_names,targets))

    def _load_adience(self):
        img_names=[]
        targets = []
        age_id =([0,4,8,15,25,38,48,60])
        label_file = self.dataset_path + 'labels_list/'
        for i in range(5):
            fname = (label_file +'fold_frontal_%d_data.txt' % (i))
            #print fname
            f = open(fname)
            lines = f.readlines()
            lines = lines[1:]
            #print lines
            f.close()
            for line in lines:
                items = line.split()
                if len(items)!=13:
                    #print  'error_length:',len(items)
                    continue
                elif items[5] == 'u':
                    # print 'error_gender',items[5]
                    continue
                elif int(items[3].split('(')[1].split(',')[0]) not in age_id:
                    #print 'age not in age_id',int(items[3].split('(')[1].split(',')[0])
                    continue
                else:
                    #print len(items),items[5],age_id.index(int(items[3].split('(')[1].split(',')[0]))
                    #image_name = 'coarse_tilt_aligned_face.'+items[1]+'.'+items[0].split('-')[0]
                    image_name = items[0] + '/' + 'coarse_tilt_aligned_face.' + items[2]+'.'+items[1]
                    age = age_id.index(int(items[3].split('(')[1].split(',')[0]))
                    gender = 0 if items[5]=='f' else 1
                    target = [gender,age]
                    #print target
                img_names.append(image_name)
                targets.append(target)
        return dict(zip(img_names,targets))

    def _load_lap(self):
        img_names=[]
        targets = []
        fname= self.dataset_path + 'train_gt.txt'
        f=open(fname)
        lines = f.readlines()
        lines = lines[1:]
        f.close()
        for line in lines:
            items = line.split(',')
            #print items
            #print items[1]
            if items[0] in img_names:
                continue
            else:
                img_names.append(items[0])
                targets.append(items[1:4])
        return  dict(zip(img_names,targets))




    def _load_mtfl(self):
        image_names=[]
        targets = []
        train_f = self.dataset_path + 'training.txt' 
        val_f   = self.dataset_path + 'testing.txt'  
        f = open(train_f)
        lines = f.readlines()
        lines = lines[1:]
        f.close()
        for line in lines:
            items = line.split()
            if items[0] in image_names:
                continue
            else:
                image_names.append(items[0])
                targets.append(int(int(items[11]) == 1))
        f=open(val_f)
        lines = f.readlines()
        lines = lines[1:]
        f.close()
        for line in lines:
            items = line.split()
            if items[0] in image_names:
                continue
            else:
                image_names.append(items[0])
                targets.append(int(int(items[11]) == 1))
        return dict(zip(image_names,targets))

        def  _load_stanford(self):
            image_names = listdir(self.dataset_path)
            for image_name in image_names:
                return False





    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions


    def _load_KDEF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)
        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))
        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions




def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    elif dataset_name == 'wiki':
        return {0:'woman',1:'man'}
    elif dataset_name == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    elif dataset_name == 'ferplus':
        return {0:'neutral',1:'happy',2:'surprise',3:'sadness',4:'anger',5:'disgust',6:'fear',7:'contempt'}
    elif dataset_name == 'adience':
        return {0:'0-2',1:'4-6',2:'8-12',3:'15-20',4:'25-32',5:'38-43',6:'48-53',7:'60-100',8:'woman',9:'man'}
    elif dataset_name == 'expw':
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'aflw':
        return{0:'frontal',1:'right profile',2:'left profile',3:'left',
                4:'right'}
    else:
        raise Exception('Invalid dataset name')

def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4,
                'surprise':5, 'neutral':6}
    elif dataset_name == 'imdb':
        return {'woman':0, 'man':1}
    elif dataset_name == 'KDEF':
        return {'AN':0, 'DI':1, 'AF':2, 'HA':3, 'SA':4, 'SU':5, 'NE':6}
    elif dataset_name == 'perplus':
        return {'neutral':0,'happiness':1,'surprise':2,'sadness':3,'anger':4,'disgust':5,'fear':6,'contempt':7}
    elif dataset_name == 'wiki':
        return {'woman':0,'man':1}
    elif dataset_name == 'stanford40':
        return{'':0,'':1,'':2,'':3,'':4,'':5,'':6}
    elif dataset_name=='adience':
        return{''}
    else:
        raise Exception('Invalid dataset name')

#split the whole dataset into training sets and validation sets
def split_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle == True:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys


   


def split_data_1(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

def _load_train_ferplus(data_path,image_size):
    traindir = data_path + 'ferplus_images/'
    d_ = {}
    maxnum = 0
    for i in range(8):
        fname = (data_path+'labels_list/train-aug-c%d.txt' % (i))
        f = open(fname)
        alllines = f.readlines()
        f.close()
        if maxnum < len(alllines):
            maxnum = len(alllines)
        d_[i] = alllines
    weight,height = image_size
    x_train = np.zeros((maxnum*8, weight, height, 1), dtype='f')
    x_p = np.zeros((maxnum*8, weight, height, 1), dtype='f')
    y_p = np.zeros((maxnum*8), dtype=np.uint8)
    idxout = 0
    for i in range(8):
        data = d_[i]
        num = 0
        idx = 0    
        while num < maxnum:
            sp = (data[idx]).split(" ")
            y_p[idxout ] = int(sp[1])
            img_path = (sp[0].split('-')[0]+ '.png')
            img = cv2.imread(os.path.join(traindir, img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            x_p[idxout, :, :, 0] = img
            idxout = idxout + 1
            idx = idx + 1
            if idx >= len(data):
                idx = 0
            num = num + 1
    randomid = np.random.permutation(x_p.shape[0])
    y_p = keras.utils.to_categorical(y_p, num_classes=8)
    y_train = np.zeros(y_p.shape, dtype=np.uint8)
    for i in range(x_p.shape[0]):
        x_train[i, :, :, :] = x_p[randomid[i], :, :, :].copy()
        y_train[i] = y_p[randomid[i]]
    return x_train,y_train

def _load_valid_ferplus(data_path,image_size):
    valdir = data_path +'ferplus_images/'
    f = open(data_path+'labels_list/validation.txt')
    alllines = f.readlines()
    f.close()
    weight,height = image_size
    x_val = np.zeros((len(alllines), weight, height, 1), dtype='f')
    y_p = np.zeros((len(alllines)), dtype=np.uint8)
    idx = 0
    for line in alllines:
        sp = line.split(" ")
        y_p[idx] = int(sp[1])
        img_valid_path = sp[0].split('_')[0]
        img = cv2.imread(os.path.join(valdir, img_valid_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)
        x_val[idx, :, :, 0] = img
        idx = idx + 1
    y_val = keras.utils.to_categorical(y_p, num_classes=8)
    return x_val,y_val


