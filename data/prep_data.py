import os
import cv2
# import dlib
import argparse
import pandas as pd
# from tqdm import tqdm
from multiprocessing import Pool
import csv
import glob
import os.path
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset',
#                     required=True,
#                     choices=['imdb', 'wiki', 'utkface', 'fgnet', 'adience'],
#                     help='Dataset name')
# parser.add_argument('--num_worker',
#                     default=2,
#                     type=int,
#                     help="num of worker")
# args = parser.parse_args()
# DATASET = args.dataset
# WORKER = args.num_worker
# predictor = dlib.shape_predictor('../model/shape_predictor_5_face_landmarks.dat')
def align_and_save(path):
    RES_DIR = '{}_aligned'.format(DATASET)
    if os.path.exists(os.path.join(RES_DIR, path)):
        return 1
    flname = os.path.join(DATASET, path)
    image = cv2.imread(flname)
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 0)
    # if detect exactly 1 face, get aligned face
    if len(rects) == 1:
        shape = predictor(image, rects[0])
        result = dlib.get_face_chip(image, shape, padding=0.4, size=140)
        folder = os.path.join(RES_DIR, path.split('/')[0])
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        flname = os.path.join(RES_DIR, path)
        if not os.path.exists(flname):
            cv2.imwrite(flname, result)
        return 1
    return 0


def ferplus():
    liness=[]
    img_paths=[]
    label_file = './ferplus/' + 'labels_list/'
    for i in range(8):
        fname = (label_file +'train-aug-c%d.txt' % (i))
        f = open(fname)
        lines = f.readlines()
        for line in lines:
            items = line.split()
            img_path = 'ferplus/ferplus_whole/'+(items[0].split('-')[0]) + '.jpg'
            img_path1 = 'data/ferplus/ferplus_whole/'+(items[0].split('-')[0]) + '.jpg'
            if img_path in (img_paths):
                continue
            if not os.path.exists(img_path):
                print('no such file')
                continue
            db = 'ferplus'
            data = [db,img_path1,items[1]]
            liness.append(data)
            img_paths.append(img_path)
        f.close()
    fname = (label_file+ 'validation.txt')
    f=open(fname)
    lines = f.readlines()
    for line in lines:
        items = line.split()
        img_path = 'ferplus/ferplus_whole/' + (items[0].split('.')[0]) + '.jpg'
        img_path1 = 'data/ferplus/ferplus_whole/'+(items[0].split('.')[0]) + '.jpg'
        print(img_path)
        if img_path in (img_paths):
            continue
        if not os.path.exists(img_path):
            print('no such file')
            continue
        db = 'ferplus'
        data = [db,img_path1,items[1]]
        liness.append(data)
        img_paths.append(img_path)
    f.close()
    name = ['db_name','full_path','emotion']
    test = pd.DataFrame(columns=name,data=liness)
    print(len(liness))
    test.to_csv('./db/ferplus_cleaned.csv',encoding='gbk')



def adience():
    datas = []
    age_id =([0,4,8,15,25,38,48,60])
    label_file = 'Adience/labels_list/'
    img_file = 'Adience/faces/'
    img_file1 = 'data/Adience/faces/'
    counts=np.zeros(8)
    for i in range(5):
        fname = (label_file +'fold_%d_data.txt' % (i))
        f = open(fname)
        lines = f.readlines()
        lines = lines[1:]
        f.close()
        for line in lines:
            items = line.split()
            image_name = items[0] + '/' + 'coarse_tilt_aligned_face.' + items[2]+'.'+items[1]
            full_path = img_file + image_name
            full_path1 = img_file1 + image_name
            # if len(items)!=13:
            #     continue
            if not os.path.exists(full_path):
                continue
            elif items[5] == 'u':
                continue
            elif int(items[3].split('(')[1].split(',')[0]) not in age_id:
                continue
            else:
                db_name = 'adience'
                age = age_id.index(int(items[3].split('(')[1].split(',')[0]))
                counts[age]+=1
                gender = 0 if items[5]=='f' else 1
                face_id = items[1]
                tilt_ang = items[9]
                fiducial_yaw_angle = items[10]
                data = [db_name,full_path1,gender,age,face_id,tilt_ang,fiducial_yaw_angle]
            datas.append(data)
    print(counts)
    name = ['db_name','full_path','gender','age','face_id','tilt_ang','fuducial_yaw_angle']
    test = pd.DataFrame(columns=name,data=datas)
    print(len(datas))
    test.to_csv('./db/adience_cleaned.csv',encoding='gbk')



def extract_SFEW_files():
    dict_label = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4,
                'Surprise':5, 'Neutral':6}
    data_file = []
    folders = ['SFEW 2.0/Train/Train_Aligned_Faces/', 'SFEW 2.0/Val/Val_Aligned_Faces/']

    for folder in folders:
        class_folders = glob.glob(folder + '*')

        for emotion_class in class_folders:
            class_files = glob.glob(emotion_class + '/*.png')
            for img_path in class_files:
                label = emotion_class.split('/')[-1]
                num_label = dict_label[label]
                path = 'data/'+img_path
                data_file.append([num_label, path])
    with open('./db/SFEW_cleaned.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)
    print("Extracted and wrote %d files." % (len(data_file)))

def extract_FER2013_files():
    folders = ['fer2013/Training/','fer2013/PublicTest/','fer2013/PrivateTest/']
    data_file=[]
    for folder in folders:
        class_folders = glob.glob(folder + '*')
        for emotion_class in class_folders:
            print('label',emotion_class)
            class_files = glob.glob(emotion_class + '/*.jpg')
            for img_path in class_files:
                print('img',img_path)
                label = emotion_class.split('/')[-1]
                path = 'data/'+img_path
                db = 'fer2013'
                print(os.path.getsize('/home/user/hy_mtl/'+path))
                if os.path.getsize('/home/user/hy_mtl/'+path)<400:
                    continue
                data_file.append([db,path,label])
    name = ['db_name','full_path','emotion']
    test = pd.DataFrame(columns=name,data=data_file)
    print(len(data_file))
    test.to_csv('./db/fer2013_cleaned.csv',encoding='gbk')


def mtfl():
    datas = []
    paths = ['MTFL/training.txt','MTFL/testing.txt']
    for path in paths:
        for line in open(path).readlines():
            items=line.split()
            landmark = items[1:11]
            landmark = [float(x) for x in items[1:11]]
            landmark = np.array((landmark),np.float64)  
            # landmark = np.transpose(landmark) 
            # landmark = landmark.reshape(-1,1)
            gender = int(int(items[11]) == 1) ## 1 for male, 0 for female
            smile =  int(int(items[12]) == 1) ## 1 for smiling, 0 for non smiling
            glass =  int(int(items[13]) == 1) ## 1 for wearing glasses, 0 for no glasses
            pose = int(int(items[14]))-1 ## 5 kinds of pose
            
            datas.append(['mftl','data/MTFL/'+items[0],landmark, gender, smile, glass, pose] )
    
    name = ['db_name','full_path','landmark','gender','smile','glass','pose']
    test = pd.DataFrame(columns=name,data=datas)
    print(len(datas))
    test.to_csv('./db/mtfl_cleaned.csv',encoding='gbk')

face_detector_path = 'haarcascade_frontalface_default.xml'


def crop_face(face_detector_path,current_image_path,stored_path):
    face_cascade=cv2.CascadeClassifier(face_detector_path)
    img = cv2.imread(current_image_path)
    img = cv2.resize(img,(396,396))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #will display another window with same vedio feed but grey in color .You can always use different                                 option and for that refer the official Documentations.
    faces=face_cascade.detectMultiScale(gray,1.3,3) 
    if len(faces) < 1:
        return 0
    else:                       
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite(stored_path,crop_img)
        return 1

def megaage_align_and_save(store_path = '/home/yanhong/Downloads/next_step/Xception/datasets/megaage_asian_croped/'):
    sfew = pd.read_csv('db/megaage_asian.csv')
    data = sfew
    del sfew
    paths = data['full_path'].values
    age_label = data['age'].values.astype('uint8')
    length = len(paths)
    datas = []
    if os.path.exists(store_path):
        return 1
    else:
        os.makedirs(store_path)

    for i in range(length):
        current_img_path = paths[i].split('/')[-1]
        future_img_path = os.path.join(store_path,current_img_path)
        future_age_label = age_label[i]
        crop_face(face_detector_path,paths[i],future_img_path)
        datas.append([future_img_path,future_age_label])
    print('finish')
    name = ['full_path','age']
    test = pd.DataFrame(columns=name,data=datas)
    print('croped_length:',len(datas))
    test.to_csv('./db/megaage_asian_cleaned.csv',encoding='gbk')

def imdb():
    imdb = pd.read_csv('db/imdb.csv')
    data = imdb
    del imdb
    db = data['db_name'].values
    paths = data['full_path'].values
    age_label = data['age'].values.astype('uint8')
    gender_label = data['gender'].values.astype('uint8')
    length = len(db)
    datas = []
    for i in range(length):
        new_db = 'imdb'
        new_path = 'data/imdb/'  + paths[i]
        # print(db[i] + '/'+paths[i])
        # print('imdb_crop' + '/'+paths[i])
        if not os.path.exists('imdb' + '/'+paths[i]):
            continue
        new_gender_label = gender_label[i]
        new_age_label = age_label[i]
        datas.append([new_db,new_path,new_gender_label,new_age_label])
    name = ['db_name','full_path','gender','age']
    test = pd.DataFrame(columns=name,data=datas)
    print('length:',len(datas))
    test.to_csv('./db/imdb_cleaned.csv',encoding='gbk')

# def megaage():
#     imdb = pd.read_csv('db/megaage_asian_cleaned_old.csv')
#     data = imdb
#     del imdb
#     paths = data['full_path'].values
#     age_label = data['age'].values.astype('uint8')
#     length = len(paths)
#     print('ori',length)
#     datas = []
#     for i in range(length): 
#         if not os.path.exists(paths[i]):
#             continue
#         new_db = 'megaage'
#         new_path = paths[i]
#         new_age_label = age_label[i]
#         datas.append([new_db,new_path,new_age_label])
#     name = ['db_name','full_path','age']
#     test = pd.DataFrame(columns=name,data=datas)
#     print('length:',len(datas))
#     test.to_csv('./db/megaage_asian_cleaned.csv',encoding='gbk')

# def adience():
#     adience = pd.read_csv('db/adience.csv')
#     data = adience
#     del adience
#     db = data['db_name'].values
#     paths = data['full_path'].values
#     age_label = data['age'].values.astype('uint8')
#     gender_label = data['gender'].values.astype('uint8')
#     length = len(db)
#     datas = []
#     for i in range(length):
#         if not os.path.exists(paths[i]):
#             continue
#         else:
#             new_db = 'adience'
#             new_path = paths[i]
#             new_gender_label = gender_label[i]
#             new_age_label = age_label[i]
#         datas.append([new_db,new_path,new_gender_label,new_age_label])
#     name = ['db_name','full_path','gender','age']
#     test = pd.DataFrame(columns=name,data=datas)
#     print('length:',len(datas))
#     test.to_csv('./db/adience_cleaned.csv',encoding='gbk')   

# def AFLW_crop():
#     data1 = pd.DataFrame()
#     data = pd.DataFrame()
#     path = ['/home/yanhong/Downloads/next_step/Xception/datasets/AFLW/faceid.csv','/home/yanhong/Downloads/next_step/Xception/datasets/AFLW/facepose.csv','/home/yanhong/Downloads/next_step/Xception/datasets/AFLW/facerect.csv']
#     img_file = '/home/yanhong/Downloads/next_step/Xception/datasets/AFLW/images/'
#     crop_file = '/home/yanhong/Downloads/next_step/Xception/datasets/AFLW/cropped_images/'



def AFLW():
    data1 = pd.DataFrame()
    data = pd.DataFrame()
    path = ['AFLW/faceid.csv','AFLW/facepose.csv','AFLW/facerect.csv']
    img_file = 'AFLW/images/'
    crop_file = 'AFLW/cropped_images/'
    data_id = pd.read_csv(path[0])
    data_pose = pd.read_csv(path[1])
    data_rect = pd.read_csv(path[2])

    if os.path.exists(crop_file):
        print('have existed!')
    else:
        os.makedirs(crop_file)

    data1 = data_id['face_id']
    data11 = data_id['file_id']
    id_img_index=dict(zip(data1, data11))

    data2 = data_pose['face_id']
    data22_roll = data_pose['roll']
    data22_pitch = data_pose['pitch']
    data22_yaw = data_pose['yaw']
    id_roll_index = dict(zip(data2,data22_roll))
    id_pitch_index = dict(zip(data2,data22_pitch))
    id_yaw_index = dict(zip(data2,data22_yaw))

    data3 = data_rect['face_id']
    data33_x = data_rect['x']
    data33_y = data_rect['y']
    data33_w = data_rect['w']
    data33_h = data_rect['h']
    id_x_index = dict(zip(data3,data33_x))
    id_y_index = dict(zip(data3,data33_y))
    id_w_index = dict(zip(data3,data33_w))
    id_h_index = dict(zip(data3,data33_h))

    datas =[]
    for face_id in data2:
        img_path = img_file + id_img_index[face_id]
        if face_id not in id_img_index.keys():
            continue
        if face_id not in id_x_index.keys():
            continue
        if not os.path.exists(img_path):
            print('no such file')
            continue
        store_path = crop_file + id_img_index[face_id]
        if not os.path.exists(store_path):
            # print('continue crop')
            # img = cv2.imread(img_path)
            # x = int(id_x_index[face_id])
            # y = int(id_y_index[face_id])
            # w =  int(id_w_index[face_id])
            # h = int(id_h_index[face_id])
            # crop_img = img[y:y+h, x:x+w]
            # cv2.imwrite(store_path,crop_img)
            continue

        datas.append(['aflw','data/'+store_path,face_id,id_roll_index[face_id],id_pitch_index[face_id],id_yaw_index[face_id]])
    name = ['db_name','full_path','face_id','roll','pitch','yaw']
    test = pd.DataFrame(columns=name,data=datas)
    print('length:',len(datas))
    test.to_csv('./db/aflw_cleaned.csv',encoding='gbk')


def EXPW():
    path = 'ExpW/data/label/label.txt'
    img_file = 'ExpW/data/image/origin/'
    crop_file = 'ExpW/data/image/cropped_images/'

    if os.path.exists(crop_file):
        print('have existed!')
    else:
        os.makedirs(crop_file)

    f = open(path,"r") 
    datas=[]
    counts=np.zeros(7)
    for line in f:
        linelist =line.split()
        img_path = img_file + linelist[0]
        if not os.path.exists(img_path):
            print ('no this file ')
        else:
            stored_path = crop_file + linelist[0]
            if not os.path.exists(stored_path):
                print('no such file')
                continue
            else:
                # img = cv2.imread(img_path)
                # x = int(linelist[2])
                # y = int(linelist[3])
                # w =  int(linelist[4])
                # h = int(linelist[5])
                # crop_img = img[x:h, y:w]
                # print(np.shape(crop_img))
                # cv2.imwrite(stored_path,crop_img)
                face_id = linelist[1]
                full_path = 'data/'+stored_path
                emotion = linelist[7]
                counts[int(emotion)]+=1
                db_name = 'expw'
                datas.append([full_path,db_name,face_id,emotion])
    f.close()
    print(counts)
    name = ['full_path','expw','face_id','emotion']
    test = pd.DataFrame(columns=name,data=datas)
    print('length:',len(datas))
    test.to_csv('./db/expw_cleaned.csv',encoding='gbk')


def CELEBA():
    data1 = pd.DataFrame()
    data = pd.DataFrame()
    path = 'CelebA/Anno/list_attr_celeba.txt'
    img_file = 'CelebA/Img/img_celeba.7z/cropped_images_official/'
    dictionary = {}
    f = open(path,"r")
    i = 0 
    name = []
    img_names = []
    for line in f :
        i = i + 1
        if i==1:
            name = [i for i in range(40)]
            name.append('db_name')
            name.append('full_path')

        listValue = []
        if(i >= 2): 
            linelist =line.split()
            img_names.append(linelist[0])
            for attribute in linelist:
                if attribute == "-1" :
                    listValue.append(float(0.0))
                elif attribute == "1":
                    listValue.append(float(1.0))
            dictionary[linelist[0]] = listValue
    f.close()
    datas = []
    for img_name in img_names:
        img_path = img_file + img_name
        if os.path.exists(img_path):
            label = dictionary[img_name]
            label.append('celeba')
            label.append('data/'+img_path)
            datas.append(label)
    test = pd.DataFrame(columns=name,data=datas)
    print('length:',len(test))
    test.to_csv('./db/celeba_cleaned.csv',encoding='gbk')

def CELEBA_crop():
    data1 = pd.DataFrame()
    data = pd.DataFrame()
    path = '/home/yanhong/Downloads/next_step/Xception/datasets/CelebA/Anno/list_bbox_celeba.txt'
    img_file = '/home/yanhong/Downloads/next_step/Xception/datasets/CelebA/Img/img_celeba.7z/img_celeba/'
    crop_file = '/home/yanhong/Downloads/next_step/Xception/datasets/CelebA/Img/img_celeba.7z/cropped_images_official/'
    f = open(path,"r")
    i = 0 
    name = []
    img_names = []

    if os.path.exists(crop_file):
        print('have existed!')
    else:
        os.makedirs(crop_file)
    for line in f :
        i = i + 1
        if(i >= 2): 
            linelist =line.split()
            img_path = img_file + linelist[0]
            if not os.path.exists(img_path):
                print ('no this file ')
            else:
                stored_path = crop_file + linelist[0]
                img = cv2.imread(img_path)
                x = int(linelist[1])
                y = int(linelist[2])
                w =  int(linelist[3])
                h = int(linelist[4])
                img = cv2.rectangle(img,(x,y),(x+w,y+h),[0,0,255],1)
                crop_img = img[y:y+h, x:x+w]
                print(np.shape(crop_img))
                cv2.imwrite(stored_path,crop_img)

def adience_crop():
    path = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/Adience/'
    img_file = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/Adience/faces/'
    crop_file = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/cropped_Adience/'
    label_file = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/Adience/labels_list/'
    if os.path.exists(crop_file):
        print('have existed!')
    else:
        os.makedirs(crop_file)
    for i in range(5):
        fname = (label_file +'fold_%d_data.txt' % (i))
        f = open(fname)
        lines = f.readlines()
        lines = lines[1:]
        f.close()
        for line in lines:
            items = line.split()
            image_name = items[0] + '/' + 'coarse_tilt_aligned_face.' + items[2]+'.'+items[1]
            img_path =  img_file + image_name
            if not os.path.exists(img_path):
                print ('no such file')
            else:
                stored_path = crop_file + image_name
                img = cv2.imread(img_path)
                x = int(items[6])
                y = int(items[7])
                w =  int(items[8])
                h = int(items[9])
                # img = cv2.rectangle(img,(x,y),(x+w,y+h),[0,0,255],1)
                crop_img = img[y:y+h, x:x+w]
                print(np.shape(crop_img))
                cv2.imwrite(stored_path,crop_img)

    


def yaw_transfer(yaw,count):
    yaw_degree=180/np.pi*yaw
    if np.abs(yaw_degree)<10:
        count[0]+=1
        return 0
    elif 10<=yaw_degree and yaw_degree<60:
        count[1]+=1
        return 1
    elif -60<yaw_degree and yaw_degree<=-10:
        count[2]+=1
        return 2
    elif yaw_degree<=-60:
        count[3]+=1
        return 3
    else:
        count[4]+=1
        return 4
def aflw_dgree_classes():
     aflw = pd.read_csv('db/aflw_cleaned_degree.csv')
     data = aflw
     del aflw
     db = data['db_name'].values
     paths = data['full_path'].values
     yaw_label = data['yaw'].values.astype('float64')
     print('maxmin:',np.max(yaw_label),np.min(yaw_label))
     count=np.zeros([5])
     length = len(db)
     datas = []
     for i in range(length):
         new_db = 'aflw'
         new_path = paths[i]       
         new_yaw = yaw_transfer(yaw_label[i],count)
         datas.append([new_db,new_path,new_yaw,yaw_label[i]])
     name = ['db_name','full_path','pose','yaw']
     test = pd.DataFrame(columns=name,data=datas)
     print('length:',len(datas))
     print(count)
     test.to_csv('./db/aflw_cleaned.csv',encoding='gbk')


def change_fullpath(dataset):
    file_path = 'db/{}_cleaned.csv'.format(dataset) 
    data1 = pd.read_csv(file_path)
    data = data1
    del data1
    paths = data['full_path'].values
    full_paths = []
    for i in range(len(paths)):
        print(paths[i])
        print(paths[i].split('/')[7:])
        full_paths.append(paths[i].split('/')[7:])
    data['full_path'] = full_paths


def new_SFEW_files():
    dict_label = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4,
                'Surprise':5, 'Neutral':6}
    data_file = []
    folders = ['SFEW 2.0/Train/Train_Aligned_Faces/', 'SFEW 2.0/Val/Val_Aligned_Faces/']
    data_train = pd.DataFrame()
    data_valid = pd.DataFrame()
    for i,folder in enumerate (folders):
        class_folders = glob.glob(folder + '*')
        for emotion_class in class_folders:
            class_files = glob.glob(emotion_class + '/*.png')
            for img_path in class_files:
                label = emotion_class.split('/')[-1]
                num_label = dict_label[label]
                path = 'data/'+img_path
                data_file.append([path,num_label])
        # name = ['full_path','emotion']
        # test = pd.DataFrame(columns=name,data=data_file)
        # num = num+len(data_file)
        # print('length:',len(data_file))
        # if i==0:
        #     test.to_csv('./db/SFEW_train_cleaned.csv',encoding='gbk')
        #     data_file = []
        # else:
        #     test.to_csv('./db/SFEW_valid_cleaned.csv',encoding='gbk')
        #     data_file = []
    name = ['full_path','emotion']
    test = pd.DataFrame(columns=name,data=data_file)
    test.to_csv('./db/SFEW_cleaned.csv',encoding='gbk')

    print("Extracted and wrote %d files." % (len(data_file)))


def main():
    args = parser.parse_args()
    DATASET = args.dataset
    WORKER = args.num_worker
    data = pd.read_csv('db/{}.csv'.format(DATASET))
    # detector = dlib.get_frontal_face_detector()

    paths = 'faces/'+ data['full_path'].values

    print('[PREPROC] Run face alignment...')
    with Pool(processes=WORKER) as p:
        res = []
        max_ = len(paths)
        with tqdm(total=max_) as pbar:
            for i, j in tqdm(enumerate(p.imap(align_and_save, paths))):
                pbar.update()
                res.append(j)
        data['flag'] = res

        # create new db with only successfully detected face
        data = data.loc[data['flag'] == 1, list(data)[:-1]]
        data.to_csv('db/{}_cleaned.csv'.format(DATASET), index=False)





if __name__ == '__main__':
    imdb()