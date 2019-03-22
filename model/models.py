from __future__ import print_function
from __future__ import absolute_import
import warnings
import tensorflow as tf
from keras import layers
import keras
from keras.preprocessing import image
import keras.backend as K
from keras.models import Model
from keras.layers import Flatten,Dense,Input,Conv2D, Convolution2D,concatenate
from keras.layers import MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers import BatchNormalization,Activation,SeparableConv2D,PReLU,AveragePooling2D
from keras.regularizers import l2
from keras.layers import Dropout,Reshape,Add,merge
from keras.layers import Input, Lambda
# from keras.applications import ResNet50,MobileNet
# from keras.applications import ResNet50,MobileNet
# from keras.applications.mobilenetv2 import MobileNetV2
from keras_vggface.vggface import VGGFace
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.models import Model
import os
from keras.utils import plot_model
from tensorflow.python.framework.ops import Tensor
from keras.layers.core import Layer
from keras.legacy import interfaces
import tensorflow.contrib as contrib


#'vgg16',     [:19]net layer,      'pool5'
#'resnet50' ,     [:174]net layer,    'avg_pool'
#'senet50' ,    [:286] net layer,        'avg_pool'




def mini_XCEPTION(input_shape, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    model = Model(img_input, x)
    return model

# class GaussianNoise(Layer):
#     @interfaces.legacy_gaussiannoise_support
#     def __init__(self, stddev, **kwargs):
#         super(GaussianNoise, self).__init__(**kwargs)
#         self.supports_masking = True
#         self.stddev = stddev

#     def call(self, inputs, training=None):
#         def noised():
#             return inputs + K.random_normal(shape=K.shape(inputs),
#                                             mean=0.,
#                                             stddev=self.stddev)
#         return K.in_train_phase(noised, inputs, training=training)

#     def get_config(self):
#         config = {'stddev': self.stddev}
#         base_config = super(GaussianNoise, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     def compute_output_shape(self, input_shape):
#         return input_shape


# class Slice(Layer):
#     def __init__(self, emotion_classes,pose_classes, **kwargs):
#         super(Slice, self).__init__(**kwargs)
#         self.emotion_classes = emotion_classes
#         self.pose_classes = pose_classes
 
#     def call(self, inputs):
#         if self.emotion_classes!=None:
#             return inputs[:,:,:self.emotion_classes]
#         elif self.pose_classes!=None: 
#             return inputs[:,:,-1:self.pose_classes]
#     def get_config(self):
#         config = {'emotion_classes': self.emotion_classes,'pose_classes': self.pose_classes}
#         base_config = super(Slice, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#     def compute_output_shape(self, input_shape):
#         return input_shape




class Cross_stitch(Layer):
    def __init__(self,input_shape_1,input_shape_2, **kwargs):
        super(Cross_stitch, self).__init__(**kwargs)
        self.input_shape_1 = input_shape_1
        self.input_shape_2 = input_shape_2
    def build(self, input_shape):
        shape = self.input_shape_1 + self.input_shape_2
        self.cross_stitch = self.add_weight(
            shape=(shape,shape),
            initializer=tf.initializers.identity(),
            name='cross_stitch')
        self.built = True
    def call(self,inputs):
        inputss = tf.concat((inputs[0], inputs[1]), axis=1)
        output = tf.matmul(inputss, self.cross_stitch)
        output1 = tf.reshape(output[:,:self.input_shape_1],shape=[-1,self.input_shape_1])
        output2 = tf.reshape(output[:,self.input_shape_2:],shape=[-1,self.input_shape_2])
        return [output1, output2]
    def get_config(self):
        config = {'input_shape_1': self.input_shape_1,'input_shape_2': self.input_shape_2}
        base_config = super(Cross_stitch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Cross_stitch_multiple(Layer):
    def __init__(self,input_shape_1,input_shape_2,input_shape_3,input_shape_4, **kwargs):
        super(Cross_stitch_multiple, self).__init__(**kwargs)
        self.input_shape_1 = input_shape_1
        self.input_shape_2 = input_shape_2
        self.input_shape_3 = input_shape_3
        self.input_shape_4 = input_shape_4
    def build(self, input_shape):
        shape = self.input_shape_1 + self.input_shape_2 + self.input_shape_3 + self.input_shape_4
        self.cross_stitch = self.add_weight(
            shape=(shape,shape),
            initializer=tf.initializers.identity(),
            name='cross_stitch')
        self.built = True
    def call(self,inputs):
        inputss = tf.concat((inputs[0], inputs[1]), axis=1)
        output = tf.matmul(inputss, self.cross_stitch)
        output1 = tf.reshape(output[:,:self.input_shape_1],shape=[-1,self.input_shape_1])
        output2 = tf.reshape(output[:,:self.input_shape_1:self.input_shape_2],shape=[-1,self.input_shape_2])
        output3 = tf.reshape(output[:,:self.input_shape_2:self.input_shape_3],shape=[-1,self.input_shape_3])
        output4 = tf.reshape(output[:,:self.input_shape_3:self.input_shape_4],shape=[-1,self.input_shape_4])
        return [output1, output2, output3, output4]
    def get_config(self):
        config = {'input_shape_1': self.input_shape_1,'input_shape_2': self.input_shape_2,'input_shape_3': self.input_shape_3,'input_shape_4': self.input_shape_4}
        base_config = super(Cross_stitch_multiple, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Net(Model):
    def __init__(self,model_name,input_type,task_type,emotion_classes,pose_classes,age_classes,gender_classes,is_droput,is_bn,weights_decay):
        self.input_size = 224
        input_shape = (self.input_size,self.input_size,3)
        self.model_name = model_name
        self.input_type = input_type
        self.task_type = task_type
        self.emotion_classes = emotion_classes
        self.gender_classes = gender_classes
        self.age_classes = age_classes
        self.pose_classes = pose_classes
        self.is_droput = is_droput
        self.is_bn = is_bn
        self.weights_decay=weights_decay

        if self.model_name == 'vggFace' or self.model_name=='mobilenetv2' :
            if self.model_name == 'vggFace':
                base = VGGFace(include_top=False,model = 'vgg16',weights='vggface',input_shape=input_shape)
                last_layer = base.get_layer('pool5').output
                # base = VGGFace(include_top=False,model = 'resnet50',weights='vggface',input_shape=input_shape)
                # last_layer = base.get_layer('avg_pool').output
                x = Flatten(name='flatten')(last_layer)
                ###overfit method
                x = Dense(1024,name = 'common_fc',activation='relu')(x)
                # x = Dense(512,name = 'common_fc',W_regularizer=l2(self.weights_decay))(x)
                if self.is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu',name='common_relu')(x)
                if self.is_droput:
                    x = Dropout(0.5)(x)
                
                
            else:
                self.input_size = 96
                base = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
                x = GlobalAveragePooling2D()(base.output)

            if self.input_type == 0:
                image1_batch = Input(shape=input_shape, name='in_t1')
                image2_batch = Input(shape=input_shape, name='in_t2')
                inter_model = Model(inputs=base.input,output= x)
                for layer in inter_model.layers[:-1]:
                    layer.trainable = False
                common1_feat=inter_model(image1_batch)
                common2_feat=inter_model(image2_batch)
                emotion_FC = Dense(128, name='emotion_FC', activation='relu')(common1_feat)
                emotion_out = Dense(self.emotion_classes, name='emotion_prediction', activation='softmax')(emotion_FC)
                gender_FC = Dense(128, name='gender_FC', activation='relu')(common2_feat)
                gender_out = Dense(self.gender_classes, name='gender_prediction', activation='softmax')(gender_FC)
                age_FC = Dense(128, name='age_FC', activation='relu')(common2_feat)
                # age_FC = Dropout(0.5)(age_FC)
                age_out = Dense(self.age_classes, name='age_prediction', activation='softmax')(age_FC)
                # emotion_FC = Dense(self.emotion_classes, name='emotion_FC')(common1_feat)
                # emotion_out = Activation( activation='softmax',name='emotion_prediction')(emotion_FC)
                # gender_FC = Dense(self.gender_classes, name='gender_FC')(common2_feat)
                # gender_out = Activation(activation='softmax',name='gender_prediction')(gender_FC)
                # age_FC = Dense(self.age_classes, name='age_FC', activation='relu')(common2_feat)
                # age_out = Activation(activation='softmax', name='age_prediction')(age_FC)
                if self.task_type == 4:
                    super().__init__(inputs=[image1_batch,image2_batch], outputs=[emotion_out, gender_out, age_out], name='EmotionAgenderNetVGGFace_vgg16')
                elif self.task_type == 3:
                    super().__init__(inputs=[image1_batch,image2_batchs], outputs=[emotion_out,  age_out], name='EmotionAgeNetVGGFace_vgg16')


            else:
                gender_FC = Dense(128, name='gender_FC_', activation='relu')(x)
                emotion_FC = Dense(128, name='emotion_fc')(x)
                pose_FC = Dense(128, name='pose_fc')(x)
                age_FC = Dense(128, name='age_FC')(x)
                if self.is_bn:
                    emotion_FC =  BatchNormalization()(emotion_FC)
                    pose_FC =  BatchNormalization()(pose_FC)
                    age_FC =  BatchNormalization()(age_FC)
                    emotion_pose_FC = BatchNormalization()(emotion_pose_FC)
                pose_FC_1 = Activation('relu',name='pose_relu')(pose_FC)
                emotion_FC_1 = Activation('relu',name='emotion_relu')(emotion_FC)
                age_FC = Activation('relu',name='age_relu')(age_FC)
                is_cross_stich = True
                if is_cross_stich:
                    emotion_FC, pose_FC = Cross_stitch(128,128)([emotion_FC_1, pose_FC_1])
                
                if self.is_droput:
                    emotion_FC=Dropout(0.5)(emotion_FC)
                    pose_FC=Dropout(0.5)(pose_FC)
                    age_FC=Dropout(0.5)(age_FC)
                
                # if is_cross_stich:
                #     emotion_FC, pose_FC, gender_FC, age_FC = Cross_stitch_multiple(128,128,128,128)([emotion_FC,pose_FC,gender_FC,age_FC])


                # gender_FC = Dense(128, name='gender_FC_1', activation='relu')(gender_FC)
                # emotion_FC = Dense(128, name='emotion_fc_1')(emotion_FC)
                # pose_FC = Dense(128, name='pose_fc_1')(pose_FC)
                # age_FC = Dense(128, name='age_FC_1')(age_FC)
                # if self.is_bn:
                #     emotion_FC =  BatchNormalization()(emotion_FC)
                #     pose_FC =  BatchNormalization()(pose_FC)
                #     age_FC =  BatchNormalization()(age_FC)
                #     emotion_pose_FC = BatchNormalization()(emotion_pose_FC)
                # pose_FC = Activation('relu',name='pose_relu_1')(pose_FC)
                # emotion_FC = Activation('relu',name='emotion_relu_1')(emotion_FC)
                # age_FC = Activation('relu',name='age_relu_1')(age_FC)
                # is_cross_stich = True
                # if is_cross_stich:
                #     emotion_FC, pose_FC = Cross_stitch(128,128)([emotion_FC, pose_FC])
                # if self.is_droput:
                #     emotion_FC=Dropout(0.5)(emotion_FC)
                #     pose_FC=Dropout(0.5)(pose_FC)
                #     age_FC=Dropout(0.5)(age_FC)




                
                # emotion_out = Dense(self.emotion_classes, name='emotion_prediction', activation='softmax')(emotion_FC) 
                # pose_out = Dense(self.pose_classes, name='pose_prediction', activation='softmax')(pose_FC)
                # age_out = Dense(self.age_classes, name='age_prediction', activation='softmax')(age_FC)
                ##### one subspace
                age_FC = Dense(self.age_classes,name='age' )(age_FC)
                age_out = Activation('softmax',name='age_prediction')(age_FC)
                ###########
                
                emotion_FC = Dense(self.emotion_classes, name='emotion')(emotion_FC)
                emotion_out = Activation('softmax',name='emotion_prediction')(emotion_FC)

                pose_FC = Dense(self.pose_classes,name='pose')(pose_FC)
                pose_out = Activation('softmax',name='pose_prediction')(pose_FC)

                manifold_emotion = keras.layers.Concatenate(axis=-1,name='manifold_emotion')([emotion_FC_1,emotion_out])
                manifold_pose = keras.layers.Concatenate(axis=-1,name='manifold_pose')([pose_FC_1,pose_out])
               
                    
               

    
                ############
                # emotion_pose_FC = Dense(self.emotion_classes+self.pose_classes,name='emotion_pose' )(emotion_pose_FC)
                # emotion_pose_out = Activation('softmax',name='emotion_pose_prediction')(emotion_pose_FC)
                # EP_emotion_out = Slice(emotion_classes=self.model.emotion_classes,pose_classes=None,name='emotion_out')(emotion_pose_out)
                # EP_pose_out =  Slice(emotion_classes=None,pose_classes=self.model.pose_classes,name='pose_out')(emotion_pose_out)

                gender_out = Dense(self.gender_classes, name='gender_prediction', activation='softmax')(gender_FC)
                
                

                attr_out = []
                attr_name = ['attr{}_predition'.format(i) for i in range(40)]
                for i in range(40):
                    y = Dense(2, activation='sigmoid',name= attr_name[i])(x)
                    attr_out.append(y)


                # emotion_FC = Dense(self.emotion_classes, name='emotion_FC')(x)
                # emotion_out = Activation( activation='softmax',name='emotion_prediction')(emotion_FC)
                # gender_FC = Dense(self.gender_classes, name='gender_FC')(x)
                # gender_out = Activation(activation='softmax',name='gender_prediction')(gender_FC)
                # age_FC = Dense(self.age_classes, name='age_FC', activation='relu')(x)
                # age_out = Activation(activation='softmax', name='age_prediction')(age_FC)
                if self.task_type == 0:
                    super().__init__(inputs=base.input, outputs=emotion_out, name='EmotionNetVGGFace_vgg16')
                elif self.task_type == 1:
                    super().__init__(inputs=base.input, outputs= age_out, name='AgeNetVGGFace_vgg16')
                elif self.task_type == 2:
                    super().__init__(inputs=base.input, outputs=[gender_out, age_out], name='AgenderNetVGGFace_vgg16')
                elif self.task_type == 3:
                    super().__init__(inputs=base.input, outputs=[emotion_out, age_out], name='EmotionAgeNetVGGFace_vgg16')
                elif self.task_type == 4:
                    super().__init__(inputs=base.input, outputs=[emotion_out, gender_out, age_out], name='EmotionAgenderNetVGGFace_vgg16')
                elif self.task_type == 5:
                    super().__init__(inputs=base.input, outputs= pose_out, name='PoseNetVGGFace_vgg16')
                elif self.task_type == 6:
                    super().__init__(inputs=base.input, outputs= [attr_out[0],attr_out[1],attr_out[2],attr_out[3],attr_out[4],
                                                                attr_out[5],attr_out[6],attr_out[7],attr_out[8],attr_out[9],
                                                                attr_out[10],attr_out[11],attr_out[12],attr_out[13],attr_out[14],
                                                                attr_out[15],attr_out[16],attr_out[17],attr_out[18],attr_out[19],
                                                                attr_out[20],attr_out[21],attr_out[22],attr_out[23],attr_out[24],
                                                                attr_out[25],attr_out[26],attr_out[27],attr_out[28],attr_out[29],
                                                                attr_out[30],attr_out[31],attr_out[32],attr_out[33],attr_out[34],
                                                                attr_out[35],attr_out[36],attr_out[37],attr_out[38],attr_out[39]], name='AttriNetVGGFace_vgg16')
                elif self.task_type == 7:
                    super().__init__(inputs=base.input, outputs= [emotion_out,
                                                                attr_out[0],attr_out[1],attr_out[2],attr_out[3],attr_out[4],
                                                                attr_out[5],attr_out[6],attr_out[7],attr_out[8],attr_out[9],
                                                                attr_out[10],attr_out[11],attr_out[12],attr_out[13],attr_out[14],
                                                                attr_out[15],attr_out[16],attr_out[17],attr_out[18],attr_out[19],
                                                                attr_out[20],attr_out[21],attr_out[22],attr_out[23],attr_out[24],
                                                                attr_out[25],attr_out[26],attr_out[27],attr_out[28],attr_out[29],
                                                                attr_out[30],attr_out[31],attr_out[32],attr_out[33],attr_out[34],
                                                                attr_out[35],attr_out[36],attr_out[37],attr_out[38],attr_out[39]
                                                                ], name='Big_EA_VGGFace_vgg16')
                elif self.task_type == 8:
                    super().__init__(inputs=base.input, outputs= [emotion_out,pose_out,gender_out,age_out,
                                                                attr_out[0],attr_out[1],attr_out[2],attr_out[3],attr_out[4],
                                                                attr_out[5],attr_out[6],attr_out[7],attr_out[8],attr_out[9],
                                                                attr_out[10],attr_out[11],attr_out[12],attr_out[13],attr_out[14],
                                                                attr_out[15],attr_out[16],attr_out[17],attr_out[18],attr_out[19],
                                                                attr_out[20],attr_out[21],attr_out[22],attr_out[23],attr_out[24],
                                                                attr_out[25],attr_out[26],attr_out[27],attr_out[28],attr_out[29],
                                                                attr_out[30],attr_out[31],attr_out[32],attr_out[33],attr_out[34],
                                                                attr_out[35],attr_out[36],attr_out[37],attr_out[38],attr_out[39]
                                                                ], name='BigBaselineVGGFace_vgg16')
                elif self.task_type == 9:
                    super().__init__(inputs=base.input, outputs= [emotion_out,pose_out,age_out], name='EPA_VGGFace_vgg16')
                elif self.task_type == 10:
                    super().__init__(inputs=base.input, outputs= gender_out, name='GenderNetVGGFace_vgg16')
                elif self.task_type == 11:
                    manifold=True
                    if manifold:
                        super().__init__(inputs=base.input, outputs= [emotion_out,pose_out,manifold_emotion,manifold_pose], name='EmotionPoseNetVGGFace_vgg16')
                    else:
                        super().__init__(inputs=base.input, outputs= [emotion_out,pose_out], name='EmotionPoseNetVGGFace_vgg16')
                elif self.task_type == 12:
                    super().__init__(inputs=base.input, outputs= [emotion_out,pose_out,gender_out,age_out], name='EPGA-VGGFace_vgg16')

        elif self.model_name == 'mini_xception':
            if self.input_type == 0:
                self.input_size = 64
                input_shape = (64,64,1)
                image1_batch = Input(shape=input_shape, name='in_t1')
                image2_batch = Input(shape=input_shape, name='in_t2')
                inter_model = mini_XCEPTION(input_shape=(self.input_size,self.input_size,1), l2_regularization=0.01)
                common1_feat=inter_model(image1_batch)
                common2_feat=inter_model(image2_batch)
                emotion_layer = Conv2D(self.emotion_classes, (3, 3),padding='same')(common1_feat)
                emotion_layer = GlobalAveragePooling2D()(emotion_layer)
                emotion_layer = Activation('softmax', name='emotion_prediction')(emotion_layer)
                gender_layer = Conv2D(self.gender_classes, (3, 3),padding='same')(common2_feat)
                gender_layer = GlobalAveragePooling2D()(gender_layer)
                gender_layer = Activation('softmax', name='gender_prediction')(gender_layer)
                age_layer = Conv2D(self.age_classes, (3, 3),padding='same')(common2_feat)
                age_layer = GlobalAveragePooling2D()(age_layer)
                age_layer = Activation('sigmoid', name='age_prediction')(age_layer)
                super().__init__(inputs=[image1_batch,image2_batch], outputs=[emotion_layer,gender_layer, age_layer], name='AgenderNetMobileNetV2')
            else:
                self.input_size = 64
                input_shape =  (self.input_size,self.input_size,1)
                base = mini_XCEPTION(input_shape=input_shape, l2_regularization=0.01)
                last_layer = base.output
                emotion_layer = Conv2D(self.emotion_classes, (3, 3),padding='same')(last_layer)
                emotion_layer = GlobalAveragePooling2D(name='gap_emotion')(emotion_layer)
                emotion_layer = Activation('softmax', name='emotion_prediction')(emotion_layer)
                
                gender_layer = Conv2D(self.gender_classes, (3, 3),padding='same')(last_layer)
                gender_layer = GlobalAveragePooling2D()(gender_layer)
                gender_layer = Activation('softmax', name='gender_prediction')(gender_layer)
                
                age_layer = Conv2D(self.age_classes, (3, 3),padding='same')(last_layer)
                age_layer = GlobalAveragePooling2D()(age_layer)
                age_layer = Activation('softmax', name='age_prediction')(age_layer)

                pose_layer = Conv2D(self.pose_classes, (3, 3),padding='same')(last_layer)
                pose_layer = GlobalAveragePooling2D(name='gap_pose')(pose_layer)
                pose_layer = Activation('softmax', name='pose_prediction')(pose_layer)


                is_cross_stich = True
                # if is_cross_stich:
                #     emotion_layer, pose_layer, gender_layer, age_layer = Cross_stitch_multiple(128,128,128,128)([emotion_FC,pose_FC,gender_FC,age_FC])
                
                

                attr_out = []
                attr_name = ['attr{}_predition'.format(i) for i in range(40)]
                for i in range(40):
                    y = Conv2D(2, (3, 3),padding='same')(last_layer)
                    y = GlobalAveragePooling2D()(y)
                    y = Activation('sigmoid', name=attr_name[i])(y)
                    attr_out.append(y)

                if self.task_type == 0:
                    super().__init__(inputs=base.input, outputs=emotion_layer, name='EmotionNetminixception')
                elif self.task_type == 1:
                    super().__init__(inputs=base.input, outputs= age_layer, name='AgeNetNetminixception')
                elif self.task_type == 2:
                    super().__init__(inputs=base.input, outputs=[gender_layer, age_layer], name='AgenderNetEmotionNetminixception')
                elif self.task_type == 3:
                    super().__init__(inputs=base.input, outputs=[emotion_layer, age_layer], name='EmotionAgeNetEmotionNetminixception')
                elif self.task_type == 4:
                    super().__init__(inputs=base.input, outputs=[emotion_layer, gender_layer, age_layer], name='EmotionAgenderEmotionNetminixception')
                elif self.task_type == 5:
                    super().__init__(inputs=base.input, outputs= pose_layer, name='PoseNetminixception')
                elif self.task_type == 6:
                    super().__init__(inputs=base.input, outputs= [attr_out[0],attr_out[1],attr_out[2],attr_out[3],attr_out[4],
                                                                attr_out[5],attr_out[6],attr_out[7],attr_out[8],attr_out[9],
                                                                attr_out[10],attr_out[11],attr_out[12],attr_out[13],attr_out[14],
                                                                attr_out[15],attr_out[16],attr_out[17],attr_out[18],attr_out[19],
                                                                attr_out[20],attr_out[21],attr_out[22],attr_out[23],attr_out[24],
                                                                attr_out[25],attr_out[26],attr_out[27],attr_out[28],attr_out[29],
                                                                attr_out[30],attr_out[31],attr_out[32],attr_out[33],attr_out[34],
                                                                attr_out[35],attr_out[36],attr_out[37],attr_out[38],attr_out[39]], name='AttriNetminixception')
                elif self.task_type == 7:
                    super().__init__(inputs=base.input, outputs= [emotion_layer,
                                                                attr_out[0],attr_out[1],attr_out[2],attr_out[3],attr_out[4],
                                                                attr_out[5],attr_out[6],attr_out[7],attr_out[8],attr_out[9],
                                                                attr_out[10],attr_out[11],attr_out[12],attr_out[13],attr_out[14],
                                                                attr_out[15],attr_out[16],attr_out[17],attr_out[18],attr_out[19],
                                                                attr_out[20],attr_out[21],attr_out[22],attr_out[23],attr_out[24],
                                                                attr_out[25],attr_out[26],attr_out[27],attr_out[28],attr_out[29],
                                                                attr_out[30],attr_out[31],attr_out[32],attr_out[33],attr_out[34],
                                                                attr_out[35],attr_out[36],attr_out[37],attr_out[38],attr_out[39]
                                                                ], name='Big_EA_minixception')
                elif self.task_type == 8:
                    super().__init__(inputs=base.input, outputs= [emotion_layer,pose_layer,
                                                                attr_out[0],attr_out[1],attr_out[2],attr_out[3],attr_out[4],
                                                                attr_out[5],attr_out[6],attr_out[7],attr_out[8],attr_out[9],
                                                                attr_out[10],attr_out[11],attr_out[12],attr_out[13],attr_out[14],
                                                                attr_out[15],attr_out[16],attr_out[17],attr_out[18],attr_out[19],
                                                                attr_out[20],attr_out[21],attr_out[22],attr_out[23],attr_out[24],
                                                                attr_out[25],attr_out[26],attr_out[27],attr_out[28],attr_out[29],
                                                                attr_out[30],attr_out[31],attr_out[32],attr_out[33],attr_out[34],
                                                                attr_out[35],attr_out[36],attr_out[37],attr_out[38],attr_out[39]
                                                                ], name='BigBaselineminixception')
                elif self.task_type == 9:
                    super().__init__(inputs=base.input, outputs= [emotion_layer,pose_layer,age_layer],name='EPA_minixception')
                elif self.task_type == 10:
                    super().__init__(inputs=base.input, outputs= gender_layer,name='GenderNetminixception')
                elif self.task_type == 11:
                    super().__init__(inputs=base.input, outputs= [emotion_layer,pose_layer],name='EmotionPoseNetminixception')
                elif self.task_type == 12:
                    super().__init__(inputs=base.input, outputs= [emotion_layer,pose_layer,gender_layer,age_layer], name='EPGA-minixception')
    @staticmethod
    def decode_prediction(prediction):
        """
        Decode prediction to age and gender prediction.
        Use softmax regression for age and argmax for gender.
        Parameters
        ----------
        prediction : list of numpy array
            Result from model prediction [gender, age]
        Return
        ----------
        gender_predicted : numpy array
            Decoded gender 1 male, 0 female
        age_predicted : numpy array
            Age from softmax regression
        """
        gender_predicted = np.argmax(prediction[0], axis=1)
        age_predicted = prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
        return gender_predicted, age_predicted

    @staticmethod
    def prep_image(data):
        """Preproces image specific to model

        Parameters
        ----------
        data : numpy ndarray
            Array of N images to be preprocessed

        Returns
        -------
        numpy ndarray
            Array of preprocessed image
        """
        data = data.astype('float16')
        data /= 128.
        data -= 1.
        return data

def main():
    MODEL = 'vggFace'
    if MODEL == 'vggFace':
        model = Net(MODEL,1,6,8,2,2)
        model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        model = Net(MODEL,1,5,7,2,2)
        MODEL = model.name

if __name__ == '__main__':
    main()