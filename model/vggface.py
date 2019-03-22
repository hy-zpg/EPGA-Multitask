from __future__ import print_function
from __future__ import absolute_import
import warnings
from keras import layers
import keras.backend as K
from keras.models import Model
from keras.layers import Flatten,Dense,Input,Conv2D, Convolution2D,concatenate
from keras.layers import MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers import BatchNormalization,Activation,SeparableConv2D,PReLU,AveragePooling2D
from keras.regularizers import l2
from keras.layers import Dropout,Reshape,Add,merge
from keras.applications import ResNet50,MobileNet
from keras.applications import ResNet50,MobileNet
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.models import Model
import os
from keras.applications.mobilenetv2 import MobileNetV2
from keras.utils import plot_model


#'vgg16',     [:19]net layer,      'pool5'
#'resnet50' ,     [:174]net layer,    'avg_pool'
#'senet50' ,    [:286] net layer,        'avg_pool'

class AgenderNetVGGFacenet(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 224
        base = VGGFace(include_top=False,model = 'vgg16',weights='vggface',input_shape=(224,224,3))
        last_layer = base.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(1024,name = 'common_fc',activation='relu')(x)
        
        emotion_FC = Dense(128, name='emotion_FC_1', activation='relu')(x)
        emotion_out = Dense(7, name='emotion_prediction', activation='softmax')(emotion_FC)

        gender_FC = Dense(128, name='gender_FC_1', activation='relu')(x)
        gender_out = Dense(2, name='gender_prediction', activation='softmax')(gender_FC)

        age_FC = Dense(128, name='age_FC_1', activation='relu')(x)
        age_out = Dense(8, name='age_prediction', activation='softmax')(age_FC)


        super().__init__(inputs=base.input, outputs=[gender_out, age_out], name='AgenderNetVGGFace_vgg16')

    def prep_phase1(self):
        """Freeze layer from input until block_14
        """
        for layer in self.layers[:130]:
            layer.trainable = False
        for layer in self.layers[130:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until blovk_8
        """
        for layer in self.layers[:78]:
            layer.trainable = False
        for layer in self.layers[78:]:
            layer.trainable = True

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

class AgeNetVGGFacenet(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 224
        base = VGGFace(include_top=False,model = 'vgg16',weights='vggface',input_shape=(224,224,3))
        last_layer = base.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(1024,name = 'common_fc',activation='relu')(x)
        
        emotion_FC = Dense(128, name='emotion_FC_1', activation='relu')(x)
        emotion_out = Dense(7, name='emotion_prediction', activation='softmax')(emotion_FC)

        gender_FC = Dense(128, name='gender_FC_1', activation='relu')(x)
        gender_out = Dense(2, name='gender_prediction', activation='softmax')(gender_FC)

        age_FC = Dense(128, name='age_FC_1', activation='relu')(x)
        age_out = Dense(71, name='age_prediction', activation='linear')(age_FC)


        super().__init__(inputs=base.input, outputs=age_out, name='AgeNetVGGFace_vgg16')

    def prep_phase1(self):
        """Freeze layer from input until block_14
        """
        for layer in self.layers[:130]:
            layer.trainable = False
        for layer in self.layers[130:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until blovk_8
        """
        for layer in self.layers[:78]:
            layer.trainable = False
        for layer in self.layers[78:]:
            layer.trainable = True

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



class EmotionNetVGGFacenet(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 224
        base = VGGFace(include_top=False,model = 'vgg16',weights='vggface',input_shape=(224,224,3))
        last_layer = base.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)

        x = Dense(1024,name = 'common_fc',activation='relu')(x)
        
        emotion_FC = Dense(128, name='emotion_FC_1', activation='relu')(x)
        emotion_out = Dense(8, name='emotion_prediction', activation='softmax')(emotion_FC)

        gender_FC = Dense(128, name='gender_FC_1', activation='relu')(x)
        gender_out = Dense(2, name='gender_prediction', activation='softmax')(gender_FC)

        age_FC = Dense(128, name='age_FC_1', activation='relu')(x)
        age_out = Dense(101, name='age_prediction', activation='softmax')(age_FC)

        super().__init__(inputs=base.input, outputs=emotion_out, name='EmotionNetVGGFace_vgg16')

    def prep_phase1(self):
        """Freeze layer from input until block_14
        """
        for layer in self.layers[:130]:
            layer.trainable = False
        for layer in self.layers[130:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until blovk_8
        """
        for layer in self.layers[:78]:
            layer.trainable = False
        for layer in self.layers[78:]:
            layer.trainable = True

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
        emotion_predicted = np.argmax(prediction, axis=1)
        return emotion_predicted

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

class MultitaskVGGFacenet(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 224
        base = VGGFace(include_top=False,model = 'vgg16',weights='vggface',input_shape=(224,224,3))
        last_layer = base.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        
        x = Dense(1024,name = 'common_fc',activation='relu')(x)
        
        emotion_FC = Dense(128, name='emotion_FC_1', activation='relu')(x)
        emotion_out = Dense(8, name='emotion_prediction', activation='softmax')(emotion_FC)

        gender_FC = Dense(128, name='gender_FC_1', activation='relu')(x)
        gender_out = Dense(2, name='gender_prediction', activation='softmax')(gender_FC)

        age_FC = Dense(128, name='age_FC_1', activation='relu')(x)
        age_out = Dense(8, name='age_prediction', activation='softmax')(age_FC)
        super().__init__(inputs=base.input, outputs=[emotion_out,gender_out, age_out], name='MultitaskVGGFacenet')

    def prep_phase1(self):
        """Freeze layer from input until block_14
        """
        for layer in self.layers[:130]:
            layer.trainable = False
        for layer in self.layers[130:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until blovk_8
        """
        for layer in self.layers[:78]:
            layer.trainable = False
        for layer in self.layers[78:]:
            layer.trainable = True

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
        emotion_predicted = np.argmax(prediction[0], axis=1)
        gender_predicted = np.argmax(prediction[1], axis=1)
        age_predicted = prediction[2].dot(np.arange(0, 101).reshape(101, 1)).flatten()
        return emotion_predicted, gender_predicted, age_predicted

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

class Multitask_two_input_VGGFacenet(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 224
        input_shape = (224,224,3)
        image1_batch = Input(shape=input_shape, name='in_t1')
        image2_batch = Input(shape=input_shape, name='in_t2')
        
        base = VGGFace(include_top=False,model = 'vgg16',weights='vggface',input_shape=(224,224,3))
        last_layer = base.get_layer('pool5').output
        top_layer = Flatten(name='flatten')(last_layer)
        top_layer = Dense(1024,name='common_fc',activation='relu')(top_layer)
     
        inter_model = Model(inputs=base.input,output=top_layer)
        common1_feat=inter_model(image1_batch)
        common2_feat=inter_model(image2_batch)


        emotion_FC = Dense(128, name='emotion_FC_1', activation='relu')(common1_feat)
        emotion_out = Dense(8, name='emotion_prediction', activation='softmax')(emotion_FC)

        gender_FC = Dense(128, name='gender_FC_1', activation='relu')(common2_feat)
        gender_out = Dense(2, name='gender_prediction', activation='softmax')(gender_FC)

        age_FC = Dense(128, name='age_FC_1', activation='relu')(common2_feat)
        age_out = Dense(8, name='age_prediction', activation='softmax')(age_FC)

        super().__init__(inputs=[image1_batch,image2_batch], outputs=[emotion_out,gender_out, age_out], name='Multitask_two_input_VGGFacenet')

    def prep_phase1(self):
        """Freeze layer from input until block_14
        """
        for layer in self.layers[:130]:
            layer.trainable = False
        for layer in self.layers[130:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until blovk_8
        """
        for layer in self.layers[:78]:
            layer.trainable = False
        for layer in self.layers[78:]:
            layer.trainable = True

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
        emotion_predicted = np.argmax(prediction[0], axis=1)
        gender_predicted = np.argmax(prediction[1], axis=1)
        age_predicted = prediction[2].dot(np.arange(0, 101).reshape(101, 1)).flatten()
        return emotion_predicted, gender_predicted, age_predicted

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

class AgeEmotionVGGFacenet(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 224
        base = VGGFace(include_top=False,model = 'vgg16',weights='vggface',input_shape=(224,224,3))
        last_layer = base.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        
        x = Dense(1024,name = 'common_fc',activation='relu')(x)
        
        emotion_FC = Dense(128, name='emotion_FC_1', activation='relu')(x)
        emotion_out = Dense(8, name='emotion_prediction', activation='softmax')(emotion_FC)

        gender_FC = Dense(128, name='gender_FC_1', activation='relu')(x)
        gender_out = Dense(2, name='gender_prediction', activation='softmax')(gender_FC)

        age_FC = Dense(128, name='age_FC_1', activation='relu')(x)
        age_out = Dense(8, name='age_prediction', activation='softmax')(age_FC)
        super().__init__(inputs=base.input, outputs=[emotion_out,age_out], name='AgeEmotionVGGFacenet')

    def prep_phase1(self):
        """Freeze layer from input until block_14
        """
        for layer in self.layers[:130]:
            layer.trainable = False
        for layer in self.layers[130:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until blovk_8
        """
        for layer in self.layers[:78]:
            layer.trainable = False
        for layer in self.layers[78:]:
            layer.trainable = True

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
        emotion_predicted = np.argmax(prediction[0], axis=1)
        gender_predicted = np.argmax(prediction[1], axis=1)
        age_predicted = prediction[2].dot(np.arange(0, 101).reshape(101, 1)).flatten()
        return emotion_predicted, gender_predicted, age_predicted

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

class Multitask_two_input_two_output_VGGFacenet(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 224
        input_shape = (224,224,3)
        image1_batch = Input(shape=input_shape, name='in_t1')
        image2_batch = Input(shape=input_shape, name='in_t2')
        
        base = VGGFace(include_top=False,model = 'vgg16',weights='vggface',input_shape=(224,224,3))
        last_layer = base.get_layer('pool5').output
        top_layer = Flatten(name='flatten')(last_layer)
        top_layer = Dense(1024,name='common_fc',activation='relu')(top_layer)
     
        inter_model = Model(inputs=base.input,output=top_layer)
        common1_feat=inter_model(image1_batch)
        common2_feat=inter_model(image2_batch)


        emotion_FC = Dense(128, name='emotion_FC_1', activation='relu')(common1_feat)
        emotion_out = Dense(7, name='emotion_prediction', activation='softmax')(emotion_FC)

        # gender_FC = Dense(128, name='gender_FC_1', activation='relu')(common2_feat)
        # gender_out = Dense(2, name='gender_prediction', activation='softmax')(gender_FC)

        age_FC = Dense(128, name='age_FC_1', activation='relu')(common2_feat)
        age_out = Dense(70, name='age_prediction', activation='softmax')(age_FC)

        super().__init__(inputs=[image1_batch,image2_batch], outputs=[emotion_out, age_out], name='two_output_VGGFacenet')

    def prep_phase1(self):
        """Freeze layer from input until block_14
        """
        for layer in self.layers[:130]:
            layer.trainable = False
        for layer in self.layers[130:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until blovk_8
        """
        for layer in self.layers[:78]:
            layer.trainable = False
        for layer in self.layers[78:]:
            layer.trainable = True

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
        emotion_predicted = np.argmax(prediction[0], axis=1)
        gender_predicted = np.argmax(prediction[1], axis=1)
        age_predicted = prediction[2].dot(np.arange(0, 101).reshape(101, 1)).flatten()
        return emotion_predicted, gender_predicted, age_predicted

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

if __name__ == '__main__':
    model = AgenderNetMobileNetV2()
    print(model.summary())
    for (i, layer) in enumerate(model.layers):
        print(i, layer.name)
