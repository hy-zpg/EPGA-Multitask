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
from keras.regularizers import l2
import numpy as np
import os
from keras.applications.mobilenetv2 import MobileNetV2
from keras.utils import plot_model


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

class AgenderNetmin_XCEPTION(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 64
        # base = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights=os.path.dirname(
        #     __file__)+'/weight/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5')
        # top_layer = GlobalAveragePooling2D()(base.output)
        base = mini_XCEPTION(input_shape=(self.input_size,self.input_size,1), l2_regularization=0.01)
        last_layer = base.output

        gender_layer = Conv2D(2, (3, 3),padding='same')(last_layer)
        gender_layer = GlobalAveragePooling2D()(gender_layer)
        gender_layer = Activation('softmax', name='gender_prediction')(gender_layer)

        age_layer = Conv2D(8, (3, 3),padding='same')(last_layer)
        age_layer = GlobalAveragePooling2D()(age_layer)
        age_layer = Activation('softmax', name='age_prediction')(age_layer)
    
        super().__init__(inputs=base.input, outputs=[gender_layer, age_layer], name='AgenderNetMobileNetV2')

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



class EmotionNetmin_XCEPTION(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 64
        base = mini_XCEPTION(input_shape=(self.input_size,self.input_size,1), l2_regularization=0.01)
        last_layer = base.output

        emotion_layer = Conv2D(8, (3, 3),padding='same')(last_layer)
        emotion_layer = GlobalAveragePooling2D()(emotion_layer)
        emotion_layer = Activation('softmax', name='emotion_prediction')(emotion_layer)
        super().__init__(inputs=base.input, outputs=emotion_layer, name='EmotionNetmin_XCEPTION')

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

class Multitaskmin_XCEPTION(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 64
        base = mini_XCEPTION(input_shape=(self.input_size,self.input_size,3), l2_regularization=0.01)
        last_layer = base.output

        emotion_layer = Conv2D(8, (3, 3),padding='same')(last_layer)
        emotion_layer = GlobalAveragePooling2D()(emotion_layer)
        emotion_layer = Activation('softmax', name='emotion_prediction')(emotion_layer)

        gender_layer = Conv2D(2, (3, 3),padding='same')(last_layer)
        gender_layer = GlobalAveragePooling2D()(gender_layer)
        gender_layer = Activation('softmax', name='gender_prediction')(gender_layer)

        age_layer = Conv2D(8, (3, 3),padding='same')(last_layer)
        age_layer = GlobalAveragePooling2D()(age_layer)
        age_layer = Activation('softmax', name='age_prediction')(age_layer)

        super().__init__(inputs=base.input, outputs=[emotion_layer,gender_layer, age_layer], name='AgenderNetMobileNetV2')

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

class Multitask_two_input_two_output_min_XCEPTION(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 64
        input_shape = (64,64,1)
        image1_batch = Input(shape=input_shape, name='in_t1')
        image2_batch = Input(shape=input_shape, name='in_t2')
        
     
        inter_model = mini_XCEPTION(input_shape=(self.input_size,self.input_size,1), l2_regularization=0.01)
        common1_feat=inter_model(image1_batch)
        common2_feat=inter_model(image2_batch)

        emotion_layer = Conv2D(8, (3, 3),padding='same')(common1_feat)
        emotion_layer = GlobalAveragePooling2D()(emotion_layer)
        emotion_layer = Activation('softmax', name='emotion_prediction')(emotion_layer)

        gender_layer = Conv2D(2, (3, 3),padding='same')(common2_feat)
        gender_layer = GlobalAveragePooling2D()(gender_layer)
        gender_layer = Activation('softmax', name='gender_prediction')(gender_layer)

        age_layer = Conv2D(8, (3, 3),padding='same')(common2_feat)
        age_layer = GlobalAveragePooling2D()(age_layer)
        age_layer = Activation('softmax', name='age_prediction')(age_layer)

        super().__init__(inputs=[image1_batch,image2_batch], outputs=[emotion_layer,gender_layer, age_layer], name='AgenderNetMobileNetV2')
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
class AgeNetmin_XCEPTION(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 64
        base = mini_XCEPTION(input_shape=(self.input_size,self.input_size,3), l2_regularization=0.01)
        last_layer = base.output

        age_layer = Conv2D(70, (3, 3),padding='same')(last_layer)
        age_layer = GlobalAveragePooling2D()(age_layer)
        age_layer = Activation('softmax', name='age_prediction')(age_layer)
        super().__init__(inputs=base.input, outputs=age_layer, name='ageNetMobileNetV2')
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
