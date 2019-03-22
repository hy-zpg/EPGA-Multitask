import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.mobilenetv2 import MobileNetV2
from keras.utils import plot_model


class AgenderNetMobileNetV2(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 96
        # base = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights=os.path.dirname(
        #     __file__)+'/weight/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5')
        # top_layer = GlobalAveragePooling2D()(base.output)
        base = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
        top_layer = GlobalAveragePooling2D()(base.output)
        
        gender_FC = Dense(128, activation='relu', name='gender_FC')(top_layer)
        gender_layer = Dense(2, activation='softmax', name='gender_prediction')(gender_FC)

        age_FC = Dense(128, activation='relu', name='age_FC')(top_layer)
        age_layer = Dense(8, activation='softmax', name='age_prediction')(age_FC)
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



class EmotionNetMobileNetV2(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 96
        base = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
        top_layer = GlobalAveragePooling2D()(base.output)
        emotion_FC = Dense(128, activation='relu', name='emotion')(top_layer)
        emotion_layer = Dense(7, activation='softmax', name='emotion_prediction')(emotion_FC)
        super().__init__(inputs=base.input, outputs=emotion_layer, name='EmotionNetMobileNetV2')

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

class MultitaskMobileNetV2(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 96
        # base = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights=os.path.dirname(
        #     __file__)+'/weight/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5')
        # top_layer = GlobalAveragePooling2D()(base.output)
        base = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
        top_layer = GlobalAveragePooling2D()(base.output)

        gender_FC = Dense(128, name='gender_FC', activation='relu')(top_layer)
        gender_layer = Dense(2, activation='softmax', name='gender_prediction')(gender_FC)

        age_FC = Dense(128, name='age_FC', activation='relu')(top_layer)
        age_layer = Dense(8, activation='softmax', name='age_prediction')(age_FC)

        emotion_FC = Dense(128, name='emotion_FC', activation='relu')(top_layer)
        emotion_layer = Dense(8, activation='softmax', name='emotion_prediction')(emotion_FC)
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


class MTFLMobileNetV2(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 224
        base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        top_layer = GlobalAveragePooling2D()(base.output)

        # landmark_FC = Dense(128,name='landmark_FC', activation="relu")(top_layer)
        # landmark_out = Dense(10,name='landmark_prediction',activation='relu')(landmark_FC)

        gender_FC = Dense(128,name='gender_FC', activation="relu")(top_layer)
        gender_out = Dense(2,activation='softmax', name='gender_prediction')(gender_FC)

        smile_FC = Dense(128,name='smile_FC', activation="relu")(top_layer)
        smile_out = Dense(2,name='smile_prediction',activation='softmax')(smile_FC)

        pose_FC = Dense(128,name='pose_FC', activation="relu")(top_layer)
        pose_out = Dense(5,name='pose_prediction',activation='softmax')(pose_FC)

        super().__init__(inputs=base.input, outputs=[pose_out,gender_out,smile_out], name='MTFLMobileNetV2')

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





if __name__ == '__main__':
    model = AgenderNetMobileNetV2()
    print(model.summary())
    for (i, layer) in enumerate(model.layers):
        print(i, layer.name)

class Multitask_two_input_MobileNetV2(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 96
        
        input_shape = (96,96,3)
        image1_batch = Input(shape=input_shape, name='in_t1')
        image2_batch = Input(shape=input_shape, name='in_t2')
        
        base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        top_layer = GlobalAveragePooling2D()(base.output)
     
        inter_model = Model(inputs=base.input,output=top_layer)
        common1_feat=inter_model(image1_batch)
        common2_feat=inter_model(image2_batch)

        emotion_FC = Dense(128, name='emotion_FC', activation='relu')(common1_feat)
        emotion_out = Dense(8, name='emotion_prediction', activation='softmax')(emotion_FC)

        gender_FC = Dense(128, name='gender_FC', activation='relu')(common2_feat)
        gender_out = Dense(2, name='gender_prediction', activation='softmax')(gender_FC)

        age_FC = Dense(128, name='age_FC', activation='relu')(common2_feat)
        age_out = Dense(8, name='age_prediction', activation='softmax')(age_FC)
        super().__init__(inputs=[image1_batch,image2_batch], outputs=[emotion_out,gender_out,age_out], name='multitask_two_input_MobileNetV2')


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
