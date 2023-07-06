from . base_model import BaseModel
import math
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout,\
Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add , Concatenate, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import softmax
import math
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda, UpSampling2D, Conv2DTranspose, concatenate
import tensorflow
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, Lambda, BatchNormalization, Dropout

class UNet3DModel(BaseModel):
    def get_config(num_opt):
        return {
                "algorithm": "bayes",
                "name": "UNet3D",
                "spec": {"maxCombo": 0, 'retryLimit': num_opt, "objective": "minimize", "metric": "val_loss"},
                "parameters": {
                    "optimizer": {"type": "categorical", "values": ["Adam", "SGD", "RMSprop"]},
                    "learning_rate": {"type": "float", "scalingType": "loguniform", "min": 0.0000001, "max": 0.01},
                    "num_filters": {"type": "integer", "min": 8, "max": 24},
                    "dropout_rate": {"type": "float", "min": 0.0, "max": 0.6},
                    "batch_size": {"type": "discrete", "values": [1]},
                    "batch_normalization": {"type": "categorical", "values": ["True", "False"]}
                },
                "trials": 1,
            }
    
    def sct_range(x):
        import tensorflow
        from tensorflow.keras import backend as K
        x = tensorflow.where(K.greater_equal(x, -1), x, -1 * K.ones_like(x))
        x = tensorflow.where(K.less_equal(x, 1), x, 1 * K.ones_like(x))
        return x

    def znorm(x):
        import tensorflow
        import tensorflow.keras.backend as K
        t_mean = K.mean(x, axis=(1, 2, 3))
        t_std = K.std(x, axis=(1, 2, 3))
        return tensorflow.math.divide_no_nan(x - t_mean[:, None, None, None], t_std[:, None, None, None])
    
    # downsampling, analysis path
    def downLayer(inputLayer, filterSize, i, bn=False):
        conv = Conv3D(filterSize, (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_1')(inputLayer)
        if bn:
            conv = BatchNormalization()(conv)
        conv = Conv3D(filterSize * 2, (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)
        pool = MaxPooling3D(pool_size=(1, 2, 2))(conv)
        return pool, conv
    
    # upsampling, synthesis path
    def upLayer(inputLayer, concatLayer, filterSize, i, bn=False, do= False):
        up = Conv3DTranspose(filterSize, (2, 2, 2), strides=(1, 2, 2), activation='relu', padding='same',  name='up'+str(i))(inputLayer)
       # print( concatLayer.shape)
        up = concatenate([up, concatLayer])
        conv = Conv3D(int(filterSize/2), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_1')(up)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed = 3, name='Dropout_' + str(i))(conv)
        conv = Conv3D(int(filterSize/2), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)
        return conv
    
    def build(experiment, generator):
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("\nBuilding model...")
        bn = experiment.get_parameter("batch_normalization") == "True"
        do = float(experiment.get_parameter("dropout_rate"))

        inputs = []
        for i in range(len(generator.inputs)):
            inputs.append(Input(shape=tuple([item for sublist in [np.array(generator.in_dims[i][1:]), [1]] for item in sublist])))
        input = Concatenate()(inputs)
        
        # self, nrInputChannels=1, learningRate=5e-5, bn = True, do = False, opti = Adam, n_imgs=1):
        sfs = int(experiment.get_parameter("num_filters")) # start filter size
        
        conv1, conv1_b_m = UNet3DModel.downLayer(input, sfs, 1, bn)
        conv2, conv2_b_m = UNet3DModel.downLayer(conv1, sfs*2, 2, bn)
        conv3 = Conv3D(sfs*4, (3, 3, 3), activation='relu', padding='same', name='conv' + str(3) + '_1')(conv2)
        if bn:
            conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(sfs * 8, (3, 3, 3), activation='relu', padding='same',  name='conv' + str(3) + '_2')(conv3)
        if bn:
            conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        #conv3, conv3_b_m = downLayer(conv2, sfs*4, 3, bn)
        conv4 = Conv3D(sfs*16 , (3, 3, 3), activation='relu', padding='same',  name='conv4_1')(pool3)
        if bn:
            conv4 = BatchNormalization()(conv4)
        if do:
            conv4= Dropout(0.5, seed = 4, name='Dropout_' + str(4))(conv4)
        conv4 = Conv3D(sfs*16 , (3, 3, 3), activation='relu', padding='same',  name='conv4_2')(conv4)
        if bn:
            conv4 = BatchNormalization()(conv4)
        #conv5 = upLayer(conv4, conv3_b_m, sfs*16, 5, bn, do)
        up1 = Conv3DTranspose(sfs*16, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same', name='up'+str(5))(conv4)
        up1 = concatenate([up1, conv3])
        conv5 = Conv3D(int(sfs*8), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(5)+'_1')(up1)
        if bn:
            conv5 = BatchNormalization()(conv5)
        if do:
            conv5 = Dropout(0.5, seed = 5, name='Dropout_' + str(5))(conv5)
        conv5 = Conv3D(int(sfs*8), (3, 3, 3), activation='relu', padding='same', name='conv'+str(5)+'_2')(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)
        conv6 = UNet3DModel.upLayer(conv5, conv2_b_m, sfs*8, 6, bn, do)
        conv7 = UNet3DModel.upLayer(conv6, conv1_b_m, sfs*4, 7, bn, do)
        conv_out = Conv3D(len(generator.outputs), (1, 1, 1), activation='softmax', name='conv_final_softmax')(conv7)

        outputs = []
        for i in range(len(generator.outputs)):
            outputs.append(conv_out[:, :, :, :, i])
            
        model = Model(inputs=[inputs], outputs=outputs)

        return model
    
    def train(model, experiment, gen_train, gen_val, epoch):
        import numpy as np
        import utils.utils_misc as utils_misc
        import time
        tf.compat.v1.disable_eager_execution()

        timer = utils_misc.Timer()
        train_loss = []
        tic_dl = 0
        tic_train = 0
        timer.start()
        for i in range(len(gen_train)):
            data = gen_train[i]
            x = data[0]
            y = data[1]
            tic_dl += timer.lap()
            loss = model.train_on_batch(x, y)
            train_loss.append(loss)
            tic_train += timer.lap()
        val_score = utils_misc.evaluate(experiment, model, gen_val, "val")
        tic_val = timer.lap()
        experiment.log_metrics({"dl_time": tic_dl}, epoch=epoch)
        experiment.log_metrics({"train_time": tic_train}, epoch=epoch)
        experiment.log_metrics({"val_time": tic_val}, epoch=epoch)

        return train_loss, val_score