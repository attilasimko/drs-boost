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

class ResNetModel(BaseModel):
    def get_config(num_opt):
        return {
                "algorithm": "bayes",
                "name": "ResNet",
                "spec": {"maxCombo": num_opt, "objective": "minimize", "metric": "val_loss"},
                "parameters": {
                    "optimizer": {"type": "categorical", "values": ["Adam", "SGD", "RMSprop"]},
                    "learning_rate": {"type": "float", "scalingType": "loguniform", "min": 0.0000001, "max": 0.01},
                    "num_filters": {"type": "integer", "min": 8, "max": 64},
                    "dropout_rate": {"type": "float", "min": 0.0, "max": 0.6},
                    "batch_size": {"type": "discrete", "values": [4, 8]},
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
    
    def res_conv(x, s, filters):
        '''
        here the input size changes''' 
        x_skip = x
        f1, f2 = filters

        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #third block
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        # shortcut 
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add 
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x
    
    def build(experiment, generator):
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("\nBuilding model...")

        x_skip = []
        outputs = []
        inputs = []
        for i in range(len(generator.inputs)):
            inputs.append(Input(shape=generator.in_dims[i][1:]))
        input = Concatenate()(inputs)
        
        x = ZeroPadding2D(padding=(3, 3))(input)

        x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((2, 2))(x)
        
        num_filters = experiment.get_parameter('num_filters')
        dropout_rate = experiment.get_parameter('dropout_rate')

        x = input
        i = 0
        while np.cumprod(np.shape(x)[1:])[-1] > 256:
            x = ResNetModel.encoding_block(x,num_filters*(2**i),dropout_prob=dropout_rate, max_pooling=True)
            i += 1
            if (i > 10):
                raise Exception("Too many layers!")
        
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(12, activation='relu')(x)
        outputs = Dense(len(generator.outputs), activation="softmax")(x)
        # outputs = []
        # start_idx = 0
        # for i in range(len(generator.outputs)):
        #     current_out = out[:, :, :, start_idx:start_idx+generator.out_dims[i][1]]
        #     if (generator.output_types[i] == np.bool):
        #         print(f"Applying sigmoid activation to Output {i}.")
        #         out = Activation('sigmoid')(out)
        #     elif (generator.output_types[i] == "znorm"):
        #         print(f"Applying Z-Normalization activation to Output {i}.")
        #         out = Activation(ResNetModel.znorm)(out)
        #     elif (generator.output_types[i] == "-11_range"):
        #         print(f"Applying [-1, 1] range activation to Output {i}.")
        #         out = Activation(ResNetModel.sct_range)(out)
        #     elif (generator.output_types[i] == "relu"):
        #         print(f"Applying ReLU activation to Output {i}.")
        #         out = Activation('relu')(out)
        #     outputs.append(out)
        
        model = Model(inputs, outputs)
        return model

    def train(model, experiment, gen_train, gen_val, epoch):
        import numpy as np
        import utils.utils_misc as utils_misc
        import time
        
        tic = time.perf_counter()
        train_loss = []
        for i, data in enumerate(gen_train):
            x = data[0]
            y = data[1]
            y = np.stack(y, 1)[:, :, 0, 0, 0] # Erik HC
            loss = model.train_on_batch(x, y)
            train_loss.append(loss)
        toc = time.perf_counter()
        experiment.log_metrics({"epoch_time": toc - tic}, epoch=epoch)
        val_score = utils_misc.evaluate(experiment, model, gen_val, "val")
        return train_loss, val_score
