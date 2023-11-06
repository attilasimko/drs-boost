from . base_model import BaseModel
import math
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, ZeroPadding3D, Dropout,\
Flatten, BatchNormalization, AveragePooling3D, Dense, Activation, Add , Concatenate, add, LeakyReLU
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
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda, UpSampling2D, Conv3DTranspose, concatenate

class ResNet3DModel(BaseModel):
    def get_config(num_opt, algorithm):
        return {
                "algorithm": algorithm,
                "name": "ResNet3D",
                "spec": {"maxCombo": 0, 'retryLimit': num_opt, "objective": "minimize", "metric": "val_loss"},
                "parameters": {
                    "optimizer": {"type": "categorical", "values": ["Adam", "SGD", "RMSprop"]},
                    "learning_rate": {"type": "float", "scalingType": "loguniform", "min": 0.0000001, "max": 0.1},
                    "num_filters": {"type": "integer", "min": 1, "max": 16},
                    "kernel_size": {"type": "discrete", "values": [1, 3, 5, 7]},
                    "pool_size": {"type": "discrete", "values": [1, 2]},
                    "depth": {"type": "integer", "min": 0, "max": 4},
                    "flat_depth": {"type": "integer", "min": 1, "max": 4},
                    "dropout_rate": {"type": "float", "min": 0.0, "max": 0.6},
                    "batch_size": {"type": "discrete", "values": [1, 2, 4]},
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
    
    def build(experiment, generator):
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("\nBuilding model...")

        outputs = []
        inputs = []
        for i in range(len(generator.inputs)):
            inputs.append(Input(shape=generator.in_dims[i][1:]))
        x = Concatenate()(inputs)
        
        num_filters = len(generator.outputs) * experiment.get_parameter('num_filters')
        depth = experiment.get_parameter('depth')
        flat_depth = experiment.get_parameter('flat_depth')
        dropout_rate = experiment.get_parameter('dropout_rate')
        kernel_size = experiment.get_parameter('kernel_size')
        pool_size = experiment.get_parameter('pool_size')
        if (len(generator.outputs) > 1):
            output_channels = len(generator.outputs)
        else:
            output_channels = generator.out_dims[0][1]

        x = tf.expand_dims(x, axis=-1)
        for i in range(depth):
            x = Conv3D(int((i + 1) * num_filters), kernel_size=(kernel_size, kernel_size, kernel_size), padding="same", activation="relu")(x)
            x = Dropout(dropout_rate)(x)
            x = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(x)
        x = Flatten()(x)

        for i in range(flat_depth):
            x = Dense(int(np.max([output_channels, (x.shape[-1] / 2)])), activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            
        x = Dense(output_channels, activation="softmax")(x)

        outputs = []
        if (len(generator.outputs) > 1):
            for i in range(len(generator.outputs)):
                outputs.append(x[:, i:i+1])
        else:
            outputs.append(x)

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
            loss = model.train_on_batch(x, y)
            train_loss.append(loss)
        toc = time.perf_counter()
        experiment.log_metrics({"epoch_time": toc - tic}, epoch=epoch)
        val_score = utils_misc.evaluate(experiment, model, gen_val, "val")
        return train_loss, val_score
