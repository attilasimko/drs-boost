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

class UNetModel(BaseModel):
    def get_config(num_opt, algorithm):
        return {
                "algorithm": algorithm,
                "name": "UNet",
                "spec": {"maxCombo": num_opt, 'retryLimit': num_opt, "objective": "minimize", "metric": "val_loss"},
                "parameters": {
                    "optimizer": {"type": "categorical", "values": ["Adam", "SGD", "RMSprop"]},
                    "learning_rate": {"type": "float", "scalingType": "loguniform", "min": 0.0000001, "max": 0.01},
                    "num_filters": {"type": "integer", "min": 8, "max": 64},
                    "dropout_rate": {"type": "float", "min": 0.0, "max": 0.6},
                    "batch_size": {"type": "discrete", "values": [4, 8, 16]},
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

        kernel_size = 3
        x_skip = []
        outputs = []
        inputs = []
        for i in range(len(generator.inputs)):
            inputs.append(Input(shape=generator.in_dims[i][1:]))
        input = Concatenate()(inputs)

        for i in range(len(generator.outputs)):
            outputs.append(Input(shape=generator.out_dims[i][1:]))
        output = Concatenate()(outputs)
        
        batchNorm = True
        num_filters = experiment.get_parameter('num_filters')
        dropout_rate = experiment.get_parameter('dropout_rate')

        x_skip.insert(0, input)
        depth = int(np.log2(input.shape[1] / 16) + 1)


        x = input
        for i in range(depth):
            x = Conv2D(3, kernel_size, activation = 'relu', padding = 'same', kernel_initializer="he_normal")(x)
            x = Conv2D((i+1) * num_filters, kernel_size, activation = 'relu', padding = 'same', kernel_initializer="he_normal")(x)
            if (batchNorm):
                x = BatchNormalization()(x)
            x_skip.insert(0, x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        
        for i in range(len(x_skip)):
            if (i < len(x_skip) - 1):
                x = UpSampling2D(size=(2, 2))(x)
            x = Concatenate(axis=-1)([x,x_skip[i]])
            x = Conv2D((len(x_skip)-i+1) * num_filters, kernel_size, activation = 'relu', padding = 'same', kernel_initializer="he_normal")(x)
            x = Conv2D((len(x_skip)-i+1) * num_filters*2, kernel_size, activation = 'relu', padding = 'same', kernel_initializer="he_normal")(x)
            if (i < len(x_skip) - 1):
                x = BatchNormalization()(x)
        x = Conv2D(output.shape[-1], 3, activation = 'relu', padding = 'same', kernel_initializer="he_normal")(x)
        out = Conv2D(output.shape[-1], 1, activation = None, padding = 'same', kernel_initializer="he_normal")(x)
        
        use_softmax = False
        if (np.all([out_type == np.bool_ for out_type in generator.output_types])):
            out = Activation('softmax')(out)
            use_softmax = True
        
        if ((use_softmax == False) & (experiment.get_parameter("name") == "gerd")):
            print("Using softmax activation for Gerd project.")
            out = Activation('softmax')(out)
            use_softmax = True

        if ((use_softmax == False) & ((experiment.get_parameter("name") == "erik"))):
            print("Using softmax activation for Erik project.")
            out = Activation('softmax')(out)
            use_softmax = True

        outputs = []
        start_idx = 0
        for i in range(len(generator.outputs)):
            current_out = out[:, :, :, start_idx:start_idx+generator.out_dims[i][3]]
            start_idx += generator.out_dims[i][3]
            if (use_softmax):
                out_i = current_out
            elif (generator.output_types[i] == np.bool_):
                print(f"Applying sigmoid activation to Output {i}.")
                out_i = Activation('sigmoid')(current_out)
            elif (generator.output_types[i] == "znorm"):
                print(f"Applying Z-Normalization activation to Output {i}.")
                out_i = Activation(UNetModel.znorm)(current_out)
            elif (generator.output_types[i] == "-11_range"):
                print(f"Applying [-1, 1] range activation to Output {i}.")
                out_i = Activation(UNetModel.sct_range)(current_out)
            elif (generator.output_types[i] == "relu"):
                print(f"Applying ReLU activation to Output {i}.")
                out_i = Activation('relu')(current_out)
            else:
                out_i = current_out
            outputs.append(out_i)
        
        
        if (experiment.get_parameter("name") == "erik"):
            model = Model(inputs, out)
        else:
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
            y = np.concatenate(y, axis=-1)
            loss = model.train_on_batch(x, y)
            train_loss.append(loss)
        toc = time.perf_counter()
        experiment.log_metrics({"epoch_time": toc - tic}, epoch=epoch)
        val_score = utils_misc.evaluate(experiment, model, gen_val, "val")
        return train_loss, val_score