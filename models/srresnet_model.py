from . base_model import BaseModel
import math

class SRResNetModel(BaseModel):
    config = {
                "algorithm": "bayes",
                "name": "SRResNet",
                "spec": {"maxCombo": 50, "objective": "minimize", "metric": "val_loss"},
                "parameters": {
                    "optimizer": {"type": "categorical", "values": ["Adam", "SGD", "RMSprop"]},
                    "learning_rate": {"type": "float", "scalingType": "loguniform", "min": 0.0000001, "max": 0.0005},
                    "num_filters": {"type": "integer", "min": 32, "max": 64},
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

    def build_model(experiment, task, dropout_rate):
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
        from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda, UpSampling2D, Conv2DTranspose
        tf.config.experimental.enable_tensor_float_32_execution(False)

        input = Input(shape=(256, 256, 1))
        num_filters = experiment.get_parameter('num_filters')
            
        x = Conv2D(4, kernel_size=3, padding='same')(input)
        x = x_1 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = x_2 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = x_3 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = x_4 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = x_5 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = x_6 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)


        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Concatenate()([x, x_6])
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_5])
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_4])
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_3])
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_2])
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_1])
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)

        output = Conv2D(1, kernel_size=1, padding='same')(x)
        if ((task == "sct") | (task == "denoise")):
            output = Activation(SRResNetModel.sct_range)(output)
        elif (task == "transfer"):
            output = Activation(SRResNetModel.znorm)(output)

        model = Model(input, output)
        if (experiment.get_parameter("optimizer") == "Adam"): # "Adam", "SGD", "RMSprop"
            model.compile(optimizer=Adam(experiment.get_parameter("learning_rate")), loss='mse', metrics=['mse'])
        elif (experiment.get_parameter("optimizer") == "SGD"):
            model.compile(optimizer=SGD(experiment.get_parameter("learning_rate")), loss='mse', metrics=['mse'])
        elif (experiment.get_parameter("optimizer") == "RMSprop"):
            model.compile(optimizer=SGD(experiment.get_parameter("learning_rate")), loss='mse', metrics=['mse'])
            
        return model

    def train(experiment, model, task, gen_train, gen_val):
        from tensorflow.keras.utils import OrderedEnqueuer
        from tensorflow.config.experimental import get_device_details, get_memory_info
        from tensorflow.config import list_physical_devices
        import numpy as np
        import utils.utils_misc as utils_misc
        import time
        
        if (utils_misc.memory_check(experiment, model) == False):
            val_score = utils_misc.evaluate(experiment, model, gen_val, "val", task)
            return
        
        min_loss = np.inf
        patience = 0
        patience_thr = 20
        for epoch in range(experiment.get_parameter("epochs")):
            tic = time.perf_counter()
            train_loss = []
            for i, data in enumerate(gen_train):
                y = np.expand_dims(data[0].numpy(), 3)
                x = np.expand_dims(data[1].numpy(), 3)
                loss, _ = model.train_on_batch(x, y)
                train_loss.append(loss)

            toc = time.perf_counter()
            experiment.log_metrics({"training_loss": np.mean(train_loss),
                                    "epoch_time": toc - tic}, epoch=epoch)

            val_score = utils_misc.evaluate(experiment, model, gen_val, "val", task)
            if (val_score < min_loss):
                patience = 0
                min_loss = val_score
                print("Validation score %s", val_score)
            else:
                patience += 1
                if patience > patience_thr:
                    print("Early stopping")
                    break

