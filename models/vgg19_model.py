from .base_model import BaseModel
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

class VGG19Model(BaseModel):
    def get_config(num_opt, algorithm):
        return {
                "algorithm": algorithm,
                "name": "VGG19",
                "spec": {"maxCombo": 0, 'retryLimit': num_opt, "objective": "minimize", "metric": "val_loss"},
                "parameters": {
                    "optimizer": {"type": "categorical", "values": ["Adam", "SGD", "RMSprop"]},
                    "learning_rate": {"type": "float", "scalingType": "loguniform", "min": 0.0000001, "max": 0.01},
                    "dropout_rate": {"type": "float", "min": 0.0, "max": 0.6},
                    "batch_size": {"type": "discrete", "values": [4, 8, 16, 32]},
                },
                "trials": 1,
            }
    
    def build(experiment, generator):
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("\nBuilding model...")

        outputs = []
        inputs = []
        for i in range(len(generator.inputs)):
            inputs.append(Input(shape=generator.in_dims[i][1:] + (1,)))
        x = Concatenate()(inputs)
        
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        )
        vgg = Model(inputs=[vgg.layers[0].input], outputs = vgg.get_layer("block5_conv4").output)
        x = vgg(x)
        x = Flatten()(x)
        x = Activation("relu")(x)
        # x = tf.math.square(x)
        # x = tf.reduce_mean(x, (1, 2, 3))

        model = Model(inputs=inputs, outputs=x)

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
