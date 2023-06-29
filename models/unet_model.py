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
    def get_config(num_opt):
        return {
                "algorithm": "bayes",
                "name": "UNet",
                "spec": {"maxCombo": 0, 'retryLimit': num_opt, "objective": "minimize", "metric": "val_loss"},
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
    
    def encoding_block(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
        """
        This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
        Dropout can be added for regularization to prevent overfitting. 
        The block returns the activation values for next layer along with a skip connection which will be used in the decoder
        """
        # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow 
        # Proper initialization prevents from the problem of exploding and vanishing gradients 
        # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
        conv = Conv2D(n_filters, 
                    3,   # Kernel size   
                    activation='relu',
                    padding='same',
                    kernel_initializer='HeNormal')(inputs)
        conv = Conv2D(n_filters, 
                    3,   # Kernel size
                    activation='relu',
                    padding='same',
                    kernel_initializer='HeNormal')(conv)
        
        # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
        conv = BatchNormalization()(conv, training=False)

        # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
        if dropout_prob > 0:     
            conv = tf.keras.layers.Dropout(dropout_prob)(conv)

        # Pooling reduces the size of the image while keeping the number of channels same
        # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
        # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
        if max_pooling:
            next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
        else:
            next_layer = conv

        # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
        skip_connection = conv
        
        return next_layer, skip_connection
    
    def decoding_block(prev_layer_input, skip_layer_input, n_filters=32):
        """
        Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
        merges the result with skip layer results from encoder block
        Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
        The function returns the decoded layer output
        """
        # Start with a transpose convolution layer to first increase the size of the image
        up = Conv2DTranspose(
                    n_filters,
                    (3,3),    # Kernel size
                    strides=(2,2),
                    padding='same')(prev_layer_input)

        # Merge the skip connection from previous block to prevent information loss
        merge = concatenate([up, skip_layer_input], axis=3)
        
        # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
        # The parameters for the function are similar to encoder
        conv = Conv2D(n_filters, 
                    3,     # Kernel size
                    activation='relu',
                    padding='same',
                    kernel_initializer='HeNormal')(merge)
        conv = Conv2D(n_filters,
                    3,   # Kernel size
                    activation='relu',
                    padding='same',
                    kernel_initializer='HeNormal')(conv)
        return conv
    
    def build(experiment, generator):
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("\nBuilding model...")

        x_skip = []
        outputs = []
        inputs = []
        for i in range(len(generator.inputs)):
            inputs.append(Input(shape=generator.in_dims[i][1:]))
        input = Concatenate()(inputs)

        for i in range(len(generator.outputs)):
            outputs.append(Input(shape=generator.out_dims[i][1:]))
        output = Concatenate()(outputs)
        
        num_filters = experiment.get_parameter('num_filters')
        dropout_rate = experiment.get_parameter('dropout_rate')
        
        for i in range(int(np.log2(input.shape[1] / 16) + 1)):
            if (i == 0):
                x_skip.insert(0, UNetModel.encoding_block(input, num_filters,dropout_prob=0, max_pooling=True))
            elif (i == int(np.log2(input.shape[1] / 16))):
                x_skip.insert(0, UNetModel.encoding_block(x_skip[0][0], num_filters*(2**i), dropout_prob=dropout_rate, max_pooling=False)) 
            else:
                x_skip.insert(0, UNetModel.encoding_block(x_skip[0][0],num_filters*(2**i),dropout_prob=0, max_pooling=True))
        
        ublock = x_skip[0][0]
        for i in range(int(np.log2(output.shape[1] / x_skip[0][1].shape[1]))):
            ublock = UNetModel.decoding_block(ublock, x_skip[i+1][1], ublock.shape[-1] / (2**i))
        out = Conv2D(output.shape[-1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock)
        out = Conv2D(output.shape[-1], 1, padding='same', kernel_initializer='he_normal')(out)
        
        if (experiment.get_parameter('name') == "gerd"):
            out = Activation('softmax')(out)

        outputs = []
        start_idx = 0
        for i in range(len(generator.outputs)):
            current_out = out[:, :, :, start_idx:start_idx+generator.out_dims[i][1]]
            if (generator.output_types[i] == np.bool):
                print(f"Applying sigmoid activation to Output {i}.")
                out = Activation('sigmoid')(current_out)
            elif (generator.output_types[i] == "znorm"):
                print(f"Applying Z-Normalization activation to Output {i}.")
                out = Activation(UNetModel.znorm)(current_out)
            elif (generator.output_types[i] == "-11_range"):
                print(f"Applying [-1, 1] range activation to Output {i}.")
                out = Activation(UNetModel.sct_range)(current_out)
            elif (generator.output_types[i] == "relu"):
                print(f"Applying ReLU activation to Output {i}.")
                out = Activation('relu')(current_out)
            outputs.append(out)
        
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
