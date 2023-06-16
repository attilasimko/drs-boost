import os
import re
import abc
import six
import datetime

import numpy as np
from six import with_metaclass

__all__ = ["BaseGenerator",
           "ImageGenerator", "ArrayGenerator",
           "Dicom3DGenerator", "DicomGenerator",
           "Numpy2DGenerator", "Numpy3DGenerator",
           "Dicom3DSaver"]


import os
import re
import abc
import six
import datetime
import tensorflow

def get_activation(image):
    thr = 1e-5
    if ((np.abs(np.mean(image)) < thr) & (np.abs(np.std(image) - 1) < thr)):
        return 'znorm'
    elif ((np.abs(np.min(image) + 1) < thr) & (np.abs(np.max(image) - 1) < thr)):
        return '-11_range'
    elif ((np.abs(np.min(image)) < thr) & (image.dtype != np.bool)):
        return 'relu'
    else:
        return image.dtype

def report_names(data_path):
    file_list = [data_path + '/' + s for s in
                        os.listdir(data_path)]
    print("Use the following field names comma-separated:")
    with np.load(file_list[0], allow_pickle=True) as npzfile:
        for file_name in npzfile.files:
            im = npzfile[file_name]
            print(f"{file_name} ({str(im.dtype)}) - {str(im.shape)}")
        npzfile.close()

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 inputs,
                 outputs,
                 batch_size=32,
                 shuffle=True
                 ):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if data_path is None:
            raise ValueError('The data path is not defined, use the argument "--data" to define the path to the dataset.')

        if not os.path.isdir(data_path):
            raise ValueError(f'The data path ({repr(data_path)}) is not a directory.')

        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s for s in
                          os.listdir(self.data_path)]
        if ~shuffle:
            self.file_list.sort()
        self.on_epoch_end()
        print(f"Found {len(self.file_list)} data files.")
        
        if ((inputs is None) | (outputs is None)):
            print("Both the inputs (--inputs) and outputs (--outputs) must be defined with comma-separated strings.")
            print("The following fields are available:")
            with np.load(self.file_list[0], allow_pickle=True) as npzfile:
                for file_name in npzfile.files:
                    im = npzfile[file_name]
                    print(f"{file_name} ({str(im.dtype)}) - {str(im.shape)}")
                npzfile.close()
            print("Use these fields as inputs and outputs.")
            exit(0)


        self.inputs = inputs.split(',')
        self.outputs = outputs.split(',')
        self.input_types = []
        self.output_types = []
        self.input_activations = []
        self.output_activations = []
        with np.load(self.file_list[0], allow_pickle=True) as npzfile:
            self.out_dims = []
            self.in_dims = []
            for i in range(len(self.inputs)):
                im = npzfile[self.inputs[i]]
                self.in_dims.append((self.batch_size,
                                    *np.shape(im)))
                self.input_types.append(im.dtype)
                self.input_activations.append(get_activation(im))
                print(f"Input {i}: {self.inputs[i]} ({str(im.dtype)}) - {str(im.shape)}")
            for i in range(len(self.outputs)):
                im = npzfile[self.outputs[i]]
                self.out_dims.append((self.batch_size,
                                        *np.shape(im)))
                self.output_types.append(im.dtype)
                self.output_activations.append(get_activation(im))
                print(f"Output {i}: {self.outputs[i]} ({str(im.dtype)}) - {str(im.shape)}")
            npzfile.close()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_list)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        self.temp_ID = [self.file_list[k] for k in indexes]

        # Generate data
        i, o = self.__data_generation(self.temp_ID)
        return i, o

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
    
    #@threadsafe_generator
    def __data_generation(self, temp_list):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        inputs = []
        outputs = []

        for i in range(self.inputs.__len__()):
            inputs.append(np.empty(self.in_dims[i]).astype(self.input_types[i]))

        for i in range(self.outputs.__len__()):
            outputs.append(np.empty(self.out_dims[i]).astype(self.output_types[i]))

        for i, ID in enumerate(temp_list):
            try:
                with np.load(ID, allow_pickle=True) as npzfile:
                    for idx in range(len(self.inputs)):
                        x = npzfile[self.inputs[idx]] \
                            .astype(self.input_types[idx])
                        inputs[idx][i, ] = x

                    for idx in range(len(self.outputs)):
                        x = npzfile[self.outputs[idx]] \
                            .astype(self.output_types[idx])
                        outputs[idx][i, ] = x
                    npzfile.close()
            except:
                raise ValueError(f"Error loading file {ID}")
            
        return inputs, outputs
