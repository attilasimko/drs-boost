def setup_generators(data_path, inputs, outputs, batch_size):
    from data import DataGenerator
    
    print("\nSetting up training generator...")
    dataset_train = DataGenerator(data_path + "training/", inputs, outputs, batch_size=batch_size)
    print("\nSetting up validation generator...")
    dataset_validate = DataGenerator(data_path + "validating/", inputs, outputs, batch_size=batch_size)
    print("\nSetting up testing generator...")
    dataset_test = DataGenerator(data_path + "testing/", inputs, outputs, batch_size=batch_size)

    return dataset_train, dataset_validate, dataset_test

def get_array_names(data_path):
    from data import report_names
    report_names(data_path + "training/")

def get_less_used_gpu(gpus=None, debug=False):
    from torch import cuda
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    if (len(sys_gpus) == 0):
        warn =f"No gpus available"
        return ""
    elif len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated

def get_TF_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_TF_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def get_dataset_path(experiment, task):
    import os
    if os.path.isdir('/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/data/'): # If running on my local machine
        data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/data/'
        experiment.log_parameter("server", "GERTY")
    elif os.path.isdir('/data_m2/lorenzo/data/'): # If running on laplace
        data_path = '/data_m2/lorenzo/data/'
        experiment.log_parameter("server", "laplace")
    elif os.path.isdir('/data/attila/data/'): # If running on gauss
        data_path = '/data/attila/data/'
        experiment.log_parameter("server", "gauss")
    else:
        raise Exception("Unknown server")

    if task == "sct":
        data_path += 'interim/Pelvis_2.1_repo_no_mask/Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10.zip'
    elif task == "transfer":
        data_path += 'interim/brats/brats.zip'
    elif task== "denoise":
        data_path += 'interim/mayo-clinic/'
    else:
        raise Exception("Unknown task")
    
    return data_path
    
def evaluate(experiment, model, gen, eval_type):
    import numpy as np
    from tensorflow.keras.utils import OrderedEnqueuer
    from utils.losses import get_metric
    fn = get_metric(experiment.get_parameter("metric"))
    
    for idx, data in enumerate(gen):
        x = data[0]
        y = data[1]

        if (idx == 0):
            loss_list = []
            for i in range(len(y)):
                loss_list.append([])

        pred = model.predict_on_batch(x)
        for i in range(len(y)):
            loss_list[i].extend([np.mean([fn(y[i], pred[i])])])

    for i in range(len(y)):
        experiment.log_metrics({f"{eval_type}_loss_{i}": np.mean(loss_list[i])})

    return np.mean(loss_list)

def plot_results(experiment, model, gen):
    import numpy as np
    import matplotlib.pyplot as plt
    experiment_name = experiment.get_parameter("name")
    inputs = experiment.get_parameter("inputs").split(',')
    outputs = experiment.get_parameter("outputs").split(',')
    plot_idx = 0
    plot_num = 10
    for i, data in enumerate(gen):
        if (plot_idx <= plot_num):
            if (experiment_name == "erik"):
                if (data[1][0][0] == 1):
                    continue

            plot_idx += 1
            y = data[1]
            x = data[0]
            pred = model.predict_on_batch(x)

            if (experiment_name == "william"): # William HC
                for i in range(len(x)):
                    x[i] = np.expand_dims(x[i][:, 16, :, :], -1)

                for i in range(len(y)):
                    y[i] = np.expand_dims(y[i][:, 16, :, :], -1)
                
                for i in range(len(pred)):
                    pred[i] = np.expand_dims(pred[i][:, 16, :, :], -1)
                  
            x_num = len(x)
            plt.clf()
            for idx in range(x_num):
                plt.subplot(3, x_num, idx + 1)
                plt.imshow(x[idx][0, :, :, 0], cmap='gray')
                plt.title(inputs[idx])
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])

            y_num = len(y)
            for idx in range(y_num):
                plt.subplot(3, y_num, y_num + idx + 1)
                if (experiment_name == "erik"):
                    plt.imshow(np.expand_dims(pred[idx][0:1], -1), vmin=0, vmax=1, cmap='gray')
                else:
                    plt.imshow(pred[idx][0, :, :, 0], cmap='gray')
                plt.colorbar()
                plt.title(outputs[idx])
                plt.xticks([])
                plt.yticks([])

            for idx in range(y_num):
                plt.subplot(3, y_num, y_num + y_num + idx + 1)
                if (experiment_name == "erik"):
                    plt.imshow(y[idx][0, ...], vmin=0, vmax=1, cmap='gray')
                else:
                    plt.imshow(y[idx][0, :, :, 0], cmap='gray')
                plt.colorbar()
                plt.title(outputs[idx])
                plt.xticks([])
                plt.yticks([])
            experiment.log_figure(figure=plt, figure_name="results_" + str(i), overwrite=True)
            plt.close('all')
    
def clear_memory():
    import gc
    from keras import backend as K

    K.clear_session()
    gc.collect()

def memory_check(experiment, model):
    # import nvidia_smi
    # import numpy as np

    # nvidia_smi.nvmlInit()
    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # available_memory = np.round(info.total / (1024.0 ** 3), 3) # We could just hard-set this to 10GB. That's the limit on the smallest GPUs we have.
    # nvidia_smi.nvmlShutdown()

    # required_memory = get_TF_memory_usage(experiment.get_parameter("batch_size"), model) 
    # experiment.log_parameter("reqmemory", required_memory)

    # if required_memory > available_memory:
    #     print(f"ERROR: Not enough memory. Required: {required_memory} GB, Limit: {available_memory} GB")
    #     return False
    return True

def export_weights_to_hero(model, experiment, save_path, name):
    from tensorflow import function, TensorSpec
    from tensorflow import io
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    import os

    os.mkdir(save_path + name)
    try:
        full_model = function(lambda x: model(x)) 
        full_model = full_model.get_concrete_function([TensorSpec(x.shape, x.dtype) for x in model.inputs])
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=save_path + name,
                        name=f'HERO_version.pb',
                        as_text=False)
        experiment.log_model("hero_model", save_path + name)
    except Exception as e:
        print("ERROR: Could not export HERO model. Message:" + str(e))

    return


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        import time
        self._start_time = time.perf_counter()

    def lap(self):
        import time
        current_time = time.perf_counter()
        elapsed_time = current_time - self._start_time
        self._start_time = current_time
        return elapsed_time
    
def prune_config(config, name):
    if (name == "gerd"):
        config["parameters"]["num_filters"] = 20
        config["parameters"]["optimizer"] = "Adam"
        config["parameters"]["batch_size"] = 8
        print("Parameters pruned: num_filters = 20, optimizer = Adam, batch_size = 8")
    if (name == "william"):
        config["parameters"]["optimizer"] = "Adam"
        config["parameters"]["learning_rate"] = {"type": "float", "scalingType": "loguniform", "min": 0.000004, "max": 0.01}
        print("Parameters pruned: optimizer = Adam, min. learning rate = 0.000004")

    return config

def gpu_growth():
    import tensorflow
    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tensorflow.compat.v1.Session(config=config)