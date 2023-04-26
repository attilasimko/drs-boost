import argparse
parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--name", required=True, help="Name your experiment to help with online tracking.")
parser.add_argument("--data", default=None, help="Name your experiment to help with online tracking.")
parser.add_argument("--inputs", default=None, help="The field(s) to use as input(s) to the model. Multiple fields should be comma-separated.")
parser.add_argument("--outputs", default=None, help="The field(s) to use as target(s) of the model. Multiple fields should be comma-separated.")
parser.add_argument("--models", default="unet", help="The model architectures that will be used. Multiple models should be comma-separated. Possible values: unet, srresnet.")
parser.add_argument("--num_epochs", default=None, help="Set the maximum number of epochs. Default is infinity.")
parser.add_argument("--num_opt", default=10, help="Set the number of optimization steps. Default is 10.")
parser.add_argument("--batch_size", default=1, help="Batch size to use during training. Default is 1.")
parser.add_argument("--gpu", default=0, help="GPU to use for training on the cluster. Default is 0.")
parser.add_argument("--log_comet", default=True, help="Log online on comet.ml. Default is True.")
args = parser.parse_args()
name = args.name
log_comet = args.log_comet
num_opt = int(args.num_opt)
data_path = args.data
input_array = args.inputs
output_array = args.outputs
model_array = args.models.split(',')

import comet_ml
comet_ml.init(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name=name, workspace="attilasimko")
print(f"Started experiment with name: {name}")

import os
import shutil
import utils.utils_misc as utils_misc
if args.gpu is not None:
    gpu = int(args.gpu)
else:
    gpu = utils_misc.get_less_used_gpu()

if (os.path.isdir(data_path + "temp/")):
    shutil.rmtree(data_path + "temp/")
os.mkdir(data_path + "temp/")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print("Training will be done on GPU: ", gpu)

import os
import time
import numpy as np
from utils.utils_misc import setup_generators
import models

if (args.num_epochs is None):
    num_epochs = np.inf
else:
    num_epochs = args.num_epochs

for model_name in model_array:
    models.find_model_using_name(model_name)

for model_name in model_array:
    model = models.find_model_using_name(model_name)
    opt_comet = comet_ml.Optimizer(model.get_config(num_opt))
    experiment_idx = 0
    for experiment in opt_comet.get_experiments(disabled=not(log_comet)):
        experiment.log_parameter("project_name", name)
        experiment.log_parameter("epochs", num_epochs)
        experiment.log_parameter("model", model_name)
        experiment.set_name(f"{experiment.get_parameter('model')}_{experiment_idx}")
        experiment.log_parameter("dataroot", data_path)
        experiment.log_parameter("workers", 4)
        experiment.log_parameter("max_queue_size", 4)
        experiment.log_parameter("use_multiprocessing", "False")
        print(f"Model: {model_name} training iteration {experiment_idx}...")
        gen_train, gen_val, gen_test = setup_generators(data_path, input_array, output_array)

        model_class = models.find_model_using_name(model_name)
        model = model_class.build(experiment, gen_train)
        
        model_class.get_summary(model)
        model_class.compile_model(model, experiment)

        project_name = experiment.get_parameter("project_name")
        print("Training...")
        print(f"You can track your experiment at: https://www.comet.ml/attilasimko/{project_name}")

        if (utils_misc.memory_check(experiment, model) == False):
            val_metric = utils_misc.evaluate(experiment, model, gen_val, "val")
            continue
        
        min_loss = np.inf
        patience = 0
        patience_thr = 20
        epoch = 0
        while (epoch < num_epochs):
            train_loss, val_metric = model_class.train(model, experiment, gen_train, gen_val, epoch)
            experiment.log_metrics({"training_loss": np.mean(train_loss),
                                    "val_metric": np.mean(val_metric)}, epoch=epoch)
            
            if val_metric < min_loss:
                patience = 0
                min_loss = val_metric
                print(f"New lowest validation score reached: {val_metric}.")
            else:
                patience += 1

            if patience > patience_thr:
                print(f"Early stopping of patience ({patience_thr}) reached, training stopped.")
                break
            epoch += 1
        if (epoch >= num_epochs):
            print(f"Maximum number of epochs ({num_epochs}) reached, training stopped.")


        # How well did it do?
        utils_misc.plot_results(experiment, model, gen_val)
        utils_misc.evaluate(experiment, model, gen_test, "test")
        utils_misc.export_weights_to_hero(model, experiment, data_path + "temp/", f"{experiment.get_parameter('model')}_{experiment_idx}")
        experiment_idx += 1
        experiment.end()