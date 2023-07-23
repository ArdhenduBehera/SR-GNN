import sys
import os, ast
import yaml, json
import numpy as np

from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention as Attention
from spektral.utils.sparse import sp_matrix_to_sp_tensor 
from spektral.layers import GCNConv, GlobalAttentionPool, SortPool, TopKPool, GlobalSumPool, GlobalAttnSumPool, ARMAConv, APPNPConv

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout, Flatten, LSTM
from tensorflow.keras import layers 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as pp_input
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop


# user-defined functions (from utils.py)
from RoiPoolingConvTF2 import RoiPoolingConv
from opt_dg_tf2_new import DirectoryDataGenerator
from custom_validate_callback import CustomCallback
from utils import getROIS, getIntegralROIS, crop, squeezefunc, stackfunc

# from models import SR_GNN as model
from models import construct_model


tf.compat.v1.experimental.output_all_intermediates(True)


#######################################################################################


# =============== load and compile ~DEFAULT CONFIGURATION~ parameters
param_dir = "config.yaml"
with open(param_dir, 'r') as file:
    param = yaml.load(file, Loader = yaml.FullLoader)
print('Loading Default parameter configuration: \n', json.dumps(param, sort_keys = True, indent = 3))

# << data parameters >>
nb_classes = param['DATA']['nb_classes']
image_size = tuple( param['DATA']['image_size'] )
dataset_dir = param['DATA']['dataset_dir']

# << hardware parameters >>   ~~~~~~~~~~~~~~~~~~~
multi_gpu = param['HARDWARE']['multi_gpu']
gpu_id = param['HARDWARE']['gpu_id']
gpu_utilisation = param['HARDWARE']['gpu_utilisation']

# << augmentation parameters >>
aug_zoom = param['AUGMENTATION']['aug_zoom']
aug_tx = param['AUGMENTATION']['aug_tx']
aug_ty = param['AUGMENTATION']['aug_ty']
aug_rotation= param['AUGMENTATION']['aug_rotation']

# << model parameters >>    ~~~~~~~~~~~~~~~~~~~
batch_size =  param['MODEL']['batch_size']
lr = param['MODEL']['learning_rate']
model_name = param['MODEL']['model_name']
checkpoint_path = param['MODEL']['checkpoint']

# << training parameters >>
validation_freq = param['TRAIN']['validation_freq']
checkpoint_freq = param['TRAIN']['checkpoint_freq']
epochs = param['TRAIN']['epochs']


# ================= check console paramaters ================== #
if len(sys.argv) > 2: #param 1 is file name
    total_params = len(sys.argv)
    for i in range(1, total_params, 2):
        var_name = sys.argv[i]
        new_val = sys.argv[i+1]
        try:
            exec("{} = {}".format(var_name, new_val))
        except:
            exec("{} = '{}'".format(var_name, new_val))


# ========== Additonal parameters and settings
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('=========================== CUDA ======================== ', os.environ["CUDA_VISIBLE_DEVICES"])
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.2)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
tf.compat.v1.disable_eager_execution()


# =========== Define additional parameters 
lstm_units = 128
base_model = 'resnet_base_flowers'
alpha = 0.3
channels = 512
completed_epochs = 0
ROIS_resolution = 42
minSize = 2
ROIS_grid_size = 3
pool_size=7

# # ====== metric 
# loss_type = 'categorical_crossentropy'
# metrics = ['accuracy']


# ================ Setting up the directories ================= #
if model_name == "model":
    model_name = sys.argv[0].split(".")[0]
# model_name = "{}_{}_{}".format(dataset_dir.split("/")[-1], channels, alpha)

working_dir = os.path.dirname(os.path.realpath(__file__))
train_data_dir = '{}/train/'.format(dataset_dir)
val_data_dir = '{}/val/'.format(dataset_dir)
if not os.path.isdir(val_data_dir):
    val_data_dir = '{}/test/'.format(dataset_dir)

output_model_dir = '{}/TrainedModels/{}'.format(working_dir, model_name)
metrics_dir = '{}/Metrics/{}'.format(working_dir, model_name)
training_metrics_filename = model_name + '(Training).csv'

nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)]) # number of images used for training, including "other" action 
nb_test_samples = 0 # number of images used for testing
nb_val_samples = sum([len(files) for r, d, files in os.walk(val_data_dir)]) # number of images used for validation
validation_steps = validation_freq

# Printing other information
print('\n Location --> ', dataset_dir, '\n', 'no_of_class --> ', nb_classes, '\n', 'input_image_size ---> ', image_size, '\n', 'model_name -->', model_name)

# ============ Building model ============== #
model = construct_model(
    name = model_name,
    pool_size = pool_size,
    ROIS_resolution = ROIS_resolution,
    ROIS_grid_size = ROIS_grid_size,
    minSize = minSize,
    alpha = alpha,
    nb_classes = nb_classes,
    batch_size = batch_size
    )

# model.summary() # !!  print model summary (optional)

# ============  Building engine ============= #
# print('++++++++++++++++++++++++++++++++ checkpoint path:', checkpoint_path, len(checkpoint_path.split('./TrainedModels/')[-1]))
if len(checkpoint_path.split('./TrainedModels/')[-1]) > 0:
    print("~~~~~~~~~~~~~~~~~~ loading previous model saved in --->", checkpoint_path)
    model.load_weights(checkpoint_path)

optimizer = SGD(lr=lr) 
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])

# model.summary() # !!  print model summary (optional)



# ~~~~~~~~~~~~~~~~~~~     Building log file  ~~~~~~~~~~~~~~~~~~~
csv_logger = CSVLogger(metrics_dir + training_metrics_filename)
checkpointer = ModelCheckpoint(
    filepath = output_model_dir + '.{epoch:02d}.h5', 
    verbose = 1, 
    save_weights_only = False, 
    period = checkpoint_freq
    )

# ~~~~~~~~~~~~~~~~~~~     Building Data generators  ~~~~~~~~~~~~~~~~~~~
train_dg = DirectoryDataGenerator(
    base_directories=[train_data_dir], 
    augmentor=True, 
    target_sizes=image_size, 
    preprocessors=pp_input, 
    batch_size=batch_size, 
    shuffle=True, 
    channel_last=True, 
    verbose=1, 
    hasROIS=False
    )

val_dg = DirectoryDataGenerator(
    base_directories=[val_data_dir], 
    augmentor=None, 
    target_sizes=image_size, 
    preprocessors=pp_input, 
    batch_size=batch_size, 
    shuffle=False, 
    channel_last=True, 
    verbose=1, 
    hasROIS=False
    )

# ################ Training Model ############################
model.fit(
    train_dg, 
    steps_per_epoch = nb_train_samples // batch_size, 
    initial_epoch = completed_epochs,  
    epochs = epochs, 
    callbacks = [checkpointer, csv_logger, CustomCallback(val_dg, validation_steps, metrics_dir + model_name)]
    ) #train and validate the model