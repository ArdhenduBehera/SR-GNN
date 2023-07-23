# Instruction with installation and envirnment setup.
---> To create a conda environment with basic package use the following command:

conda create -n <<environment_name>> anaconda python=<<python_version>>

We have used python version --->  3.9.16

---> Other necessary required imports can be install by running the "requirements.txt" file:

pip install -r requirements.txt

<< Optional >>
If you want to use gpu for training first check your driver version using command "nvidia-smi" from console, then install the appropiate cudatoolkit, cudnn packages after verifing from these sources

-->  Linux cuda compatibility : https://www.tensorflow.org/install/source#tested_build_configurations
-->  Windows cuda compatibility : https://www.tensorflow.org/install/source_windows#tested_build_configurations

# Instruction with dataset organization 

The dataset directory structure (inside the datasets folder) should be like the below tree structure where each folder xx inside train and test should contain fine grained images

datasets <br>
├── dataset_1 <br>
│   ├── train <br>
│   │    ├── folder 01 <br>
│   │    ├── folder 02 <br>
│   │    └── ...  <br>
│   └── test <br>
│        ├── folder 01 <br>
│        ├── folder 02 <br>
│        └── ... <br>
│ <br>
│  ... <br>
│ <br>
└── dataset_2 <br>
    ├── train <br>
    │    ├── folder 01 <br>
    │    ├── folder 02 <br>
    │    └── ... <br>
    └── test <br>
        ├── folder 01 <br>
        ├── folder 02 <br>
        └── ... <br>
 <br>

# Model hyperparameters and other configuration setting 

Find some of the default parameter configuration in "config.yaml" file. You can directly change in this config file or pass as an argument in console


# From python console 

--> To train from a particular checkpoint from a particular epoch (say 50) use command:

python ./script/main.py dataset_dir ./datasets/Cars nb_classes 196 gpu_id -1 batch_size 8 completed_epochs 50 epochs 150 checkpoint_path ./TrainedModels/Cars_512_0.3.50.h5 validation_freq 2 model_name srgnn


--> To train from scratch use command:

python ./script/main.py dataset_dir ./datasets/Cars nb_classes 196 gpu_id -1 batch_size 8 epochs 150 validation_freq 2 model_name srgnn
