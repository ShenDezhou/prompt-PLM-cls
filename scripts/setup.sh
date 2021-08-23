#import os
#assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'

DIST_BUCKET="gs://tpu-pytorch/wheels"
TORCH_WHEEL="torch-1.15-cp37-cp37m-linux_x86_64.whl"
TORCH_XLA_WHEEL="torch_xla-1.15-cp37-cp37m-linux_x86_64.whl"
TORCHVISION_WHEEL="torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl"

# Install Colab TPU compat PyTorch/TPU wheels and dependencies
pip uninstall -y torch torchvision
gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" .
gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .
gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .
pip install "$TORCH_WHEEL"
pip install "$TORCH_XLA_WHEEL"
pip install "$TORCHVISION_WHEEL"
sudo apt-get install libomp5

VERSION="20200516"  # @param ["1.5" , "20200516", "nightly"]
curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
python pytorch-xla-env-setup.py --version $VERSION

pip3 install -r requirements.txt

#import os
#print(os.environ["COLAB_TPU_ADDR"])
#import torch
#
## imports the torch_xla package
#import torch_xla
#import torch_xla.core.xla_model as xm
#import torch_xla.distributed.parallel_loader as pl
#import torch_xla.distributed.data_parallel as dp
#import torch_xla.distributed.xla_multiprocessing as xmp
#num_cores = 8
#devices = (
#    xm.get_xla_supported_devices(
#        max_devices=num_cores) if num_cores = 0 else [])
#print("Devices: {}".format(devices))

#os.chdir('/content/drive/My Drive/lawbert/')
#python train_model.py -c config/bert_config.json
#
#import os
#os.chdir('/content/drive/MyDrive/lawbert/')
#python model_test.py -c config/bert_verify_config.json
#
#os.chdir('/content/drive/MyDrive/lawbert/')
#python train_model.py -c config/lbert_config.json
#
#os.chdir('/content/drive/My Drive/lawbert/')
#python train_model.py -c config/lbertx_config.json