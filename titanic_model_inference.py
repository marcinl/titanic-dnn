import sys
import os.path
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from titanic_dataset import TitanicDataset
from model_titanic import Network, NetworkSequential


def parse_args(argv, model_path):
    _model_path = model_path
    parser = argparse.ArgumentParser(description='Inference and evaluation for PyTorch Model for Titatnic Dataset')
    parser.add_argument('--model_filename', dest='filename', type=argparse.FileType('rb'), \
        help="Specify the name for saved model file.", metavar="FILE", default=_model_path)

    if len(sys.argv) < 2:
        parser.print_help()

    args = parser.parse_args(argv)
    if (args.filename != None and args.filename != ""):
        _model_path = args.filename  
    return _model_path  



DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_PATH = os.path.join('.', 'titanic_model.pt')


#-----------------------------------------------------------------------
# Read and prepare data 
#-----------------------------------------------------------------------

data = TitanicDataset('test_combined.csv', mode='test')
batch_size = DEFAULT_BATCH_SIZE
data_test = DataLoader(dataset = data, batch_size = batch_size, shuffle =False)

#-----------------------------------------------------------------------
# Load model
#-----------------------------------------------------------------------


# get the project model filename
model_path = parse_args(sys.argv[1:], DEFAULT_MODEL_PATH)

model = NetworkSequential()
model.load_state_dict(torch.load(model_path))
model.eval()


#-----------------------------------------------------------------------
# Run inference 
#-----------------------------------------------------------------------

cuda_available = False #torch.cuda.is_available()

if cuda_available == True:
    device = torch.device("cuda:0")
    print("Cuda Device Available")
    print("Name of the Cuda Device: ", torch.cuda.get_device_name())
    print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
else:
    print("CUDA device not available. Using CPU")
    device = torch.device("cpu")


total_accuracy = list()

model.to(device)
for bidx, batch in tqdm(enumerate(data_test)):
    x_train, y_train = batch['inp'], batch['oup']
    x_train = x_train.view(-1,8)        
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    correct = 0
    acc = 0.0
    predictions = model(x_train.float())
    for idx, i in enumerate(predictions):
        i  = torch.round(i)
        
        if i == y_train[idx].float():
            correct += 1
        
    acc = (correct/len(y_train))
    total_accuracy.append(acc * 100.0)
    print('Batch {} accuracy : {}'.format(bidx + 1, acc * 100.0))

print("Total accuracy: {}".format(np.average(total_accuracy)))