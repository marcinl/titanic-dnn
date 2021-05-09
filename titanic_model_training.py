import sys
import os.path
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from titanic_dataset import TitanicDataset
from model_titanic import Network, NetworkSequential

from torch.optim import Adam


def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x.float())
  
    #print("Model output: {} for trained: {}".format(output, y.float()))
    loss = criterion(output,y.float())   
    loss.backward()
    optimizer.step()

    return loss, output
   
def parse_args(argv, model_path):
    _model_path = model_path
    parser = argparse.ArgumentParser(description='Train PyTorch Model for Titatnic Dataset')
    parser.add_argument('--model_filename', dest='filename', type=argparse.FileType('wb'), \
        help="Specify the name for model output file.", metavar="FILE", default=_model_path)

    if len(sys.argv) < 2:
        parser.print_help()

    args = parser.parse_args(argv)
    if (args.filename != None and args.filename != ""):
        _model_path = args.filename  
    return _model_path  

#-----------------------------------------------------------------------
# Read and prepare data 
#-----------------------------------------------------------------------

EPOCHS = 200
BATCH_SIZE = 16
DEFAULT_MODEL_PATH = os.path.join('.', 'titanic_model.pt')

# get the project model filename
model_path = parse_args(sys.argv[1:], DEFAULT_MODEL_PATH)

#net = Network()
net = NetworkSequential()
#optm = Adam(net.parameters(), lr = 0.001)
optm = Adam(net.parameters(), lr = 0.003)

data = TitanicDataset('train.csv', mode='train')
data_train = DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle =False)

cuda_available = False #torch.cuda.is_available()

if cuda_available == True:
    device = torch.device("cuda:0")
    print("Cuda Device Available")
    print("Name of the Cuda Device: ", torch.cuda.get_device_name())
    print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
else:
    print("CUDA device not available. Using CPU")
    device = torch.device("cpu")

#-----------------------------------------------------------------------
# Run traning 
#-----------------------------------------------------------------------


criterion = nn.MSELoss()
for epoch in range(EPOCHS):
    correct = 0
    acc = 0.0
    epoch_loss = 0
    net.to(device)
    for bidx, batch in tqdm(enumerate(data_train)):
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.view(-1,8)        
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        
        loss, predictions = train(net,x_train,y_train, optm, criterion)
        for idx, i in enumerate(predictions):
            i  = torch.round(i)
            if i == y_train[idx].float():
                correct += 1
        
        acc = (correct/len(data))
        if torch.isnan(loss) == False:
            epoch_loss += loss.item()
    print('Epoch {} Accuracy : {}'.format(epoch + 1, acc * 100.0))
    print('Epoch {} Loss : {}'.format((epoch + 1),epoch_loss))


#-----------------------------------------------------------------------
# Save the model to a file
#-----------------------------------------------------------------------


torch.save(net.state_dict(), model_path)
