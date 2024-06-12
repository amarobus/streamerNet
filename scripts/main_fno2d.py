import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import yaml

from streamernet.datasets.streamer_dataset import StreamerDataset
from streamernet.models import FNO2d

torch.set_num_threads(2)

def main():

    model, train_loader, valid_loader, optimizer, loss_fn = setup(config)

    train(model, train_loader, valid_loader, optimizer, loss_fn, config)


def setup(config):
    
    #############################
    # Set up the data loaders
    #############################
    data_dir = os.path.expanduser(config['data']['directory'])

    input_feature = config['input']['feature']
    t_input = config['input']['t_input']
    T = config['input']['T']

    # Training dataset loader
    train_filename = config['data']['train_filename']
    train_file_path = os.path.join(data_dir, train_filename)
    batch_size = config['training']['batch_size']
    shuffle = True

    train_ds = StreamerDataset(train_file_path, input_feature, t_input=t_input, T=T, partition='train')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    # Get min and max values for consistent normalization of the validation data
    min_value = train_ds.min
    max_value = train_ds.max

    # Validation dataset loader
    valid_filename = config['data']['valid_filename']
    valid_file_path = os.path.join(data_dir, valid_filename)
    shuffle = False
    valid_ds = StreamerDataset(valid_file_path, input_feature, t_input=t_input, T=T, partition='valid', min=min_value, max=max_value)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=shuffle)

    #############################
    # Model initialization
    #############################
    in_channels = t_input
    out_channels = 1
    modes1 = config['model']['modes']['mode1']
    modes2 = config['model']['modes']['mode2']
    width = config['model']['width']
    depth = config['model']['depth']
    activation = config['model']['activation']
    model = FNO2d(in_channels, out_channels, modes1, modes2, width, depth, activation=activation).cuda()

    # Setting optimizer
    lr = config['training']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function
    if config['training']['loss'] == 'mse':
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif config['training']['loss'] == 'mae':
        loss_fn = torch.nn.L1Loss(reduction='sum')
    elif config['training']['loss'] == 'normalized_mse':
        loss_fn = lambda x, y: torch.sum(torch.norm(x, y, 2, (1,2,3))/ torch.norm(y, 2, (1,2,3)))
    elif config['training']['loss'] == 'normalized_mae':
        loss_fn = lambda x, y: torch.sum(torch.norm(x, y, 1, (1,2,3))/ torch.norm(y, 1, (1,2,3)))
    else:
        raise ValueError(f"Loss function {config['training']['loss']} not supported.")

    return model, train_loader, valid_loader, optimizer, loss_fn


def train(model, train_loader, valid_loader, optimizer, loss_fn, config):
    
    best = np.inf
    T = train_loader.dataset.T
    step = config['training']['step']
    epochs = config['training']['epochs']
    ntrain = len(train_loader.dataset)
    nvalid = len(valid_loader.dataset)
    
    # Training loop
    for epoch in range(epochs):
        
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Training
        train_loss_step, train_loss_full = training_loop(model, train_loader, optimizer, loss_fn, step)
        
        # Evaluation
        valid_loss_step, valid_loss_full = validation_loop(model, valid_loader, loss_fn, step)

        train_loss_step = train_loss_step / ntrain / (T/step)
        train_loss_full /= ntrain
        valid_loss_step = valid_loss_step / nvalid / (T/step)
        valid_loss_full /= nvalid
        print(f'Full Loss: {train_loss_full}, Step Loss: {train_loss_step}, Valid Full Loss: {valid_loss_full}, Valid Step Loss: {valid_loss_step}')

        # Save best model
        if valid_loss_full < best:
            best = valid_loss_full
            print(f'Saving best model with validation loss: {best}')
            torch.save(model.state_dict(), f"checkpoints/{config['training']['model_name']}.pth")

      
def training_loop(model, train_loader, optimizer, loss_fn, step):

    model.train()
    loss_step = 0
    loss_full = 0
    for x, y in tqdm(train_loader):
        
        optimizer.zero_grad()
        
        loss, full_loss = feed_forward_sequence(model, x, y, loss_fn, step, train_loader.dataset.T)
        
        loss.backward()
        optimizer.step()
        
        loss_step += loss.item()
        loss_full += full_loss.item()

    return loss_step, loss_full


def validation_loop(model, valid_loader, loss_fn, step):
    
    model.eval()
    loss_step = 0
    loss_full = 0
    with torch.no_grad():
        for x, y in valid_loader:           
            loss, full_loss = feed_forward_sequence(model, x, y, loss_fn, step, valid_loader.dataset.T)
        
        loss_step += loss.item()
        loss_full += full_loss.item()

    return loss_step, loss_full


def feed_forward_sequence(model, x, y, loss_fn, step, T):
    
    loss = 0
    x, y = x.cuda(), y.cuda()
    for t in  range(0, T, step):
        
        yy = y[..., t:t+step]
        out = model(x)
        loss += loss_fn(out, yy)

        if t == 0:
            pred = out
        else:
            pred = torch.cat((pred, out), dim=-1)
        
        x = torch.cat((x[..., step:], out), dim=-1)
    
    return loss, loss_fn(pred, y)    


# Function to read the YAML configuration file
def read_config(filepath):
    
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train FNO2d model on streamer discharge samples.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Read the configuration file
    config = read_config(args.input)

    main()