import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from dotenv import load_dotenv
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torch.optim.lr_scheduler import ReduceLROnPlateau

from streamernet.datasets.streamer_dataset import StreamerDataset
from streamernet.models import UNet

torch.set_num_threads(2)

def main():
    model, train_loader, valid_loader, optimizer, scheduler, loss_fn = setup(config)
    train(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, config)

def setup(config):
    # Set up the data loaders
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

    # Model initialization
    in_channels = t_input
    out_channels = 1
    width = config['model']['width']
    filters = config['model']['filters']
    kernel_size = config['model']['kernel_size']
    activation = config['model']['activation']
    padding_type = config['model']['padding_type']
    downsampling = config['model']['downsampling']
    upsampling = config['model']['upsampling']
    
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        width=width,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        padding_type=padding_type,
        downsampling=downsampling,
        upsampling=upsampling
    ).cuda()

    # Setting optimizer
    lr = config['training']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Loss function
    if config['training']['loss'] == 'mse':
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif config['training']['loss'] == 'mae':
        loss_fn = torch.nn.L1Loss(reduction='sum')
    elif config['training']['loss'] == 'normalized_mse':
        loss_fn = lambda x, y: torch.sum(torch.norm(x - y, 2, (1,2,3))/ torch.norm(y, 2, (1,2,3)))
    elif config['training']['loss'] == 'normalized_mae':
        loss_fn = lambda x, y: torch.sum(torch.norm(x - y, 1, (1,2,3))/ torch.norm(y, 1, (1,2,3)))
    else:
        raise ValueError(f"Loss function {config['training']['loss']} not supported.")

    return model, train_loader, valid_loader, optimizer, scheduler, loss_fn

def train(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, config):
    best = np.inf
    T = train_loader.dataset.T
    t_input = train_loader.dataset.t_input
    step = config['training']['step']
    epochs = config['training']['epochs']
    ntrain = len(train_loader.dataset)
    nvalid = len(valid_loader.dataset)
    
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
        print(f'Full Loss: {train_loss_full}, Step Loss: {train_loss_step}, Valid Full Loss: {valid_loss_full}, Valid Step Loss: {valid_loss_step}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        # Save best model
        if valid_loss_full < best:
            best = valid_loss_full
            print(f'Saving best model with validation loss: {best}')
            if args.wandb:
                torch.save(model.state_dict(), f'checkpoints/{config["training"]["model_name"]}_{wandb.run.id}.pth')
            else:
                torch.save(model.state_dict(), f'checkpoints/{config["training"]["model_name"]}.pth')

        # Validation plots and logs
        if args.wandb:
            idx = 3
            num_plots = 10
            if (epoch+1)%5 == 0 or epoch==0:
                # Compute predictions
                model.eval()
                with torch.no_grad():
                    data_batch = next(iter(valid_loader))
                    x, y = data_batch
                    x = x.cuda()
                    for t in range(0, T, step):
                        out = model(x)
                        if t == 0:
                            pred = out
                        else:
                            pred = torch.cat((pred, out), dim=-1)
                        x = torch.cat((x[..., step:], out), dim=-1)

                plot_validation_results(y, pred, valid_loader, num_plots, idx, epoch)
            
            wandb.log(
                {
                    "train_loss_full": train_loss_full,
                    "train_loss_step": train_loss_step,
                    "valid_loss_full": valid_loss_full,
                    "valid_loss_step": valid_loss_step,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"]
                },
                commit=True
            )

        scheduler.step(valid_loss_full)

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
    for t in range(0, T, step):
        yy = y[..., t:t+step]
        out = model(x)
        loss += loss_fn(out, yy)
        if t == 0:
            pred = out
        else:
            pred = torch.cat((pred, out), dim=-1)
        x = torch.cat((x[..., step:], out), dim=-1)
    return loss, loss_fn(pred, y)

def plot_validation_results(y, pred, valid_loader, num_plots, idx, epoch):

    T = valid_loader.dataset.T
    t_input = valid_loader.dataset.t_input

    print('Plotting')
    # Compute predictions

    pred = pred.cpu().numpy()

    # 2D plots
    fig = make_subplots(
        rows=2,
        cols=num_plots,
        subplot_titles=[f't + {i+1}' for i in range(num_plots)]
    )

    color_range = (0, 1)
    for i in range(2):
        for j in range(num_plots):
            if i==0:
                img = y[idx, ..., j]
            else:
                img = pred[idx, ..., j]

            heatmap = go.Heatmap(
                z=img,
                colorscale='jet',
                zmin=color_range[0],
                zmax=color_range[1],
                showscale= True if j==num_plots - 1 else False
            )
            fig.add_trace(heatmap, row=i+1, col=j+1)

    # Fig title
    fig.update_layout(title_text=f'Epoch {epoch}: f:[0, {t_input}] -> [{t_input},{t_input+T}]')

    wandb.log({"Validation Plot Epoch": fig}, commit=False)

    # 1D (along r) plots
    fig = make_subplots(cols=num_plots, subplot_titles=[f't + {i+1}' for i in range(num_plots)])
    for j in range(num_plots):
        line = y[idx, :, 64, j]
        fig.add_trace(go.Scatter(y=line, mode='lines'), row=1, col=j+1)
        line = pred[idx, :, 64, j]
        fig.add_trace(go.Scatter(y=line, mode='lines', line=dict(dash='dot')), row=1, col=j+1)
    fig.update_layout(title_text=f'Epoch {epoch}: f:[0, {t_input}] -> [{t_input},{t_input+T}]')
    wandb.log({"Validation Plot (along r) 1D Epoch": fig}, commit=False)

    # 1D single (time evolution along z) plot
    fig = go.Figure()
    for i in range(num_plots):
            line = y[idx, 0, :, i]
            fig.add_trace(go.Scatter(y=line, mode='lines', line=dict(color='gray')))
            line = pred[idx, 0, :, i]
            # Plot with dotted line
            fig.add_trace(go.Scatter(y=line, mode='lines', line=dict(dash='dot')))
    fig.update_layout(title_text=f'Epoch {epoch}: f:[0, {t_input}] -> [{t_input},{t_input+T}]')
    wandb.log({"Validation Plot (along r) 1D single Epoch": fig}, commit=False)

    # Plot errors
    fig = make_subplots(rows=1, cols=num_plots, subplot_titles=[f't + {i+1}' for i in range(num_plots)])
    # pred = pred.cpu().numpy()
    diff = torch.abs(y[idx] - pred[idx])
    color_range = (diff.min().item(), diff.max().item())
    for j in range(num_plots):
        img = torch.abs(y[idx, ..., j] - pred[idx, ..., j])

        fig.add_trace(go.Heatmap(z=img, colorscale='inferno', zmin=color_range[0], zmax=color_range[1], showscale= True if j==num_plots - 1 else False), row=1, col=j+1)

    # Fig title
    fig.update_layout(title_text=f'Epoch {epoch}: Error f:[0, {t_input}] -> [{t_input},{t_input+T}]')

    wandb.log({"Validation Error Plot": fig}, commit=False)

def read_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet model on streamer discharge samples.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('-l', '--wandb', action='store_true', help='Use Weights & Biases for logging')
    args = parser.parse_args()

    config = read_config(args.input)

    if args.wandb:
        import wandb
        wandb.init(project='streamernet', config=config)
        try:
            main()
        except KeyboardInterrupt:
            print("Interrupted by user. Finishing wandb...")
        finally:
            wandb.finish()
    else:
        main()
