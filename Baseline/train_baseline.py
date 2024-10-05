import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from tqdm import tqdm
import csv
from baseline_mod import NeLoRa
from utils_baseline import spec_to_network_input

import time

SEED = 11

# Set the random seed manually for reproducibility.
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def print_models(Model):
    """Prints model information for the generators and discriminators.
    """
    print("                 Model                ")
    print("---------------------------------------")
    print(Model)
    print("---------------------------------------")

def create_model(opts):
    model = NeLoRa(opts)
    
    if torch.cuda.is_available():
        model.cuda()
        print('Models moved to GPU.')
    # print_models(model)
    return model

def checkpoint(iteration, model, opts):
    model_path = os.path.join(opts.checkpoint_dir, str(iteration) + '_nelora_jgj.pkl')
    torch.save(model.state_dict(), model_path)

def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    model = create_model(opts)
    model_dir = os.path.join(opts.root_path,opts.load)    
    if opts.load_epoch == -1:
        model_path = os.path.join(model_dir, 'best_nelora_jgj.pkl')
    else:
        model_path = os.path.join(model_dir, str(opts.load_epoch) + '_nelora_jgj.pkl')
    print("\n\nLoading model from: {}\n".format(model_path))        
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage),strict=False)

    if torch.cuda.is_available():
        model.cuda()
        print('Models moved to GPU.')
    return model

def training_loop(train_dloader,test_dloader, opts):
    if opts.load:
        model = load_checkpoint(opts)
    else:
        model = create_model(opts)

    mse_loss_fn = torch.nn.MSELoss(reduction='mean')
    cat_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

    best_loss = np.inf
    early_stopping_patience = opts.early_stopping_patience
    early_stopping_counter = 0

    log_file_path = os.path.join(opts.checkpoint_dir, 'training_log.csv')
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Phase', 'Loss', 'MSELoss', 'CCELoss', 'Accuracy'])

    for epoch in range(opts.num_epochs):
        print(f'Epoch {epoch+1}/{opts.num_epochs}')
        print('-' * 10)

        epoch_metrics = {}
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_dloader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_dloader

            running_loss = 0.0
            running_mse_loss = 0.0
            running_cat_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            progress_bar = tqdm(dataloader, desc=f"{phase} Progress", leave=False)
            for (symbol,symbol_gt_stft), labels in progress_bar:
                symbol,symbol_gt_stft = symbol.to(opts.device), symbol_gt_stft.to(opts.device)
                labels = labels.to(opts.device)

                symbol_recd_stft = torch.stft(input=symbol, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap, win_length=opts.stft_window, pad_mode='constant')
                symbol_recd_stft = spec_to_network_input(symbol_recd_stft, opts)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    masked_symbol,outputs = model(symbol_recd_stft)
                    _, preds = torch.max(outputs, 1)

                    mse_loss = mse_loss_fn(masked_symbol,symbol_gt_stft)
                    cat_loss = cat_loss_fn(outputs, labels)
                    loss = (opts.scaling_for_imaging_loss * mse_loss) + (opts.scaling_for_classification_loss * cat_loss)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * symbol_recd_stft.size(0)
                running_mse_loss += mse_loss.item() * symbol_recd_stft.size(0)
                running_cat_loss += cat_loss.item() * symbol_recd_stft.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_description(f"{phase} Progress (Loss: {loss.item():.4f})")

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_mse_loss = running_mse_loss / len(dataloader.dataset)
            epoch_cat_loss = running_cat_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset) * 100
            
            print(f'{phase} Loss: {epoch_loss:.4f}    MSE Loss:{epoch_mse_loss:.4f}    CCE Loss:{epoch_cat_loss:.4f}    Acc: {epoch_acc:.4f}')
            epoch_metrics[phase] = {'Loss': epoch_loss,  'Accuracy': epoch_acc, 'MSELoss': epoch_mse_loss, 'CCELoss': epoch_cat_loss}

            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint('best', model, opts)
                early_stopping_counter = 0
            elif phase == 'val':
                early_stopping_counter += 1


        # Save the model parameters
        if epoch % opts.checkpoint_every == 0:
            checkpoint(epoch, model, opts)

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for phase in ['train', 'val']:
                writer.writerow([epoch+1, phase, 
                                 epoch_metrics[phase]['Loss'], 
                                 epoch_metrics[phase]['MSELoss'],
                                 epoch_metrics[phase]['CCELoss'],
                                 epoch_metrics[phase]['Accuracy']])

    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Save the best model checkpoint
    torch.save(model.state_dict(), os.path.join(opts.checkpoint_dir, 'best_model_checkpoint_jgj.pth'))
    print('Best val loss: {:4f}'.format(best_loss))

def testing_loop(dataloader, opts):
    assert opts.load, "opts.load must be set to load the model for testing"
    model = load_checkpoint(opts)

    model.eval()   # Set model to evaluate mode

    mse_loss_fn = nn.MSELoss(reduction='mean')

    log_file_path = os.path.join(opts.checkpoint_dir, 'testing_log.csv')

    running_mse_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Testing Progress", leave=False)
        for (symbol,symbol_gt_stft), labels in progress_bar:
            symbol,symbol_gt_stft = symbol.to(opts.device), symbol_gt_stft.to(opts.device)
            labels = labels.to(opts.device)

            symbol_recd_stft = torch.stft(input=symbol, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap, win_length=opts.stft_window, pad_mode='constant')
            symbol_recd_stft = spec_to_network_input(symbol_recd_stft, opts)
            masked_symbol,outputs = model(symbol_recd_stft)
            _, preds = torch.max(outputs, 1)
            # Statistics
            curr_corrects = torch.sum(preds == labels.data)
            mse_loss = mse_loss_fn(masked_symbol,symbol_gt_stft)
            running_mse_loss += mse_loss.item() * symbol_recd_stft.size(0)
            running_corrects += curr_corrects
            progress_bar.set_description(f"Testing Progress (Acc: {curr_corrects*100/opts.batch_size:.4f})")

        epoch_mse_loss = running_mse_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset) * 100
        print(f'Test Acc: {epoch_acc:.4f}')
        print(f'Test MSE Loss: {epoch_mse_loss:.4f}')

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['NELoRa', epoch_acc.item()])