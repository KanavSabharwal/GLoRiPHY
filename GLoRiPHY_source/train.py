import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from tqdm import tqdm
from utils import to_data
import csv
from conformer_mod.model import *
from utils import increase_dropout,spec_to_network_input, spec_to_network_input_complex, get_channel_estimate

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

# Model Creation and Loading
def create_model(opts):
    sf_enc = opts.sf
    if opts.transformer_encoder_dim == 512:
        sf_enc -= 1
    if opts.transformer_encoder_dim == 1024:
        sf_enc -= 2    
    if opts.train_denoiseGen:
        model = ConformerGen2(opts,
                        conformer_for = f'sf_{sf_enc}',
                        encoder_dim=opts.transformer_encoder_dim, 
                        num_attention_heads= opts.num_attention_heads,
                        num_encoder_layers=opts.transformer_layers,
                        input_dropout_p = opts.inp_dropout,
                        feed_forward_dropout_p = opts.ff_dropout,
                        attention_dropout_p = opts.attention_dropout,
                        conv_dropout_p = opts.conv_dropout)

        model_dir = os.path.join(opts.root_path,opts.load_symbol_conformer)
        pretrained_state_dict = torch.load(model_dir, map_location='cpu')
        filtered_model_state_dict = {key: value for key, value in pretrained_state_dict.items() if 'encoder.conv_subsample' not in key}
        model.conformer_symbol.load_state_dict(filtered_model_state_dict, strict=True)
        print(f"Symbol Conformer loaded from {model_dir}")
        for param in model.conformer_symbol.parameters():
            param.requires_grad = False
        print("Symbol Conformer parameters frozen")

    elif opts.train_denoiseGenCore:
        model = ConformerGen_Core(opts,
                                conformer_for = f'sf_{sf_enc}',
                                encoder_dim=opts.transformer_encoder_dim, 
                                num_attention_heads= opts.num_attention_heads,
                                num_encoder_layers=opts.transformer_layers,
                                input_dropout_p = opts.inp_dropout,
                                feed_forward_dropout_p = opts.ff_dropout,
                                attention_dropout_p = opts.attention_dropout,
                                conv_dropout_p = opts.conv_dropout)
    else:
        AssertionError("Invalid model type")
    
    if torch.cuda.is_available():
        model.cuda()
        print('Models moved to GPU.')
    # print_models(model)
    return model

def checkpoint(iteration, conformer, opts):
    conformer_path = os.path.join(opts.checkpoint_dir, str(iteration) + '_conformer_jgj.pkl')
    torch.save(conformer.state_dict(), conformer_path)

def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    model_dir = os.path.join(opts.root_path,opts.load)
    if opts.load_epoch == -1:
        conformer_path = os.path.join(model_dir, 'best_conformer_jgj.pkl')
    else:
        conformer_path = os.path.join(model_dir, str(opts.load_epoch) + '_conformer_jgj.pkl')
    print("\n\nLoading model from: {}\n".format(conformer_path))

    conformer = create_model(opts)

    conformer.load_state_dict(torch.load( conformer_path, map_location=lambda storage, loc: storage),
        strict=True)

    if torch.cuda.is_available():
        conformer.cuda()
        print('Models moved to GPU.')

    return conformer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Denoise Core Training
def training_loop_denoiseCore(train_dloader,test_dloader, opts):
    # Create generators and discriminators
    if opts.load:
        model = load_checkpoint(opts)
    else:
        model = create_model(opts)

    mse_loss_fn = nn.MSELoss(reduction='mean')
    cat_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opts.lr, betas=(opts.beta1, opts.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opts.sched_factor, 
                                                     patience=opts.sched_patience, min_lr=1e-7,
                                                     threshold = 1e-4)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    early_stopping_patience = opts.early_stopping_patience
    early_stopping_counter = 0

    log_file_path = os.path.join(opts.checkpoint_dir, 'training_log.csv')
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Phase','Loss', 'MSELoss', 'CCELoss', 'Accuracy'])

    for epoch in range(opts.num_epochs):

        if epoch == opts.phase1:
            opts.snr_list = list(range(-20,0))
            model.encoder.input_dropout.p = 0.1
        if epoch == opts.phase2:
            opts.snr_list.extend(list(range(-25,-20)))
            increase_dropout(model, 0.05)
        if epoch == opts.phase3:
            opts.snr_list.extend(list(range(-30,-25)))
            increase_dropout(model, 0.05)
        if epoch in [opts.phase1,opts.phase2,opts.phase3]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opts.lr
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opts.sched_factor, 
                                            patience=opts.sched_patience, min_lr=1e-9,
                                            threshold = 1e-4)
            if os.path.exists( os.path.join(opts.checkpoint_dir,'best_conformer_jgj.pkl')):
                model.load_state_dict(torch.load(os.path.join(opts.checkpoint_dir,'best_conformer_jgj.pkl'), map_location=lambda storage, loc: storage),strict=True)
                print("Model loaded from best_conformer_jgj.pkl")
            model.process_output[2].p = 0.1
            # print_models(model)

        print(f'Epoch {epoch+1}/{opts.num_epochs}')
        print('-' * 10)
        
        # print("Using SNR values: ",opts.snr_list)
        
        epoch_metrics = {}

        # Each epoch has a training and validation phase
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
                
                symbol_recd_stft = torch.stft(symbol, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=opts.hamming_window)
                symbol_recd_stft = spec_to_network_input(symbol_recd_stft,opts)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    masked_symbol,outputs,preds = model(symbol_recd_stft)
                    preds = torch.round(preds)

                    mse_loss = mse_loss_fn(masked_symbol,symbol_gt_stft)
                    cat_loss = cat_loss_fn(outputs, labels)
                    loss = opts.mse_scaling*mse_loss + cat_loss

                    # Backward + optimize only if in training phase
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

            if phase == 'val':
                scheduler.step(epoch_loss)
                current_lr = scheduler.get_last_lr()
                print(f"Current learning rate: {current_lr}")
            
            print(f'{phase} Loss: {epoch_loss:.4f}    MSE Loss:{epoch_mse_loss:.4f}    CCE Loss:{epoch_cat_loss:.4f}    Acc: {epoch_acc:.4f}')
            epoch_metrics[phase] = {'Loss': epoch_loss,  'Accuracy': epoch_acc, 'MSELoss': epoch_mse_loss, 'CCELoss': epoch_cat_loss}

            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint('best', model, opts)
                early_stopping_counter = 0  # Reset counter if validation loss improves
            elif epoch in [opts.phase1,opts.phase2,opts.phase3]:
                best_loss = epoch_loss
                early_stopping_counter = 0 
            elif phase == 'val':
                early_stopping_counter += 1

        # Save the model parameters
        if epoch % opts.checkpoint_every == 0:
            checkpoint(epoch, model, opts)

        # Early stopping:
        if early_stopping_counter >= early_stopping_patience:
            train_phases = [opts.phase1,opts.phase2,opts.phase3]
            next_phase_index = next((i for i, phase in enumerate(train_phases) if phase > epoch), None)
            
            if next_phase_index is not None:
                if next_phase_index == 0:
                    opts.phase1 = epoch + 1
                elif next_phase_index == 1:
                    opts.phase2 = epoch + 1
                elif next_phase_index == 2:
                    opts.phase3= epoch + 1
                print(f"Early stopping adjusted, phase {next_phase_index + 1} set to epoch {epoch + 1}")
                print(f"Phase 1 set to epoch {opts.phase1}, Phase 2 set to epoch {opts.phase2}, Phase 3 set to epoch {opts.phase3}")
                early_stopping_counter = 0
            else:
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

        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the best model checkpoint
    torch.save(model.state_dict(), os.path.join(opts.checkpoint_dir, 'best_model_checkpoint_jgj.pth'))

    print('Best val loss: {:4f}'.format(best_loss))

# DenoiseCore Testing Loop 
def testing_loop_denoiseCore(test_dloader, opts):
    # Create generators and discriminators
    model = load_checkpoint(opts)
    model.eval()   # Set model to evaluate mode
    dataloader = test_dloader

    running_corrects = 0

    # Iterate over data.
    progress_bar = tqdm(dataloader, desc=f"Val Progress", leave=False)
    for (symbol,_), labels in progress_bar:
        symbol = symbol.to(opts.device)
        labels = labels.to(opts.device)
        
        symbol_recd_stft = torch.stft(symbol, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=opts.hamming_window)
        symbol_recd_stft = spec_to_network_input(symbol_recd_stft,opts)

        with torch.no_grad():
            _,_,preds = model(symbol_recd_stft)
            preds = torch.round(preds)

        curr_corrects = torch.sum(preds == labels.data)
        running_corrects += curr_corrects

        progress_bar.set_description(f"Val Progress (Acc: {curr_corrects*100/opts.batch_size:.4f})")

    epoch_acc = running_corrects.double() / len(dataloader.dataset) * 100
    
    print(f'Test Acc: {epoch_acc:.4f}')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Denoise with Preamble Training
def training_loop_denoise(train_dloader, test_dloader, opts):
    if opts.load:
        model = load_checkpoint(opts)
    else:
        model = create_model(opts)

    mse_loss_fn = nn.MSELoss(reduction='mean')
    cat_loss_fn = nn.CrossEntropyLoss()

    print("Number of parameters in the model being trained: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total number of parameters in the model: ", sum(p.numel() for p in model.parameters()))
    print(f"Testing for Nodes {opts.test_nodes}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opts.lr, betas=(opts.beta1, opts.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opts.sched_factor, 
                                                     patience=opts.sched_patience, min_lr=1e-7,
                                                     threshold = 1e-4)

    best_model_wts = copy.deepcopy(model.state_dict())
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

        phases = ['train', 'val']
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
                if opts.train_denoiseGen:
                    model.conformer_symbol.eval()
                dataloader = train_dloader
            elif phase == 'val':
                model.eval()   # Set model to evaluate mode
                dataloader = test_dloader

            running_loss = 0.0
            running_mse_loss = 0.0
            running_cat_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            progress_bar = tqdm(dataloader, desc=f"{phase} Progress", leave=False)
            for (preamble_recd,sfd_recd,symbol,symbol_gt_stft,symbol_ind), labels in progress_bar:
                preamble_recd,sfd_recd,symbol,symbol_gt_stft,symbol_ind = preamble_recd.to(opts.device),sfd_recd.to(opts.device),symbol.to(opts.device), symbol_gt_stft.to(opts.device),symbol_ind.to(opts.device)
                labels = labels.to(opts.device)

                preamble_recd_stft = torch.stft(preamble_recd, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=opts.hamming_window)
                sfd_recd_stft = torch.stft(sfd_recd, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=opts.hamming_window)

                preamble_recd_stft = spec_to_network_input_complex(preamble_recd_stft,opts)
                sfd_recd_stft = spec_to_network_input_complex(sfd_recd_stft,opts)
                preamble_recd_stft = torch.cat((preamble_recd_stft,sfd_recd_stft), dim = -1)
                preamble_recd_stft_input = get_channel_estimate(preamble_recd_stft,opts)
                
                symbol_recd_stft = torch.stft(symbol, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=opts.hamming_window)
                symbol_recd_stft = spec_to_network_input(symbol_recd_stft,opts)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    masked_symbol,outputs,preds = model(preamble_recd_stft_input,symbol_recd_stft)
                    preds = torch.round(preds)

                    mse_loss = mse_loss_fn(masked_symbol,symbol_gt_stft)
                    cat_loss = cat_loss_fn(outputs, labels)
                    loss = opts.mse_scaling*mse_loss + cat_loss

                    # Backward + optimize only if in training phase
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

            if phase == 'val':
                scheduler.step(epoch_loss)
                current_lr = scheduler.get_last_lr()
                print(f"Current learning rate: {current_lr}")
            
            print(f'{phase} Loss: {epoch_loss:.4f}    MSE Loss:{epoch_mse_loss:.4f}    CCE Loss:{epoch_cat_loss:.4f}    Acc: {epoch_acc:.4f}')
            epoch_metrics[phase] = {'Loss': epoch_loss,  'Accuracy': epoch_acc, 'MSELoss': epoch_mse_loss, 'CCELoss': epoch_cat_loss}

            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint('best', model, opts)
                early_stopping_counter = 0  # Reset counter if validation loss improves
            elif phase == 'val':
                early_stopping_counter += 1

        # Save the model parameters
        if epoch % opts.checkpoint_every == 0:
            checkpoint(epoch, model, opts)

        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for phase in phases:
                writer.writerow([epoch+1, phase, 
                                 epoch_metrics[phase]['Loss'], 
                                 epoch_metrics[phase]['MSELoss'],
                                 epoch_metrics[phase]['CCELoss'],
                                 epoch_metrics[phase]['Accuracy']])
        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the best model checkpoint
    torch.save(model.state_dict(), os.path.join(opts.checkpoint_dir, 'best_model_checkpoint_jgj.pth'))

    print('Best val loss: {:4f}'.format(best_loss))

# Denoise with Preamble Testing Loop
def testing_loop_denoise(dataloader, opts):
    assert opts.load, "opts.load must be set to load the model for testing"
    model = load_checkpoint(opts)

    mse_loss_fn = nn.MSELoss(reduction='mean')
    cat_loss_fn = nn.CrossEntropyLoss()

    log_file_path = os.path.join(opts.checkpoint_dir, 'testing_log.csv')

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Testing Progress", leave=False)
        for (preamble_recd,sfd_recd,symbol,symbol_gt_stft,symbol_ind), labels in progress_bar:
            preamble_recd,sfd_recd,symbol,symbol_gt_stft,symbol_ind = preamble_recd.to(opts.device),sfd_recd.to(opts.device),symbol.to(opts.device), symbol_gt_stft.to(opts.device),symbol_ind.to(opts.device)
            labels = labels.to(opts.device)

            preamble_recd_stft = torch.stft(preamble_recd, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=opts.hamming_window)
            sfd_recd_stft = torch.stft(sfd_recd, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=opts.hamming_window)
            
            preamble_recd_stft = spec_to_network_input_complex(preamble_recd_stft,opts)
            sfd_recd_stft = spec_to_network_input_complex(sfd_recd_stft,opts)
            preamble_recd_stft = torch.cat((preamble_recd_stft,sfd_recd_stft), dim = -1)
            preamble_recd_stft_input = get_channel_estimate(preamble_recd_stft,opts)
        
            symbol_recd_stft = torch.stft(symbol, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=opts.hamming_window)
            symbol_recd_stft = spec_to_network_input(symbol_recd_stft,opts)

            masked_symbol,outputs,preds = model(preamble_recd_stft_input,symbol_recd_stft)
            preds = torch.round(preds)
            
            mse_loss = mse_loss_fn(masked_symbol,symbol_gt_stft)
            cat_loss = cat_loss_fn(outputs, labels)
            loss = opts.mse_scaling*mse_loss + cat_loss
            # Statistics
            running_loss += loss.item() * symbol_recd_stft.size(0)
            running_corrects += torch.sum(preds == labels.data)

            progress_bar.set_description(f"Testing Progress (Loss: {loss.item():.4f})")

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset) * 100

        print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['GLoRiPHY', epoch_acc.item()])

