# Training a GLoRiPHY Model

This section provides instructions on how to train the GLoRiPHY model.

## 1. Training GLoRiPHY's AWGN Filter

To train the AWGN filter of the GLoRiPHY model, run the following command:

```bash
python main.py \
    --root_path "$(pwd)" \
    --train_denoiseGenCore --num_epochs 700 \
    --transformer_layers 2 \
    --sf 8 \
    --transformer_encoder_dim 256 \
    --attention_dropout 0.15 \
    --ff_dropout 0.1 --conv_dropout 0.1 --inp_dropout 0.1 \
    --checkpoint_dir checkpoints/AWGN_Filter_new/simulation/SF \
    --sched_factor 0.4
```
## 2. Training the Channel Compensation Module (Overall Model)
### On Real-World Dataset
To train the model on real-world data, run the following command:
```bash
python main.py \
    --root_path "$(pwd)" \
    --train_denoiseGen --num_epochs 200 \
    --data_dir /absolute/path/to/Dataset/real_world/test_Node_D \
    --sf 8 \
    --num_packets 200 \
    --test_nodes 5 \
    --checkpoint_dir checkpoints/denoiseGen/real_world/SF8_check \
    --load_symbol_conformer checkpoints/AWGN_Filter_new/simulation/SF_8/best_conformer_jgj.pkl \
    --real_data
```

### On Simulated Dataset
To train the model on the simulated dataset, run the following command:
```bash
python main.py \
    --root_path "$(pwd)" \
    --train_denoiseGen  --num_epochs 200 \
    --data_dir /absolute/path/to/Dataset/simulation/test_SF_8 \
    --num_packets 100 \
    --num_perturbations 1200 \
    --checkpoint_dir checkpoints/denoiseGen/simulated/SF8_check \
    --load_symbol_conformer checkpoints/AWGN_Filter_new/simulation/SF_8/best_conformer_jgj.pkl \
    --sim_data
```
### Quick Testing of the Training Procedure
To quickly test the training procedure, use the above commands by pointing to the provided test dataset and modifying the number of training epochs.

## Model and Training Parameters
The ```config.py``` file holds the default values for the command line arguments passed during model training and testing. Although the default values are set to the recommended settings, below are some key arguments that you can modify:

- ```transformer_encoder_dim```: Refers to the Embedding Dimension of the GLoRiPHY model. For SF >= 10, you should change the default value to 512. A higher value (e.g., 1024) is expected to provide better accuracy but at the cost of a larger model, resulting in longer training/inference times.
- ```sf```: Refers to the *Spreading Factor* for the current configuration.
- ```num_packets```: Refers to the number of unique packets in the dataset.
-```num_perturbations```: Refers to the number of distinct perturbation channels in the Simulated Dataset.
- ```lr```: Learning Rate. For training the AWGN filter for higher SFs (i.e., SF >= 10), we found a lower learning rate (7e-05) to be more efficient.
- ```sched_factor``` and ```sched_patience```: Control the learning rate scheduler's parameters.
- ```test_mode```: Triggers model testing.
- `test_node`: Specifies the node ID to be excluded from the training dataset, allowing for *leave-one-out training*. 
- ```real_data``` OR ```sim_data```: Specifies whether to use the real-world or simulated dataset.
- ```load```: Specifies the path to an existing trained model's weights. The load_epoch argument specifies which epoch to load (default -1 loads the best epoch).
- ```load_symbol_conformer```: Used to specify the path to the pre-trained AWGN filter model weights.
- ```phase1/2/3```: Specify the number of epochs for each phase in the curriculum learning approach used to train the AWGN filter. Each phase expands the range of SNRs, increasing the negative range.
- ```free_gpu_id```: ID of the available GPU to be used for training. This is typically passed to utilize a specific GPU in a multi-GPU setup.

