# GLoRiPHY - Artifact for "Enhancing LoRa Reception with Generative Models: Channel-Aware Denoising of LoRaPHY Signals" (SenSys'24)

We maintain this repository to share the artifacts from our paper titled **"Enhancing LoRa Reception with Generative Models: Channel-Aware Denoising of LoRaPHY Signals"**, published in SenSys'24. The repository includes implementations and datasets necessary to reproduce some of the key results from the paper.

## Setup and Test to Reproduce Results

### 1. Environment Setup
To get started, set up the environment using the dependencies shared in `requirements.txt`

### 2. Dataset Download
Given the size of the dataset, we split it into two parts. Follow these steps to download and prepare the data:

1. Download the first part from [Dataset.tar.gz](https://drive.google.com/file/d/1GG1llA4EEPRPYk-kzCheMH918EOK6Oxu/view?usp=share_link).
2. Uncompress the file
```bash
tar -xvzf Dataset.tar.gz
```
3. Download the second part from [test_SF_9.tar.gz](https://drive.google.com/file/d/1zo6f_Tu907BzBThTrA1Cz68MEkZQxkWF/view?usp=share_link).
4. Uncompress it in the same way and move it to ```Dataset/simulation/```:
```bash
tar -xvzf test_SF_9.tar.gz
mv test_SF_9 Dataset/simulation/
```
### 3. Checkpoints Download
Before running the script, you also need to download the ```checkpoints``` folder from [checkpoints_gloriphy.tar.gz](https://drive.google.com/file/d/1OHo4wR2lN6_T0rTOzv-os4mySRGWZ75m/view?usp=share_link). These contain the pre-trained model weights.
1. Download the ```checkpoints_gloriphy.tar.gz``` file.
2. Uncompress it 
```bash
tar -xvzf checkpoints_gloriphy.tar.gz
```
3. Move the uncompressed folder to the ```GLoRiPHY/GLoRiPHY_source``` directory:
```bash
mv checkpoints GLoRiPHY/GLoRiPHY_source
```
4. Download the [checkpoints_nelora.tar.gz](https://drive.google.com/file/d/1BdM1v3CVF10VFWurLquNKmMpoV6gJaAh/view?usp=sharing) file.
5. Uncompress it and move the uncompressed folder to the ```GLoRiPHY/Basline/``` directory:
```bash
tar -xvzf checkpoints_nelora.tar.gz
mv checkpoints GLoRiPHY/Baseline/
```

**Note:** Before running the script, ensure the file structure looks like this:
>
> ```
> - GLoRiPHY/
>   -- GLoRiPHY_source/
>       --- checkpoints/
>   -- Baseline/
>       --- checkpoints/
> ```
>
And the ```Dataset``` folder should look like this:
>
> ```
> - Dataset/
>   -- real_world/
>      --- SF_8_GT
>      --- test_Node_D
>      --- test_Node_E
>      --- test_Node_F
>      --- test_Node_G
>      --- test_Node_H
>   -- simulation/
>      --- test_SF_8
>      --- test_SF_9
> ```


### 4. Running the Script
Now you can run the bash script to generate the results. Please ensure to pass the *absolute path* to the `Dataset/`, and optionally, your preferred GPU ID:

```bash
./test_script.sh /absolute/path/to/Dataset [GPU_ID]
```

> **Note:** If you encounter a "Permission Denied" error when running the script, use the following command to make it executable:
>```bash
>chmod +x test_script.sh
>```

The process may take some time but will eventually generate two graphs. These graphs can be found in `GLoRiPHY/GLoRiPHY_source/testing`. The files `testing_sim.pdf` and `real_nodes_unseen.pdf` correspond to **Figure 8** and **Figure 9(b)** from the paper, respectively. These graphs represent a subset of the main results. Given the extensive data requirements, we provide a representative subset of the data.

## Repository Structure
### 1. GLoRiPHY_source
This folder contains the core implementation of *GLoRiPHY*, including data loaders and other relevant files. The Conformer-based model is built on the [Conformer Implementation](https://github.com/sooftware/conformer), *last accessed on 2 July 2024*.

### 2. Baseline
The Baseline folder implements the baseline *NELoRa*. We adapt the original code from the [NELoRa-Sensys](https://github.com/hanqingguo/NELoRa-Sensys) and [NELoRa-Dataset](https://github.com/daibiaoxuwu/NeLoRa_Dataset) repositories, *last accessed on 2 July 2024*.

### 3. LoRaPHY
We share a Python simulation environment for end-to-end encoding, modulation, decoding, and demodulation in LoRaPHY. This implementation is based on findings from [From Demodulation to Decoding: Toward Complete LoRa PHY Understanding and Implementation](https://dl.acm.org/doi/10.1145/3546869), and we adapted the GNU Radio [code](https://github.com/jkadbear/gr-lora) shared by the authors to implement a python-based simulation.

### 4. Simulation_MATLAB
This folder provides our implementation of the simulation framework used to generate the simulated dataset, as presented in *Figure 7(b)* of our paper.

## Further Details
We will share more information specific to each module in their respective ```README``` files.
