# LoRaPHY Python Simulation

We have developed a Python simulation environment for end-to-end encoding, modulation, decoding, and demodulation in **LoRaPHY**. These functionalities are implemented as independent classes in the following files:

- `encode.py`
- `mod.py`
- `decode.py`
- `demod.py`

This simulation environment is based on the findings from the paper [From Demodulation to Decoding: Toward Complete LoRa PHY Understanding and Implementation](https://dl.acm.org/doi/10.1145/3546869).

## Running the Simulation

We provide an example simulation in the Jupyter notebook `LoRaPHY_Simulation_Example.ipynb`. This example simulates a *LoRaPHY packet* with a random payload, demonstrating the complete process from encoding to demodulation.

## Modularity and Customization

The simulation environment is designed to be modular, enabling users to easily modify parameters and functionalities to suit their needs. You can adjust the following settings:

- Spreading Factor (SF)
- Coding Rate
- Bandwidth
- Sampling Rate
- Implicit/Explicit Header Mode
- CRC
- Payload Length
- FFT parameters for demodulation

This flexibility makes the environment highly suitable for researchers and engineers who want to explore and experiment with various aspects of the LoRa PHY layer.
