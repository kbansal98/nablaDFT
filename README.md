# Nabla VAE

A Variational Autoencoder (VAE) implementation for processing and modeling Hamiltonian datasets using custom neural network architectures.

<img src="https://github.com/kbansal98/nablaDFT/blob/main/nablaData.jpg" width= "700" height = "400"/>




## Overview

This project provides a VAE model specifically designed to work with Hamiltonian matrices from the nablaDFT dataset. It includes preprocessing functions, a custom neural network implementation, and methods for training and evaluating the model. The goal is to encode Hamiltonian data into a latent space and then reconstruct it, making it useful for tasks such as anomaly detection, data compression, or generating new samples.

<img src="https://github.com/kbansal98/nablaDFT/blob/main/architecture.jpg" width="200" height="400" />,      <img src="https://github.com/kbansal98/nablaDFT/blob/main/generalvaearch.png" width= "500" height = "400"/>
## Features

- **Data Processing**: Function to preprocess and split datasets into training and testing sets.
- **Custom VAE Implementation**: A VAE class with methods for encoding, decoding, and training.
- **Gradient Computation**: Backward pass implementation to compute gradients and update network parameters.

## Installation

To run the code, you need to have the following Python packages installed:

- `numpy`
- `torch` (for potential future integration)
- `sklearn`
-  `nablaDFT`

You can install the required packages using pip:

```bash
pip install numpy torch scikit-learn
```

Then, install the nablaDFT plugin in order to extract the information for each conformation (Hamiltonian, energy, atom list, etc)
```python
git clone https://github.com/AIRI-Institute/nablaDFT && cd nablaDFT/
pip install .
```



## Usage

Preprocessing
The process_dataset function preprocesses the Hamiltonian dataset, including padding and splitting into training and testing sets.

First, acquire the training dataset from nablaDFT, there are more collections, but here we choose to use the set of 2000 conformations.
```python
wget https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/hamiltonian_databases/train_2k.db
```
Here we pre-process the data, 
```python

def process_dataset(train_dataset, start_idx, end_idx, test_size=0.2, random_state=42):
    # Function body here
```
Choose which features you want to extract and base the maximum padding size on the largest Hamiltonian in the dataset
```python
 Z, R, E, _, H, S, _ = sample
 hamiltonian_shape = H.shape
 pad_width_H = max(0, max_size['H'][0] - H.shape[0])
 pad_height_H = max(0, max_size['H'][1] - H.shape[1])
```

## Model instantiation and training
The nablaVAE class defines the Variational Autoencoder. It includes methods for encoding, decoding, training, and backpropagation. Choose the input shape of your Hamiltonian matrices and the number of dimensions you desire for your VAE latent space

```python
vae = nablaVAE(hamiltonian_input_shape=(64, 64), latent_dim=20)
```

```python
train = hamiltonian_dataset.HamiltonianDatabase("dataset_train_2k.db")
padded_X_train, padded_X_test, y_train, y_test, maxSize= process_dataset(train,start_idx,end_idx)
max_hamiltonian_shape = maxSize['H']
print(max_hamiltonian_shape)
latent_dim = 100
cvae = nablaVAE(max_hamiltonian_shape, latent_dim)
train_model(cvae, padded_X_train,y_train)
```

After training, the nablaVAE generate function can be used to generate samples from the latent space, these samples can be conditioned by changing the loss function (ie conditioning on the energy of the molecule) as well as the features used
```python  
def generate(self, latent_vector):
        reconstructed_output = self.decoder_forward(latent_vector)
        return reconstructed_output
```
