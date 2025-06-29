# Deep Learning Model for Nanoparticle Detection

## Overview

This project implements nanoparticle detection using a Variational Autoencoder (VAE) combined with contrastive loss and clustering strategies. The goal is to enable unsupervised particle discovery and improve detection performance through latent space refinement.

---

## Project Directory Overview

This repository is organized to support **training, evaluation, and visualization** for nanoparticle detection using a Variational Autoencoder (VAE) with contrastive loss and clustering techniques.

### **Top-Level Files and Directories:**

| Path                      | Description                                                          |
|---------------------------|----------------------------------------------------------------------|
| `config/`                 | Contains configuration JSON files for training and evaluation setup. |
| `examples/`               | Example jupyter notebooks for running clustering and evaluation.               |
| `particle_detection/`     | Main source code including data loading, VAE model, training, evaluation, and clustering utilities. |
| `.gitattributes`          | Configures Git LFS to handle large model checkpoint files (`.pth`).   |
| `.gitignore`              | Specifies files and directories to ignore (e.g., outputs, cache files, visualization artifacts). |
| `README.md`               | This documentation file.                                             |
| `requirements.txt`        | Lists Python package dependencies required to run the project.       |

---

## **Notes:**
- The project follows a **modular structure**, separating core model logic, utilities, and example usage for easier development and testing.

---

## How to Use

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/blah-buttery/particle-detection.git
   ```
2. Navigate into the project directory:
    ```bash
    cd particle-detection
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

### Training the model

To train the variational autoencoder model, run the following command:

```bash
python3 -m particle_detection.vae.train_model --config config/training_config.json
```

This will save the model in the `saved_models` directory.

> **Note:** Any parameters specified in the config file can be overridden directly from the command line.  
> For example:  
> ```bash
> python3 -m particle_detection.vae.train_model --config config/training_config.json --num_epochs 100 --batch_size 32
> ```

> **Tip:** Use the `--help` flag with any script to see available arguments.  
> Example:  
> ```bash
> python3 -m particle_detection.vae.train_model --help
> ```

> **Note:** GPU acceleration is supported. For multi-GPU training, use the `--is_ddp` flag.

---

### Evaluating the model

To evaluate the variational autoencoder model, run the following command:

```bash
python3 -m particle_detection.vae.eval --config config/eval_config.json
```

> **Note:** As with training, evaluation parameters from the config file can be overridden using command-line arguments.

---

## License

This project is not open-source.

© 2025 Todd A. Lee. All rights reserved.

This code is published for educational and portfolio purposes only.  
You may not copy, modify, distribute, or use this project in any form  
without express written permission.
