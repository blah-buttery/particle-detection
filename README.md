# Particle Detection

This repository contains code for training and evaluating deep learning models for nanoparticle detection.

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
## Project Directory Overview

This repository is organized to support **training, evaluation, and visualization** for nanoparticle detection using a Variational Autoencoder (VAE) with contrastive loss and clustering techniques.

### **Top-Level Files and Directories:**

| Path                      | Description                                                          |
|---------------------------|----------------------------------------------------------------------|
| `config/`                 | Contains configuration JSON files for training and evaluation setup. |
| `docs/`                   | Contains project documentation, including build process notes, weekly reports, final report, and presentation materials.                              |
| `examples/`               | Example scripts for running clustering and evaluation.               |
| `particle_detection/`     | Main source code including data loading, VAE model, training, evaluation, and clustering utilities. |
| `.gitattributes`          | Configures Git LFS to handle large model checkpoint files (`.pth`).   |
| `.gitignore`              | Specifies files and directories to ignore (e.g., outputs, cache files, visualization artifacts). |
| `README.md`               | This documentation file.                                             |
| `requirements.txt`        | Lists Python package dependencies required to run the project.       |
| `setup.py`                | Package setup script for installation and organization.              |

---

## 📄 Documentation

The `docs/` directory contains supporting documentation for the project, including:

- **Weekly Reports:** Summaries of weekly progress, tasks completed, challenges faced, and plans for the following week.
- **Final Report:** Comprehensive overview of the project, including methodology, results, evaluation metrics, and conclusions.
- **PowerPoint:** Presentation slides for the final project demonstration, covering goals, design, implementation, testing, and lessons learned.


This section serves as a living document of the project's progress, decisions, and results.

## **Notes:**
- The project follows a **modular structure**, separating core model logic, utilities, and example usage for easier development and testing.

## Training the model
To train the variational autoencoder model, run the following command:
```bash
python3 -m particle_detection.vae.train_model --config config/training_config.json
```
This will save the model in the saved directory 

To evaluate the variational autoencoder model, run the following command:
```bash
python -m particle_detection.vae.train_model --num_epochs 10 --batch_size 128 \            
    --learning_rate 0.01 --data_dir ./patches_dataset --image_size 128 128
```
