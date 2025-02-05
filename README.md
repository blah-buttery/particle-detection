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
4. Install Git LFS (For saved models)
    - macOS: ```bash sudo apt intall-lfs ```
    - Linux: ```bash brew install git-lfs ```
5. After cloning, Git LFS will automatically handle large files like saved models. If needed, run:
    ```bash
    git lfs pull
    ```

## Training the model
To train the model, run the following command:
```bash
python -m particle_detection.autoencoder.train_model \
    --num_epochs 10 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --dataset_dir path/to/data \
    --model_path saved_models/<model-file>
```

Training on Multiple GPUs
```bash
python -m particle_detection.autoencoder.train_model \
    --num_epochs 10 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --dataset_dir path/to/data \
    --model_path saved_models/<model-file> \
    --is_ddp
```
This command trains the model using Distributed Data Parallel (DDP).

## Evaluating the model
To evaluate the model, run:
```bash
python -m particle_detection.autoencoder.evaluate \
    --model_path saved_models/model.pth \
    --dataset_dir path/to/data
```
## Saved Models
This project uses Git LFS (Large File Storage) to manage large files like saved models in the saved_models/ directory. If you encounter issues with saved models, ensure Git LFS is installed and run:
```bash
git lfs pull
```

### Adding or Updating Saved Models
To add or update models in the saved_models/ directory:
1. Save the model in the directory.
2. Track the file type with Git LFS:
    ```bash
    git lfs track "*.pth"
    ```
3. Commit and push:
    ```bash
    git add saved_models/<model-file>
    git commit -m "Add new model"
    git push origin main
    ```

## Notebooks
The notebooks directory contains Jupyter Notebook examples for:

1. Running experiments: Pre-configured examples to train or evaluate the model on sample datasets.
2. Viewing results: Visualizations of original and reconstructed images from the model, along with clustering results.

These notebooks are designed to provide a quick and interactive way to understand and test the functionality of the project.

## Contributing

1. Fork the repository
2. Create a new branch:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Add your feature description"
    ```
4. Push to the branch
    ```bash
    git push origin feature/your-feature-name
    ```
5. Open a pull request 
