# Particle Detection

This repository contains code for training and evaluating deep learning models for nanoparticle detection.

## How to Use

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/particle-detection.git
   ```
2. Navigate into the project directory:
    cd particle-detection

3. Install dependencies
    pip install -r requirements.txt

## Training the model
To train the model, run the following command:
```bash
python -m particle_detection.autoencoder.train_model \
    --num_epochs 10 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --dataset_dir path/to/data \
    --model_path saved_models/autoencoder_small.pth
```
## Evaluating the model
To evaluate the model, run:
```bash
python -m particle_detection.autoencoder.evaluate \
    --model_path saved_models/autoencoder_small.pth \
    --dataset_dir path/to/data
```
Notes
The saved_models directory contains pre-trained models for quick experimentation.
To keep datasets or sensitive data local, ensure they are not pushed to the repository.

## Contributing

1. Fork the repository
2. Create a new branch:
    git checkout -b feature/your-feature-name
3. Commit your changes:
    git commit -m "Add your feature description"
4. Push to the branch
    git push origin feature/your-feature-name
5. Open a pull request 
