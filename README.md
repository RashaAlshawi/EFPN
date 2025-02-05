
# EFPN Segmentation Project
This project is focused on building an efficient and flexible segmentation model using the Efficient Feature Pyramid Networks (EFPN) architecture. The model is trained on a dataset of images and labels, and the project includes scripts for training, evaluating, and testing the model, along with necessary utilities and a Jupyter notebook for interactive experimentation.


## Requirements
- TensorFlow
- NumPy
- scikit-learn

## How to Run

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:

    cd e_fpn

   ```
   python train.py
   ```

3. Evaluate the model:
   ```
   python eval.py
   ```
# File Structure

- model.py: Contains the model architecture and the function to build the EFPN model (build_fpn).
- train.py: Defines the training loop, including data loading, model compilation, and training.
- eval.py: Handles model evaluation, including computing metrics like F1 score and IoU.
- utils.py: Contains utility functions for metrics calculation (e.g., F1 score, IoU).
- dataset.py: Defines functions for loading and processing the dataset.
- notebooks/EFPN.ipynb: A Jupyter notebook for training and evaluating the model interactively, especially useful when running in Google Colab.

