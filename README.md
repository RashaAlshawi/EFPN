
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
## File Structure

### 1. **Main Project Files**
- **`e_fpn/`**: Contains the core code for building and managing the EFPN architecture.
  - **`model.py`**: Contains the model architecture and the function to build the EFPN model (`build_fpn`).
  - **`train.py`**: Defines the training loop, including data loading, model compilation, and training.
  - **`eval.py`**: Handles model evaluation, including computing metrics like F1 score and IoU.
  - **`utils.py`**: Contains utility functions for metrics calculation (e.g., F1 score, IoU).
  - **`dataset.py`**: Defines functions for loading and processing the dataset.

### 2. **Notebooks**
- **`EFPN.ipynb`**: A Jupyter notebook for training and evaluating the model interactively, especially useful when running in Google Colab.

### 3. **Dataset Folder**
The dataset consists of 9 classes representing different deficiencies in culverts and sewer pipes, with pixel-level annotations. Currently, the dataset is not publicly available, but we are in the process of obtaining the necessary permissions to make it accessible.

### 4. **Data Imbalance Folder** This directory contains the code for:
- **`DataAugmentation.ipynb/`**: handling data imbalance through augmentation techniques.
- **`ClassDecomposition_DataAugmentation.ipynb/`**: It includes scripts for applying class decomposition strategies and data augmentation while training the EFPN model to address class imbalance.

### 5. **experiments Folder**

