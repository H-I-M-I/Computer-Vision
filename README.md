# Transfer Learning on CIFAR-10 Dataset

This project applies transfer learning to classify images from the CIFAR-10 dataset, which consists of 10 classes of objects such as airplanes, cars, and cats.

## Features
- Utilizes pre-trained models (e.g., VGG16, ResNet) for feature extraction.
- Fine-tunes the model for improved accuracy on CIFAR-10.
- Demonstrates training, validation, and evaluation steps.

## How It Works
1. **Data Preparation**:
   - Loads the CIFAR-10 dataset and preprocesses images.
   - Normalizes pixel values and applies data augmentation.
   - Dataset is loaded using TensorFlow and split into training and testing sets.

2. **Model Selection**:
   - Uses a pre-trained CNN model without its top classification layer.
   - Adds a custom dense layer for CIFAR-10 classification.
   - Pre-trained model's convolutional layers are frozen initially.

3. **Training & Fine-Tuning**:
   - Initially trains the new layers while keeping the pre-trained layers frozen.
   - Fine-tunes the entire model for better adaptation to the dataset.
   - Model is compiled with categorical crossentropy loss and Adam optimizer.
   - Training progress is monitored using validation accuracy and loss.

4. **Evaluation & Output**:
   - Computes accuracy and loss metrics.
   - Displays confusion matrix and sample predictions.
   - Final accuracy and loss are printed for performance assessment.

## Output
- The model predicts image classes with improved accuracy using transfer learning.
- Outputs visualizations such as training curves and classification results.
- Predictions are displayed alongside true labels for better understanding.
