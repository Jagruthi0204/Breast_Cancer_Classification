# Breast_Cancer_Classification

Breast Cancer Classification is an object detection model built on Keras implementation of FRCNN for mammograms to detect breast cancer in women.

![alt text](https://github.com/Jagruthi0204/Breast_Cancer_Classification/blob/main/Images/frcnn.PNG)

![alt_text](https://github.com/Jagruthi0204/Breast_Cancer_Classification/blob/main/Images/Model%20Architecture.jpeg)

The model is aimed to provide the risk factor associated with the lesion and alert the user quickly for furthur actions. It also draws a bounding box around the lesion to provide the exact location of the Breast Cancer in the mammograms.


## Getting Started

### Python Libraries:

The python libraries required for the model are Tensorflow, Numpy, h5py, Keras, opencv-python, sklearn

#### Installation:

    pip install numpy h5py opencv-python scikit-learn Keras==2.3.1 Tensorflow==1.14

### Trained Model Weights:

Trained model weights can be downloaded from the following link: https://drive.google.com/file/d/1xhFNI2b7jhpeL3sYhiZZVbHBF8rRGuEu/view?usp=sharing

### Datasets:

The datasets used for training and testing the model are CBIS-DDSM & MAIS datasets which can be downloaded from the following links:

  https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
  
  http://peipa.essex.ac.uk/info/mias.html

### Training Process:
-   Copy pretrained weights for resnet50 (resnet50_weights_tf_dim_ordering_tf_kernels.h5) in the main directory.
- train_frcnn.py can be used to train the model. To train the data, it must be in PASCAL VOC format. Data can be converted to PASCAL VOC format using any image labelling tools.
- The train_path variable need to be changed to the data location path and output_weights_path to the location where weights needs to be saved.
-  The num_epochs is defaulted to 50 but can be changed as per requirement.
-   If there are any pre-trained weights available, the input_weights_path variable can be initiated with the weights folder path.
- To run the file:

      python train_frcnn.py

### Testing Process:
- Copy trained model(model_frcnn.hd5) and config.pickle file in the main directory.
- The test_path variable need to be changed to the testing data location path.
- To run the file:

      python test_frcnn.py


