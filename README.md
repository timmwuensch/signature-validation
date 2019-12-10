# signature-validation
A CNN based classifier to evaluate handwritten signatures towards legal validity.

A lot of companies have to deal with hand signed declarations of assignment of the customer.
All signatures are subjected to specific requirements which make them legally valid. 
These requirements regarding the format (arrangement of the name parts), the readability or the overall quality of the signature. 
The result of this project is a modular system that classifies the validity of a signature based on the format and the equality 
of the first recognized letter in the signature with the first letter of the given surname. The system consists of the two components 
Format Recognition and Letter Recognition. Both are using Convolutional Neural Networks trained on different data. 
The CNN of the Format Recognition Component achieves 97.1 percent accuracy on validation data. The CNN of the Letter Recognition 
Component scores 99.04 percent accuracy on validation data. The entire system achieves an overall success rate of 74.1 percent on 
the basis of 1,000 different signatures. So far, the system can be used as a validator in a production environment. 


#### Table of contents
1. [What is this code used for?](#what-is-this-code-used-for)
2. [How does it solve the problem?](#how-does-it-solve-the-problem)
3. [Whats is the implementation?](#whats-is-the-implementation)
4. [How to use this code?](#how-to-use-this-code)


## What is this code used for?
The published code is a short extract from a recent Machine Learning project. It was developed in cooperation with the Business Intelligence Department and is open for further improvements and research. 

Please be aware of the fact, that I had to exclude the concrete train data and all samples from the project due to privacy issues. 

### Problem
In some legal-tech business cases it is required to have a digital handwritten signature from the customer to validate his identity. 
There are special requirements to a handwritten signature to make them legally valid and usable (see requirements). 
Unfortunately, not all signatures given by customers are valid and usable and that could be a big problem in further operations.

The idea was to develop a Deep Learning (DL) based classifier that evaluates each given signature towards the following requirements.

### Requirements
On the basis of german law and internal papers there are a couple requirements to a handwritten signature:
- The signature has to include the full family name of the person
- At least the first letter of the family name has to be recognizable and readable
- The signature can include the first names (even abbreviated or written out)
- Signatures including single lines, circles, crosses or other symbols are not valid

### Further Specifications
In the context of this prototype there are some further specifications:
- The signature has to fit the special format of an abbreviated first name and a full family name (i.e. "M. Mustermann")
- The signature exists as a two-dimensional grayscale image with a black line on white background 
- The signature has to consist of letters out of the latin alphabet 

## How does it solve the problem?
The developed classifier is divided into two main components. For further details please take a look into the implementation chapter. 
### Format Recognition
The Format Recognition Component is a single classifier which analyses the format of a signature. In detail, it differs between two
classes of formats:

- Format 1: abbreviated first name and a full family name (i.e. "M. Mustermann")
- Format 0: all other existing formats

This classifier was trained on a dataset of 10,000 labeled signature images (90% train data, 10% validation data) and was expanded with
the help of Data Augmentation (horizontal and vertical shift).

To find the optimal CNN network structure, I did a small grid search over 64 fully self-trained architectures and 16 partly transfer-learned
architectures. 

The best examined model consists of 13 layer as shown in the following figure:

*Figure Model A11*

The model was trained over 300 epochs with a batch size of 16 and the Stochastic Gradient Decent optimizer. The used loss function was
Categorical Crossentropy and all layers were activated with Rectified Linear Unit (ReLU), except for the output layer which is activated with Softmax.

The following figure shows the accuracy of the model in relation to the validation data (orange) and to the train data (blue) over 300 train epochs.
 
*Figure Accuracy of A11(SGD)*

In addition to the accuracy diagram let's have a look to the loss development as well. The notation equals the figure above. 

*Figure Loss of A11(SGD)*

In both figures we can see some kind of overfitting symptoms after 180 epochs of training. The best accuracy value occurs in epoch 284 with 97.1% accuracy 
rate on the validation set. The state of epoch 284 is saved by a checkpoint during the training and is now used as the main model of the Format Recognition classifier.

For further work on this component it would be very promising to expand the train dataset with images or to intensify the Data Augmentation. 
This leads to a more robust classifier. To improve the architecture of the model, it would be promising to extend the parameter space of the grid search in order
to evaluate even more architectures. Furthermore, we can use a random search instead of a grid search or use special transfer models.   


### Letter Recognition
The Letter Recognition Component is a single classifier which recognizes single capital letters of the latin alphabet. Here we differ between 
26 capital letters and an additional class for not recognizable samples (labeled with "?").

It was trained on a dataset from Kaggle ([A-Z Handwritten Alphabets in .csv format](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format))
including 372,451 samples of labeled letter images. This dataset was expanded with 10,000 randomly created samples for the additional class.
The entire dataset was unbalanced and further expanded with Data Augmentation (horizontal and vertical shift, zoom and rotation).

The architecture was taken from Kaggle as well (see the solution of [CodHeK](https://www.kaggle.com/codhek/cnn-using-keras-using-csv-accuracy-99-82)).
It was trained over 20 epochs with a batch size of 64 and optimized with Adam optimizer. The concrete model is visualized in the following figure:

*Figure Model Kaggle*

This model achieves 99.04% accuracy rate on the validation set after 20 epochs of training.

### Segmentation
To analyze the beginning capital letters of the signature parts the correct segmentation of these party is one of the main 
challenges. To serve correct samples to the Letter Recognition, the segmentation process happens in the following order:

1. Reducing the image information to Region of Interest (ROI)
2. Invert the color space of the image to white line on black background
3. Analyze the the sum of pixels for each column on the horizontal axis
4. Find spaces and segments in the sequence of sums
5. Define valid letter segments with the help of detected spaces, segments and average length
6. Cut the letter segments out of the image and rescale it to 28x28 square images to feed them into the Letter Recognition classifier

*Figure to visualize the process*

To improve the segmentation process it would be very promising optimize the in-code-threshold (i.e. for the average letter segment length) or to normalize 
the signature images (i.e. with shear and rotation parameters) due to handwriting issues. Furthermore, it is possible to train an additional CNN
to find the locations of the capital letter segments.

## Whats is the implementation?
As already mentioned, the system is divided into two sub components which are used and called in the main script [validate_signature.py](https://github.com/timmwuensch/signature-validation/blob/master/main/validate_signature.py).
The following figure gives a short system overview:

*Figure System overview*

The evaluation is based on the results of both components. A signature is valid if the format equals Format 1 and if at least 
one of the two capital letters is recognized by the Letter Recognition and equals a character of the real name string. To visualize 
this decision process please have a look on the following flowchart diagram:

*Flowchart*

## How to use this code?
To use this code please copy this repository and install the requirements. You can find them in `environment.yml` and `requirements.txt` to easily create a conda environment.

To create a conda environment just run the following line in your anaconda prompt:

`cond env create -f environemnt.yml`

To install the requirements via pip:

`pip install -r requirements.txt`

<hr>

In your Python interpreter you now just have to call the method in [validate_signature.py](https://github.com/timmwuensch/signature-validation/blob/master/main/validate_signature.py). This method requires a signature image in numpy format 
(i.e. via OpenCV method imread) and the full name of the person in sting format. 

```
path = '/path/to/signature/image' 
image = cv2.imread(path, 0)
name = 'Max Mustermann' 

result = validate_signature(image, name)
``` 

