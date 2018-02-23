# Visual Speech Processing using a LSTM

The LSTM implementation of our project

## Getting Started

Clone the repo using 
```
git clone https://github.com/drylbear/visualspeechplayoffs.git
```
Create a new enviorment and install the dependencies in the requierments.txt file. Also create the following folder structure for the LSTM project:

lstm/
├──original/
│  ├──01M
│  ├──02F
│  ├──...
│  └──59F	
├──data/	
│  ├──logs/
│  ├──sequence/
│  ├──train/
│  │  ├──01M
│  │  ├──02F
│  │  ├──...
│  │  └──59F
│  └──resnet/
│     ├──01M
│     ├──02F
│     ├──...
│     └──59F
└──predictor/

The TCD-TIMIT dataset should be in the ´original´ folder, with a subfolder for each speaker. The data folder contains all the data needed for training. In the ´train´ folder all the cropped and processed images for feature extraction are stored. In the ´resnet´ folder, all the images for features extraction using the ResNet50 are storred.



### First time running

For the first time running, be sure to call the function ´extract_files()´ in the main.py file. This will calculate all the features necessary for running the training data. If everything went smoothly, comment out the ´extract_files()´ and uncomment the ´main_best_params()´ function in the main.py file and run it again for training the network.

