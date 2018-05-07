# NumberCamera #
**Goal**: Develop an application based on Tensorflow to recognize numbers in images with cameras in real time.

**Source**:[Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1312.6082.pdf)

## Requirements ##
1. Python 3.5/Python 2.7
2. TensorFlow
3. h5py

        In Windows:
        > pip3 install h5py
        In Ubuntu:
		$ sudo pip3 install h5py

4. Pillow, Jupyter Notebook etc.
5. Android env

    >Android SDK & NDK (see [https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/README.md))

6. Street dataset 

    >View House Numbers (SVHN [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/))
## Steps ##
1. Clone the source code(git bash environment)

        > git clone git@github.com:Hedlen/SVHNNumber.git
        > cd NumberCamera
        > cd NumberCamera_Based_h5py
2. Download the format 1 dataset based on the above dataset link
3. Extract the data from the file,The data is as follows:

         -data
           -train
			 -1.png
			 -2.pnd
			 -...
			 -digitStruct.mat
		 -data
           -test
			 -1.png
			 -2.pnd
			 -...
			 -digitStruct.mat
		 -data
           -extra
			 -1.png
			 -2.pnd
			 -...
			 -digitStruct.mat
The bounding box information are stored in digitStruct.mat instead of drawn directly on the images in the dataset.Each tar.gz file contains the orignal images in png format, together with a digitStruct.mat file.In our program, we use **h5py.File** to read the data in .mat format.

## Results ##
### **You can view the README.md file in the main folder.** ###
