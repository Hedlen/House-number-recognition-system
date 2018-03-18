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

        > git clone git@github.com:KarlXiao/NumberCamera.git
        > cd NumberCamera
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
## Versions ##
### Using TFRecord data processing to do the training ###
#### Usage ####
1. Convert to TFRecords format

        > python convert_to_tfrecords.py
2. Train

		> python train.py 
3. Retrain if you need

		> python train.py --restore_checkpoint ./logs/train/latest.ckpt
4. Evaluate

		> python eval.py
5. Visualize

		> tensorboard --logdir=./logs		
6. (Optional) Perform actual testing of other images

		Open 'Test_sample.ipynb' in Jupyter notebook
		Open 'Test_outside_sample.ipynb' in Jupyter notebook
		Above all,you can see the results you want.
### Using h5py data processing to do the training ###
1. Convert to h5py format

		> cd NumberCamber_Based_h5py
		Make sure the data folder is under the path.
		> python convert_to_h5py.py
2. (Optional)Try to read h5py file and take a glance at original images.

		Open "read_h5py_test.ipynb"
3. Train

		> python train.py
4. Evaluate

		> python eval.py

5. Visualize

		> tensorboard --logdir=./logs
## Results ##
### Graph ###
<img src="./images/graph.png" width=800 height=800></img>
### Accuracy ###
![image_accuracy](./images/accuracy.png)

93.33% correct rate on the validation dataset

87.43% correct rate on the test dataset (*The correct rate is lower than the correct rate of the original paper, and the improvement of the model continues.*)

### Loss ###
![image_Loss](./images/loss.png)

#### Practical testing ####
<img src="./images/58.png" width=300 height=320 alt="image"></img>

<img src="./images/8888.png" width=300 height=320 alt="image"></img>

In the figure, the number 10 represents empty.

