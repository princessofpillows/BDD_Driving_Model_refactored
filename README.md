# Description

The [Berkeley Deep Drive](https://github.com/gy20073/BDD_Driving_Model) neueral network rewritten with modern TensorFlow to be more developer friendly

[Visit their website](https://deepdrive.berkeley.edu/) for more information on the dataset and project

#### Requirements

* a working Python 3 development environment
* [pip3](https://pip.pypa.io/en/latest/installing.html) to install Python dependencies (must be latest version -> pip install --upgrade pip)
* [pipenv](https://github.com/pypa/pipenv) to manage dependencies
* [driving dataset](https://drive.google.com/drive/folders/1z6hjT9JMrC2w30jyyxAbpbLgFEKpnsw2?usp=sharing)
* [trained alexnet weights](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) for optimal accuracy

#### Pipfile Requirements

* [TensorFlow](https://www.tensorflow.org/install/) for creation of graph and most functionality
* [openCV 3.0+](https://pypi.python.org/pypi/opencv-python) for image processing
* [tqdm](https://pypi.python.org/pypi/tqdm) to view progress throughout runtime
* [scikit-image](http://scikit-image.org/docs/dev/install.html) for image processing
* [h5py]("http://docs.h5py.org/en/latest/build.html") for data storage

pipenv will install all of the Pipfile required packages.

To do so, run the following command:
```
pipenv install
```

#### Dataset

The dataset can be found [here](https://drive.google.com/drive/folders/1z6hjT9JMrC2w30jyyxAbpbLgFEKpnsw2?usp=sharing) and is property of Berkely Deep Drive
