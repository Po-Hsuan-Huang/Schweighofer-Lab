# Efficient movement representation by embedding dynamic movement premitives in deep autoencoders #

#### Author:  Po-Hsuan Huang 06/01/2018  

#### Email : pohsuanh_at_usc.edu 

This is my rotation project with Prof. Schweighofer trying to replicate the model proposed by Nutan Chen at. al. in their Efficient movement representation by embedding dynamic movement premitives in deep autoencoders (2015 IEEE-RAS). https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7363570

The model is able to reconstruct human movements by embedding dynamice movement primitives in latent space of the variational autoencoder with variational bayes fileters (Maximilian Karl et al. ICLR 2017)https://arxiv.org/abs/1605.06432.

The movement dataset comes from Master Motor Map - Framework and Toolkit for Captuing. Representing, and Reproducing Human Motion on Humanoid Robots (Oemer Terlemez et al.) https://ieeexplore.ieee.org/document/7041470/.

The project aims to use the powerful generative model to help rehabilitate patients.

### What is this repository for? ###

These folder contains all scripts written during my rotation

The files include dynamic motor primitive example based on Matlab code from Prof. Stefan Schaal's websit

The files include stacked auto-encoders and variational auto-encoders for NMIST dataset and CIFAR10 dataset

### What is not done yet ? ###

* Writing the data pipeline interface from Master Motor Map for the auto-encoder, and test able to generate stationary movements

* Integrating Dynamic Movement Primitive to the variational autoencoder, and test able to generate dynamice movments 

* Integrate deep Bayesian filters to the variational autoencoder, and test able to generate movement transitions

* Finally, design better regularizers to learn more efficient ebeddings in latent space

### How do I get set up? ###

Script Dependencies

* Linxu operation system (Ubuntu) 

* Python3

* Matplotlib

* Numpy

* Tensorflow-gpu

* Spyder

# Installation Guide for running Tensorflow on GPU

Detailed instruction : https://medium.com/@taylordenouden/installing-tensorflow-gpu-on-ubuntu-18-04-89a142325138

## Install proper version of Nvidia GPU driver

Depending on the GPU card you have, update your GPU driver and reboot your computer.

For example, install nvidia-390 for Nvidia 1080Ti GPU.https://tecadmin.net/install-latest-nvidia-drivers-ubuntu/

You must have Nvidia GPU in order to run Tensorflow on GPU. Otherwise you can run Tensorflow on CPU. 

## Install with Pip

Download and install CUDA Toolkit https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
    
Download and install CuDNN 

Then :

    sudo apt-get install pip
	pip install python
	pip install matplotlib
	pip install numpy
	pip install tensorflow-gpu
	pip install spyder
	
## Install in virtual environment such as Anaconda

Virtual environemtn prevents dependencies conflicts of different softwares. For more information ask Google.

Download Anaconda and install it folloing the installation guide on https://conda.io/docs/user-guide/install/index.html

after install Anaconda, create an virtual environment 'tensorflow' in Python3 :	
    
	conda create -n tensorflow python = python3
	source activate tensorflow
	conda install matplotlib, spyder
	conda install tensorflow-gpu
	
# Installation Guide for running Tensorflow on CPU

## With Pip

    sudo apt-get install pip
	pip install python
	pip install matplotlib
	pip install numpy
	pip install tensorflow
	pip install spyder
	
## In virtual environment such as Anaconda

download Anaconda and install it folloing the installation guide on https://conda.io/docs/user-guide/install/index.html

after install Anaconda, create an virtual environment 'tensorflow' in Python3 :	
    
	conda create -n tensorflow python = python3
	source activate tensorflow
	conda install matplotlib, spyder
	conda install tensorflow

# How to run tests

Each script is independent from each other. So each of them can be executed independently by typing in your terminal :

    $ python the/path/to/the/script.py

Or run script in Spyder editor by typing in your terminal  :
	  
	  $ spyder

In Spyder3, open the respective file by clicking the open file icon on the toolbar in the API
  
However, it is recommended to run the code in Python edictor such as Spyder3 https://pythonhosted.org/spyder/ to produce images correctly.

* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)
