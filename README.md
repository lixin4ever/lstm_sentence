# lstm_sentence
A simple implmentation of LSTM for sentence classification based on [Keras](https://keras.io/), an excellent python library for deep learning researcher. 

# Pre-install
* [python 2.7](https://www.python.org/downloads/)
* [Keras](https://keras.io/) (>=1.0.1)
* [Theano](http://deeplearning.net/software/theano/) (>=0.8.2)
* [numpy](http://www.numpy.org/) (>=1.10.4)
* [scikit-learn](http://scikit-learn.org/stable/) (>=0.17.1)
* [pandas](http://pandas.pydata.org/) (0.17.0)

##### For simplication, you can just download [Anaconda](https://www.continuum.io/), a python superset that pre-installs large number of 3rd-party libraries.

# Data preprocessing
The preprocessing module is kept same with [Kim Yoon's code](https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py) and you can use the program to process your own datasets.
    
    python preprocess.py

A **.p** file will be created which contains the input information of dataset. 

# Run the program
    python lstm_sentence.py
If GPU is availabel on your machine, the program will be executed on GPU by default. You can use different gpu to run the code by specifying **gpu** parameter on the command line:

    THEANO_FLAGS=mode=FAST_RUN,device=#YOUR_GPU_NUMBER# python lstm_sentence.py

