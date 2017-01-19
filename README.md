# neon_course

This repository contains several [jupyter](http://jupyter.org/) notebooks to help users learn to use [neon](https://github.com/NervanaSystems/neon), our deep learning framework. For more information, see our [documentation](http://neon.nervanasys.com/docs/latest/index.html) and our [API](http://neon.nervanasys.com/docs/latest/api.html).

Note: this version of the neon course is synchronized to work with neon v1.8.1, and some notebooks require installation of the aeon dataloader. For install instructions, see the [neon](http://neon.nervanasys.com)  and [aeon](http://aeon.nervanasys.com) documentation. See neon_course v1.2 for a version of this repository that works with neon version 1.2.

The jupyter notebooks in this repository include:

### 01 MNIST example

Comprehensive walk-through of how to use neon to build a simple model to recognize handwritten digits. Recommended as an introduction to the neon framework.

### 02 Fine-tuning

A popular application of deep learning is to load a pre-trained model and fine-tune on a new dataset that may have a different number of categories. This example walks through how to load a VGG model that has been pre-trained on ImageNet, a large corpus of natural images belonging to 1000 categories, and re-train the final few layers on the CIFAR-10 dataset, which has only 10 categories.

### 03 Writing a custom dataset object

neon provides many built-in methods for loading data from images, videos, audio, text, and [more](http://neon.nervanasys.com/docs/latest/loading_data.html). In the rare cases where you may have to implement a custom dataset object, this notebooks guides users through building a custom dataset object for a modified version of the [Street View House Number](http://ufldl.stanford.edu/housenumbers/) (SVHN) dataset. Users will not only write a custom dataset, but also design a network to, given an image, draw a bounding box around the digit sequence.

### 04 Writing a custom activation function and a custom layer

This notebook walks developers through how to implement custom activation functions and layers within neon. We implement the Affine layer, and demonstrate the speed-up difference between using a python-based computation and our own heavily optimized kernels.

### 05 Defining complex branching models

When simple sequential lists of layers do not suffice for your complex models, we present how to build complex branching models within neon.

### 06 Deep Residual network on the CIFAR-10 dataset

In neon, models are constructed as python lists, which makes it easy to use for-loops to define complex models that have repeated patterns, such as [deep residual networks](https://arxiv.org/abs/1512.03385). This notebook is an end-to-end walkthrough of building a deep residual network, training on the CIFAR-10 dataset, and then applying the model to predict categories on novel images.

### 07 Writing a custom callback

[Callbacks](http://neon.nervanasys.com/docs/latest/callbacks.html) allow models to report back to users its progress during training. In this notebook, we present a callback that plots training cost in real-time within the jupyter notebook.

### 08 Detecting overfitting

Overfitting is often encountered when training deep learning models. This tutorial demonstrates how to use our visualization tools to detect when a model has overfit on the training data, and how to apply `Dropout` layers to correct the problem.

For several of the guided exercises, answer keys are provided in the `answers/` folder.

### 09 Sentiment Analysis with LSTM

These two notebooks guide the user through training a recurrent neural network to classify paragraphs of movie reviews into either a positive or negative sentiment. The second notebook contains an example of inference with a trained model, including a section for users to write their own reviews and submit to the model for classification. 

### Setting up notebooks on remote machines

Some of these notebooks require access to a Titan X GPU. For full instructions on launching a notebook server that one could connect to from a different machine, see http://jupyter-notebook.readthedocs.io/en/latest/public_server.html. For a simple setup, first generate a configuration file:

```
$ jupyter notebook --generate-config
```

In your `~/.jupyter directory`, edit the notebook config file, `jupyter_notebook_config.py` and edit the following lines:

```
c.NotebookApp.ip = '*'

c.NotebookApp.port = 8888
```

Save your changes and launch the jupyter notebook:

```
$ jupyter notebook
```

From a separate machine, open your browser and point to `https://[server address]:8888` to connect to the jupyter notebook.

## Nervana Cloud

The [Nervana Cloud](https://www.cloud.nervanasys.com/login) includes an interactive mode to launch jupyter notebooks on our Titan X GPU servers. If you have cloud credentials, launch an interactive session with the `ncloud interact` command.

For more information, see: http://doc.cloud.nervanasys.com/docs/latest/interact.html


