# neon_course

This repository contains several ipython notebooks to help users learn to use neon. These include:

* MNIST example
* Example of fine-tuning a pre-trained VGG network on the CIFAR-10 dataset
* Exercise in writing your own dataset object for the SVHN dataset
* Writing a custom activation function and a custom layer
* Defining complex branching models
* Deep Residual network on the CIFAR-10 dataset
* Writing a custom callback
* Using out visualization tools to detect and correct overfitting.

For several of the guided exercises, answer keys are provided in the answers folder.

## Setting up notebooks on remote machines

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

From a separate machine, open your browser and point to `https://[server address]:8888` to connect to the ipython notebook.



