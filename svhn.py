# import some useful packages
from neon.data import NervanaDataIterator
import numpy as np
import cPickle
import os
from neon.callbacks.callbacks import Callbacks
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, Conv, Pooling, Linear, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp, StepSchedule
from neon.transforms import Rectlin, Logistic, CrossEntropyMulti, Misclassification, SumSquared, SumSquaredMetric
from viz_callback import CostVisCallback
from neon.backends import gen_backend
from bokeh.layouts import row, gridplot, column
from bokeh.models import CustomJS, ColumnDataSource, Slider, Button, RadioGroup, WidgetBox
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, gridplot, output_file, show
import matplotlib
import matplotlib.pyplot as plt

class SVHN(NervanaDataIterator):

    def __init__(self, X, Y, lshape):

        # Load the numpy data into some variables. We divide the image by 255 to normalize the values
        # between 0 and 1.
        self.X = X / 255.
        self.Y = Y
        self.shape = lshape  # shape of the input data (e.g. for images, (C, H, W))

        # 1. assign some required and useful attributes
        self.start = 0  # start at zero
        self.ndata = self.X.shape[0]  # number of images in X (hint: use X.shape)
        self.nfeatures = self.X.shape[1]  # number of features in X (hint: use X.shape)

        # number of minibatches per epoch
        # to calculate this, use the batchsize, which is stored in self.be.bsz
        self.nbatches = self.ndata / self.be.bsz


        # 2. allocate memory on the GPU for a minibatch's worth of data.
        # (e.g. use `self.be` to access the backend.). See the backend documentation.
        # to get the minibatch size, use self.be.bsz
        # hint: X should have shape (# features, mini-batch size)
        # hint: use some of the attributes previously defined above
        self.dev_X = self.be.zeros((self.nfeatures, self.be.bsz))
        self.dev_Y = self.be.zeros((self.Y.shape[1], self.be.bsz))

    def reset(self):
        self.start = 0

    def __iter__(self):
        # 3. loop through minibatches in the dataset
        for index in range(self.start, self.ndata, self.be.bsz):
            # 3a. grab the right slice from the numpy arrays
            inputs = self.X[index:(index + self.be.bsz), :]
            targets = self.Y[index:(index + self.be.bsz), :]

            # The arrays X and Y data are in shape (batch_size, num_features),
            # but the iterator needs to return data with shape (num_features, batch_size).
            # here we transpose the data, and then store it as a contiguous array.
            # numpy arrays need to be contiguous before being loaded onto the GPU.
            inputs = np.ascontiguousarray(inputs.T)
            targets = np.ascontiguousarray(targets.T)

            # here we test your implementation
            # your slice has to have the same shape as the GPU tensors you allocated
            assert inputs.shape == self.dev_X.shape, \
                   "inputs has shape {}, but dev_X is {}".format(inputs.shape, self.dev_X.shape)
            assert targets.shape == self.dev_Y.shape, \
                   "targets has shape {}, but dev_Y is {}".format(targets.shape, self.dev_Y.shape)

            # 3b. transfer from numpy arrays to device
            # - use the GPU memory buffers allocated previously,
            #    and call the myTensorBuffer.set() function.
            self.dev_X.set(inputs)
            self.dev_Y.set(targets)

            # 3c. yield a tuple of the device tensors.
            # X should be of shape (num_features, batch_size)
            # Y should be of shape (4, batch_size)
            yield (self.dev_X, self.dev_Y)


def train_model_button():
    print "hello"
    # train_model(slider, fig=fig, handle=fh, train_source=train_source, val_source=val_source)



def train_model(learning_inputs,
                fig=None, handle=None, train_source=None, val_source=None):


    be = gen_backend(batch_size=128, backend='gpu')
    be.enable_winograd=0

    train_set = SVHN(X=svhn['X_train'], Y=svhn['y_train'], lshape=(3, 64, 64))
    test_set = SVHN(X=svhn['X_test'], Y=svhn['y_test'], lshape=(3, 64, 64))

    init_norm = Gaussian(loc=0.0, scale=0.01)

    # set up model layers
    convp1 = dict(init=init_norm, batch_norm=True, activation=Rectlin(), padding=1)

    layers = [Conv((3, 3, 32), **convp1),  # 64x64 feature map
              Conv((3, 3, 32), **convp1),
              Pooling((2, 2)),
              Dropout(keep=.5),
              Conv((3, 3, 64), **convp1),  # 32x32 feature map
              Conv((3, 3, 64), **convp1),
              Pooling((2, 2)),
              Dropout(keep=.5),
              Conv((3, 3, 128), **convp1),  # 16x16 feature map
              Conv((3, 3, 128), **convp1),
              Linear(nout=4, init=init_norm)]  # last layer good for bbox

    # use SumSquared cost
    cost = GeneralizedCost(costfunc=SumSquared())
    learning_rates = [10**(-1 * (5-f)) for f in learning_inputs]
    # learning_rates = list(base_lr * np.cumprod(factors))
    # learning_rates = [np.float(lr) for lr in learning_rates]
    stepSchedule = StepSchedule(step_config=[0, 1, 2, 3], change=learning_rates)
    optimizer = GradientDescentMomentum(learning_rate=0.0001, momentum_coef=0.9, schedule=stepSchedule)

    # initialize model object
    mlp = Model(layers=layers)

    # configure callbacks
    callbacks = Callbacks(mlp, eval_set=test_set, eval_freq=1)
    callbacks.add_callback(CostVisCallback(h=300, w=300, nepochs=4, y_range=None, fig=fig, handle=handle,
                                           train_source=train_source, val_source=val_source))

    # run fit
    mlp.fit(train_set, optimizer=optimizer, num_epochs=4, cost=cost, callbacks=callbacks)
    metric = mlp.eval(test_set, metric=SumSquaredMetric())
    
    test_set.reset()
    (X, T) = test_set.__iter__().next()
    y = mlp.fprop(X)
    y = y.get()
    T = T.get()
    result = {'img': X.get(), 'pred': y, 'gt': T}
    
    
    be.cleanup_backend()
    be = None

    return (metric, result)


class Dashboard():

    def __init__(self):
        self.best_cost = 10000
        self.setup_dashboard()

    def setup_dashboard(self):
        output_notebook()

        x = [0,    1,     1,      2,     2,     3,    3,    4]
        y = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        source = ColumnDataSource(data=dict(x=x, y=y))

        plot = figure(plot_width=300, plot_height=200, y_axis_type="log", y_range=[0.0000000001, 1], x_range=[0, 4],
                      x_axis_label='Epoch', y_axis_label='Learning Rate')
        plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

        learn_inputs = 4 * [3]

        base_code = """
                var data = source.data;
                var f = cb_obj.value
                x = data['x']
                y = data['y']
                y[{}] = Math.pow(10.0, -1.0 * (10-f))
                y[{}] = Math.pow(10.0, -1.0 * (10-f))
                source.trigger('change');

                var command = 'learn_inputs[{}] = ' + f;
                var kernel = IPython.notebook.kernel;
                kernel.execute(command)
            """

        # set up figure
        fig = figure(name="cost", y_axis_label="Cost", x_range=(0, 4),
                     x_axis_label="Epoch", plot_width=300, plot_height=300)
        self.fig = fig
        train_source = ColumnDataSource(data=dict(x=[], y=[]))
        train_cost = fig.line('x', 'y', source=train_source)
        self.train_source = train_source

        val_source = ColumnDataSource(data=dict(x=[], y=[]))
        val_cost = fig.line('x', 'y', source=val_source, color='red')
        self.val_source = val_source

        # set up sliders and callback
        callbacks = [CustomJS(args=dict(source=source), code=base_code.format(k, k+1, k/2)) for k in [0, 2, 4, 6]]
        slider = [Slider(start=0.1, end=4, value=3, step=.1, title=None, callback=C, orientation='vertical', width=80, height=50) for C in callbacks]

        radio_group = RadioGroup(labels=[""], active=0, width=65)

        def train_model_button(run=True):
            train_model(slider, fig=fig, handle=fh, train_source=train_source, val_source=val_source)

        # bcall = CustomJS(code="""
        #     var kernel = IPython.notebook.kernel;
        #     cmd = "train_model_button()";
        #     kernel.execute(cmd, {}, {});
        # """)

        button = Button(label='Train', width=300, height=50, sizing_mode='scale_width', button_type='primary')

        sliders = row(radio_group, slider[0], slider[1], slider[2], slider[3])
        settings = column(button, plot, sliders)


        layout = gridplot([[settings, fig]], sizing_mode='fixed', merge_tools=True, toolbar_location=None)

        self.fh = show(layout, notebook_handle=True)

    def plot_results(self, result):
        plt.figure(2)
        imgs_to_plot = [0, 1, 2, 3]
        for i in imgs_to_plot:
            plt.subplot(2, 2, i+1)

            title = "test {}".format(i)
            plt.imshow(result['img'][:, i].reshape(3, 64, 64).transpose(1, 2, 0))
            y = result['pred']
            T = result['gt']
            ax = plt.gca()
            ax.add_patch(plt.Rectangle((y[0,i], y[1,i]), y[2,i], y[3,i], fill=False, edgecolor="red"))
            ax.add_patch(plt.Rectangle((T[0,i], T[1,i]), T[2,i], T[3,i], fill=False, edgecolor="blue"))
            plt.title(title)
            plt.axis('off')

    def train(self, learn_inputs):
        (cost, result) = train_model(learn_inputs, fig=self.fig, handle=self.fh, train_source=self.train_source, val_source=self.val_source)
        if cost < self.best_cost:
            self.best_cost = cost
        print "Final Cost: {}".format(cost)
        print "Best Cost: {}".format(self.best_cost)
        print "Note: lower is better."
        self.plot_results(result)


fileName = 'data/svhn_64.p'
print "Setting up the data..."
with open(fileName) as f:
    svhn = cPickle.load(f)

# 323

