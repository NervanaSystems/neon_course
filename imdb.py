import pickle as pkl
from neon.data import ArrayIterator
import numpy as np
import pickle as pkl
from neon.data.text_preprocessing import clean_string
from neon.initializers import Uniform, GlorotUniform
from neon.layers import LSTM, Affine, Dropout, LookupTable, RecurrentSum, Recurrent
from neon.transforms import Logistic, Tanh, Softmax
from neon.models import Model
from neon.optimizers import Adagrad, GradientDescentMomentum, Schedule
from neon.transforms import CrossEntropyMulti, Logistic, Tanh, Softmax
from neon.layers import GeneralizedCost
from neon.callbacks import Callbacks
from neon.transforms.cost import Accuracy
from viz_callback import CostVisCallback
from neon.backends import gen_backend
from IPython.display import display
import ipywidgets as widgets

sentence_length = 128
vocab_size = 20000

input_numpy = np.zeros((sentence_length, 128), dtype=np.int32)

vocab, rev_vocab = pkl.load(open('data/imdb.vocab', 'rb'))
data = pkl.load(open('data/imdb_data.pkl', 'r'))


def preprocess(sent, input_device):

    tokens = clean_string(sent).strip().split()

    sent = [len(vocab) + 1 if t not in vocab else vocab[t] for t in tokens]
    sent = [1] + [w + 3 for w in sent]
    sent = [2 if w >= vocab_size else w for w in sent]

    trunc = sent[-sentence_length:]  # take the last sentence_length words

    input_numpy[:] = 0  # fill with zeros
    input_numpy[-len(trunc):, 0] = trunc   # place the input into the numpy array
    input_device[:] = input_numpy  # copy the numpy array to device

    return input_device


class IMDB(object):

    def __init__(self, be, data_path=''):


        n = data['X_train'].shape[0]
        # n = np.int32(n / 2.0)
        data['Y_train'] = np.array(data['Y_train'], dtype=np.int32)
        data['Y_valid'] = np.array(data['Y_valid'], dtype=np.int32)

        self.train_set = ArrayIterator(data['X_train'][:n, :], data['Y_train'][:n, :], nclass=2)
        self.valid_set = ArrayIterator(data['X_valid'], data['Y_valid'], nclass=2)


def train_model(learning_rate=0.0002):
    be = gen_backend(backend='gpu', batch_size=128, rng_seed=0)
    imdb = IMDB(be)

    init_glorot = GlorotUniform()
    init_uniform = Uniform(-0.1 / 128, 0.1 / 128)

    layers = [
        LookupTable(vocab_size=20000, embedding_dim=128, init=init_uniform),
        LSTM(output_size=64, init=init_glorot, activation=Tanh(),
             gate_activation=Logistic(), reset_cells=True),
        RecurrentSum(),
        Dropout(0.5),
        Affine(nout=2, init=init_glorot, bias=init_glorot, activation=Softmax())
    ]

    # create model object
    model = Model(layers=layers)

    # define cost
    cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))

    # define optimizer with a learning rate of 0.01
    optimizer = Adagrad(learning_rate=learning_rate)
    callbacks = Callbacks(model, eval_set=imdb.valid_set, serialize=1, save_path='imdb_lstm.model')
    callbacks.add_callback(CostVisCallback())
    model.fit(imdb.train_set, optimizer=optimizer, num_epochs=1, cost=cost, callbacks=callbacks)
    acc = 100 * model.eval(imdb.valid_set, metric=Accuracy())
    print "Accuracy - {}".format(acc)
    return acc


def text_window():

    text = widgets.Textarea(
    value="""This movie was a well-written story of Intel's history, from its highs to its lows. I especially liked the character development. They should have gotten Brad Pitt to play Robert Noyce though, the actor's acting was bad. For example, that scene where they left Fairchild to start Intel was way too exaggerated and melodramatic. The pace of the movie was exciting enough to overlook those minor issues. I was on the edge of my seat the whole time, and my brother was equally enthralled!""",
    placeholder='Type something',
    description='Review:',
    disabled=False,
    layout=widgets.Layout(height='200px', width='50%'))

    return text
