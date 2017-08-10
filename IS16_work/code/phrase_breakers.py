import os, codecs, argparse
from data_load import process_data

# numpy imports
import numpy as np

# scikit learn imports
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

# keras imports
# pip install keras=1.1.1
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from nltk.tokenize import TweetTokenizer


class LSTMBreaker:

    def __init__(self,
                 vocab_size=7286,
                 embedding_dims=100,
                 rnn_layer_dim=256,
                 num_classes=2,
                 load_weights='../model/LSTM_weights2.hdf5'):

        self.model = Sequential()  # Sequential self.model
        # Embedding layer (Word embeddings)
        self.model.add(Embedding(vocab_size, embedding_dims))
        # Recurrent layer
        self.model.add(LSTM(int(rnn_layer_dim),
                            # input_shape=(embedding_dims,),
                            init='glorot_uniform',
                            inner_init='orthogonal',
                            forget_bias_init='one',
                            activation='tanh',
                            inner_activation='hard_sigmoid',
                            W_regularizer=None,
                            U_regularizer=None,
                            b_regularizer=None,
                            dropout_W=0.0,
                            dropout_U = 0.0,
                            return_sequences=True,
                            stateful=False)
        )
        # Time distributed dense layer (activation is softmax, since it is 
        # a classification problem)
        self.model.add(TimeDistributedDense(num_classes,
                                            init='glorot_uniform',
                                            activation='softmax')
        )
        if load_weights:
            self.model.load_weights(load_weights)

            print("LOADEDED")
            self.model.compile(loss='categorical_crossentropy',
                               optimizer = 'sgd', metrics = ['accuracy'])
            print("COMPILED")

    @staticmethod
    def proc_text(text, mappings):
        #lst = text.split(" ")
        tknzr = TweetTokenizer()
        lst = tknzr.tokenize(text)
        print lst
        out_vec = []
        for word in lst:
            if word in mappings:
                out_vec.append(mappings[word])
            else:
                out_vec.append(mappings["_pad_"])
        return out_vec

    def predict(self, x):
        return self.model.predict_classes(x)

if __name__ == "__main__":
    model = LSTMBreaker()
    text, labels, word2index, label2index = process_data("../data/PAP")
    # print(word2index)
    print(text[0], len(text[0]))#, text.shape)
    print(map(len, text))
    example = ("State media said Hwasong-12 rockets would pass over Japan and land in the sea about 30km (17 miles) from Guam, if the plan was approved by Kim Jong-un."
               + "It denounced Donald Trump's warnings of \"fire and fury\" and said the US leader was \"bereft of reason\".")

    exp_vec = np.asarray(model.proc_text(example, word2index))
    print(exp_vec)
    prediction = model.predict(np.array(text[-1]).reshape(1,-1))
    print(prediction)