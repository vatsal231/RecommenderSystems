# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
 

"""
Implementation of Iautorec (item-based autorec).
The author proved that Iautorec (item-item matrix) is better than Uautorec (user-item matrix). 
Paper: http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import sklearn
import tensorflow as tf
import keras
from keras import optimizers
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential



class autorec:

    def BatchRange_Compute(self, N, batch_index, batch_size):
        """
        Compute batch range
        :param N: the number of observation ratings
        :param batch_index: the index of batch
        :param batch_size: batch's size
        :return:
        """
        upper = np.min([N, (batch_index + 1) * batch_size])
        lower = batch_index * batch_size
        return lower, upper

    def MiniBatchLoss_Compute(self, y_true, y_pred):
        """
        Compute loss
        :param y_true: the true output
        :param y_pred: the predicted output
        :return: loss
        """
        # mask = y_true > 0 # wrong, y_true may be smaller than zero
        mask = tf.cast(tf.not_equal(y_true, 0), dtype='float32')  # convert float64 to float32

        e = y_true - y_pred
        # should use '*' rather than tf.multiply because of performance. why?
        se = e * e  # element-wise multiplication
        se = se * mask  # ignore the missing value (having value of zero) in loss computation
        mse = 1.0 * tf.reduce_sum(se) / tf.reduce_sum(mask)  # why does np.sum() not work here?
        rmse = tf.math.sqrt(mse)
        return rmse  # root mean square error

    def train_generator(self, Xtrain, batch_size):
        """
        Generate batch samples. Use in  fit_generator() in Keras
        :param Xtrain: input matrix NxD
        :param batch_size: batch's size
        :return: batch samples
        """
        while True:  # loop indefinitely. important!
            N = Xtrain.shape[0]
            n_batches = int(np.ceil(N / batch_size))

            Xtrain2 = sklearn.utils.shuffle(Xtrain)  # affect rows
            # mask = np.not_equal(Xtrain, 0)
            # average_rating = np.sum(Xtrain) / np.sum(mask)

            for batch in range(n_batches):
                lower, upper = self.BatchRange_Compute(N, batch, batch_size)
                inputs = Xtrain2[lower:upper, :]

                # inputs = inputs - average_rating * mask[lower:upper, :]

                targets = inputs
                yield inputs, targets

    def fit(self, Xtrain, Xtest, batch_size=128, epoch=500):
        Ntrain, D = Xtrain.shape
        print('Item-Item matrix shape: ' + str(Xtrain.shape))

        # create layer
        model = Sequential()
        model.add(Dense(input_dim=(D), units=500, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001),
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
        # model.add(Dropout(rate=0.3))  # Rate should be set to `rate = 1 - keep_prob`
        model.add(Dense(units=D, activation='linear', kernel_regularizer=regularizers.l2(0.0001),
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=self.MiniBatchLoss_Compute, metrics=[self.MiniBatchLoss_Compute])
        history = model.fit_generator(generator=self.train_generator(Xtrain, batch_size), epochs=epoch,
                                      steps_per_epoch=int(np.ceil(Ntrain / batch_size)),
                                      validation_data=(Xtest, Xtest),
                                      validation_steps=int(np.ceil(Xtest.shape[0] / batch_size)))
        ypred_test, ypred_train = model.predict(Xtest), model.predict(Xtrain)
        np.savez('item-item_matrix.npz', ypred_test = ypred_test, ypred_train = ypred_train)

        # plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.show()


def main():
    filepath = ''
    X = scipy.sparse.load_npz(filepath).todense()
    X = np.array(X)  # manipulate on matrix is extremely slow
    cutoff = np.math.floor(X.shape[0] * 0.9)  # 90% train, 10% test
    Xtrain = X[:cutoff, :]
    Xtest = X[cutoff:, :]

    AutoRec = autorec()
    AutoRec.fit(Xtrain, Xtest)


main()