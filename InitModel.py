from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import losses
import time
import h5py
import keras


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    '''def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()'''


def build_model(layers):
    model = Sequential()
    model.add(LSTM(layers[1], input_shape=(25, 12), return_sequences=True))
    model.add(LSTM(layers[2], return_sequences=False))
    # model.add(Dense(layers[3], activation="softmax"))
    model.add(Dense(layers[3]))

    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer='rmsprop', metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)
    return model


def Train_model(data, modelname):
    model = build_model([12, 256, 32, 2])
    x_train, y_train = data['x_train'], data['y_train']
    history = LossHistory()
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=500,
        validation_split=0.05,
        callbacks=[history])

    model.save(modelname)
    return
