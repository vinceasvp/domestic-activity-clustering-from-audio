import threading
import keras
import numpy as np
import os
import logging
from matplotlib import pyplot as plt
import math
import pandas


class ParamsLogger:
    def __init__(self, savedir):
        self.log = logging.getLogger(savedir)
        self.log.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s: %(message)s")

        self.fhander = logging.FileHandler(os.path.join(savedir, 'params&results.log'), mode='w')
        self.fhander.setLevel(logging.DEBUG)
        self.fhander.setFormatter(self.formatter)
        self.log.addHandler(self.fhander)

        self.shander = logging.StreamHandler()
        self.shander.setLevel(logging.DEBUG)
        self.shander.setFormatter(self.formatter)
        self.log.addHandler(self.shander)

    def write_log(self, *args, **kwargs):
        self.log.info(args)
        # self.log.info(kwargs)


def model_log(save_dir, model):
    from contextlib import redirect_stdout

    with open(os.path.join(save_dir, 'modelsummary.log'), 'w') as f:
        with redirect_stdout(f):
            model.summary()


def get_history_figure(history, savedir, has_val=True, draw_acc=True, draw_loss=True, acc_title=None, loss_title=None):

    if draw_acc:
        plt.plot(history.history['acc'])
        if has_val:
            plt.plot(history.history['val_acc'])
        if acc_title:
            plt.title(acc_title)
        else:
            plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        if acc_title:
            plt.savefig(os.path.join(savedir, acc_title))
        else:
            plt.savefig(os.path.join(savedir, "acc_figure"))
        plt.show()
    if draw_loss:
        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'])
        if has_val:
            plt.plot(history.history['val_loss'])
        if loss_title:
            plt.title(loss_title)
        else:
            plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        if loss_title:
            plt.savefig(os.path.join(savedir, loss_title))
        else:
            plt.savefig(os.path.join(savedir, "loss_figure"))
        plt.show()


def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class Generator_withdelta_splitted():
    def __init__(self, feat_data, feat_label, feat_dim, batch_size=32, alpha=0.2, shuffle=True, crop_length=400,
                 splitted_num=4):
        # self.feat_path = feat_path
        # self.train_csv = train_csv
        # self.feat_list = os.listdir(feat_path)
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        # if os.path.exists(train_csv):
        #     self.sample_num = len(open(train_csv, 'r').readlines()) - 1
        # else:
        self.sample_num = len(feat_label)
        self.tr_datas, self.tr_lbs = feat_data, feat_label
        self.lock = threading.Lock()
        self.NewLength = crop_length
        self.splitted_num = splitted_num

    def __iter__(self):
        return self

    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()

                # split data and then load it to memory one by one
                item_num = self.sample_num // self.splitted_num - (
                            self.sample_num // self.splitted_num) % self.batch_size
                for k in range(self.splitted_num):
                    cur_item_num = item_num
                    s = k * item_num
                    e = (k + 1) * item_num
                    if k == self.splitted_num - 1:
                        cur_item_num = self.sample_num - (self.splitted_num - 1) * item_num
                        e = self.sample_num

                    lines = indexes[s:e]
                    # if os.path.exists(self.train_csv):
                    #     X_train, y_train = load_data_2020_splitted(self.feat_path, self.train_csv, self.feat_dim, lines,
                    #                                                'logmel')
                    # else:
                        # X_train, y_train = self._load_hdf5_data(self.feat_path, self.feat_list,lines)
                    X_train = self.tr_datas[lines, :]
                    y_train = self.tr_lbs[lines]
                    y_train = keras.utils.to_categorical(y_train, 10)

                    # X_deltas_train = deltas(X_train)
                    # X_deltas_deltas_train = deltas(X_deltas_train)
                    # X_train = np.concatenate(
                    #     (X_train[:, :, 4:-4, :], X_deltas_train[:, :, 2:-2, :], X_deltas_deltas_train), axis=-1)
                    # mean = np.mean(X_train, axis=2, keepdims=True)
                    # std = np.std(X_train, axis=2, keepdims=True)
                    # X_train = (X_train-mean) / std
                    # cur_item_num:data numbers in memory
                    itr_num = int(cur_item_num // (self.batch_size * 2))
                    for i in range(itr_num):
                        batch_ids = np.arange(cur_item_num)[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                        X, y = self.__data_generation(batch_ids, X_train, y_train)

                        yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)
        indexes = list(indexes)
        return indexes

    def __data_generation(self, batch_ids, X_train, y_train):
        _, h, w, c = X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = X_train[batch_ids[:self.batch_size]]
        X2 = X_train[batch_ids[self.batch_size:]]

        # random cropping
        # for j in range(X1.shape[0]):
        #     StartLoc1 = np.random.randint(0, X1.shape[2] - self.NewLength)
        #     StartLoc2 = np.random.randint(0, X2.shape[2] - self.NewLength)
        #
        #     X1[j, :, 0:self.NewLength, :] = X1[j, :, StartLoc1:StartLoc1 + self.NewLength, :]
        #     X2[j, :, 0:self.NewLength, :] = X2[j, :, StartLoc2:StartLoc2 + self.NewLength, :]
        #
        # X1 = X1[:, :, 0:self.NewLength, :]
        # X2 = X2[:, :, 0:self.NewLength, :]

        # mixup
        X = X1 * X_l + X2 * (1.0 - X_l)

        if isinstance(y_train, list):
            y = []

            for y_train_ in y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1.0 - y_l))
        else:
            y1 = y_train[batch_ids[:self.batch_size]]
            y2 = y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1.0 - y_l)

        return X, X






