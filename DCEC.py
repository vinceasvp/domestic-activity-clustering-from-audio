import shutil
import subprocess
from datetime import datetime
from time import time
import os
from contextlib import redirect_stdout
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from sklearn.cluster import KMeans

import utils.metrics as metrics
from utils.util_layer import ClusteringLayer
from utils.utils import get_history_figure, ParamsLogger, Generator_withdelta_splitted
from model import DSCAE


class DCEC(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 pretrain_epochs=20,
                 alpha=1.0,
                 args=None,
                 data_generator=None):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.pretrained = False
        self.y_pred = []
        subprocess.check_call(["cp", args.source_file, args.save_dir])
        self.cae = DSCAE(input_shape=input_shape, filters=filters)
        hidden = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output])
        self.data_generator = data_generator

    def pretrain(self, x, batch_size=256, epochs=20, optimizer='adam', save_dir='results/temp', logger=None):
        logger.write_log('...Pretraining...')
        # print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(save_dir + '/pretrain_log.csv')

        # begin training
        t0 = time()
        # history = self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        history = self.cae.fit_generator(self.data_generator, epochs=epochs,
                                         callbacks=[csv_logger],
                                         steps_per_epoch=np.ceil(len(x)/batch_size))
        get_history_figure(history, save_dir, has_val=False, draw_acc=False, loss_title='pretrain_model_loss')
        logger.write_log('Pretraining time: ', time() - t0)
        # print('Pretraining time: ', time() - t0)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        logger.write_log('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        # print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3,
            update_interval=140, cae_weights=None, save_dir='./results/temp', logger=None):

        # print('Update interval', update_interval)
        logger.write_log('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5
        # print('Save interval', save_interval)
        logger.write_log('Save interval', save_interval)
        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            logger.write_log('...pretraining CAE using default hyper-parameters:')
            logger.write_log('   optimizer=\'adam\';   epochs={}'.format(self.pretrain_epochs))
            # print('...pretraining CAE using default hyper-parameters:')
            # print('   optimizer=\'adam\';   epochs={}'.format(self.pretrain_epochs))
            # self.pretrain(x, batch_size, epochs=self.pretrain_epochs, save_dir=save_dir, logger=logger)
            self.pretrained = True
        elif cae_weights is not None:
            import os
            if os.path.exists(cae_weights):
                shutil.copyfile(cae_weights, save_dir+'/pretrain_cae_model.h5')
                logger.write_log("copy %s -> %s" % (cae_weights, save_dir+'/pretrain_cae_model.h5'))
            self.cae.load_weights(cae_weights)
            logger.write_log('cae_weights is loaded successfully.')

            pretrain_log = os.path.dirname(cae_weights) + '/pretrain_log.csv'
            if os.path.exists(pretrain_log):
                shutil.copyfile(pretrain_log, save_dir+'/pretrain_log.csv')
                logger.write_log("copy %s -> %s" % (pretrain_log, save_dir+'/pretrain_log.csv'))

            # print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        logger.write_log('Initializing cluster centers with k-means.')
        # print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    logger.write_log('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)
                    # print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    logger.write_log('delta_label ', delta_label, '< tol ', tol)
                    logger.write_log('Reached tolerance threshold. Stopping training.')
                    # print('delta_label ', delta_label, '< tol ', tol)
                    # print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                logger.write_log('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                # print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        logger.write_log('saving model to:', save_dir + '/dcec_model_final.h5')
        # print('saving model to:', save_dir + '/dcec_model_final.h5')
        self.model.save_weights(save_dir + '/dcec_model_final.h5')
        t3 = time()
        logger.write_log('Pretrain time:  ', t1 - t0)
        logger.write_log('Clustering time:', t3 - t1)
        logger.write_log('Total time:     ', t3 - t0)
        # print('Pretrain time:  ', t1 - t0)
        # print('Clustering time:', t3 - t1)
        # print('Total time:     ', t3 - t0)


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('dataset', default='dcase-test', choices=['mnist', 'usps', 'mnist-test', 'dcase'])
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--n_clusters', default=9, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--pretrain_epochs', default=1, type=int)
    parser.add_argument('--maxiter', default=1, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results')
    parser.add_argument('--pretrain_arch', default='DSCAE', type=str)
    parser.add_argument('--source_file', default='model.py')
    parser.add_argument('--mixup_alpha', default=0.4, type=float)
    parser.add_argument('--splitted_num', default=2, type=int)
    parser.add_argument('--num_freq_bin', type=int, default=128)
    args = parser.parse_args()
    print(args)
    stamp = datetime.now().strftime('%y%m%d%H%M%S')
    tag = stamp + '_' + args.dataset + '_' + 'ncluster' + str(args.n_clusters) + '_' + 'batchsize' + str(args.batch_size) + \
          '_' + 'pretrainepoch' + str(args.pretrain_epochs) + '_' + 'maxiter' + str(args.maxiter)
    args.save_dir = os.path.join('results', args.pretrain_arch, tag)


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = ParamsLogger(args.save_dir)
    # load dataset
    from utils.datasets import load_mnist, load_usps, load_dcase

    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')
    elif args.dataset == 'mnist-test':
        x, y = load_mnist()
        x, y = x[60000:], y[60000:]
    elif args.dataset == 'dcase':
        x, y = load_dcase(args.data_path)

    train_data_generator = Generator_withdelta_splitted(x, y, args.num_freq_bin,
                                  batch_size=args.batch_size,
                                  alpha=args.mixup_alpha,
                                  crop_length=156, splitted_num=args.splitted_num)()
    logger.write_log('train data shape:', x.shape)
    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=args.n_clusters,
                pretrain_epochs=args.pretrain_epochs, args=args,
                data_generator=train_data_generator)
    dcec.model.summary()
    if args.model_path is not None:
        dcec.model.load_weights(args.model_path)

    logger.write_log(args)


    with open(os.path.join(args.save_dir, 'modelsummary.log'), 'w') as f:
        with redirect_stdout(f):
            dcec.model.summary()

    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[args.gamma, 1], optimizer=optimizer)
    dcec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval,
             save_dir=args.save_dir,
             cae_weights=args.cae_weights,
             logger=logger)

    y_pred = dcec.y_pred
    # print('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))
    logger.write_log('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))