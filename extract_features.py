import argparse
import pathlib
import librosa
import h5py
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def find_label_index(ado_path, ado_paths_list):
    for i, path in enumerate(ado_paths_list):
        if ado_path == path:
            return i
    return False

class Extractor:
    def __init__(self, dtpth, args, exmtd='logmel',  test=False):
        # paths
        self.dtpth = dtpth
        self.test = test
        self.args = args
        self.args.num_time_bin = int(np.ceil(args.duration * args.sr / args.hop_length))
        df_meta = pd.read_table(os.path.join(dtpth, "meta.txt"), sep='\t')
        self.ado_pths_list = [os.path.join(dtpth, i) for i in df_meta['path'].tolist()]
        self.labels_kind = df_meta['label'].astype('category').cat.categories.tolist
        self.label = df_meta['label'].astype('category').cat.codes.values

    def ex_raw(self):
        for i, ado in tqdm(enumerate(self.ado_pths_list), total=len(self.ado_pths_list)):
            y, self.args.sr = librosa.load(ado, sr=None)
            logmel_data = np.zeros((self.args.num_channel,self.args.num_freq_bin, self.args.num_time_bin), 'float32')
            logmel_data[0, :, :] = librosa.feature.melspectrogram(y[:], sr=self.args.sr, n_fft=self.args.num_fft,
                                                                  hop_length=self.args.hop_length,
                                                                  n_mels=self.args.num_freq_bin, fmin=0.0,
                                                                  fmax=self.args.sr / 2, htk=True, norm=None)
            logmel_data = np.log(logmel_data + 1e-8)
            feat_data = logmel_data
            fet = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
            self._save_feature(path=self._get_aug_save_path(ado, aug='raw'),
                               feature=fet, label=self.label[i])
            if self.test:
                return fet, self.label[i]
                # break
        self._print_save_path(self._get_aug_save_path(self.ado_pths_list[0], aug='raw'))

    def ex_noise(self):
        for i, ado in tqdm(enumerate(self.ado_pths_list), total=len(self.ado_pths_list)):
            y, sr = librosa.load(ado, sr=None)
            noise = np.random.normal(0, 1, len(y))
            augmented_data = np.where(y != 0.0, y.astype('float64') + 0.01 * noise, 0.0).astype(np.float32)
            y = augmented_data
            logmel_data = np.zeros((self.args.num_channel,self.args.num_freq_bin, self.args.num_time_bin), 'float32')
            logmel_data[0, :, :] = librosa.feature.melspectrogram(y[:], sr=self.args.sr, n_fft=self.args.num_fft,
                                                                  hop_length=self.args.hop_length,
                                                                  n_mels=self.args.num_freq_bin, fmin=0.0,
                                                                  fmax=self.args.sr / 2, htk=True, norm=None)
            logmel_data = np.log(logmel_data + 1e-8)
            feat_data = logmel_data
            fet = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
            self._save_feature(path=self._get_aug_save_path(ado, aug='noise2'),
                               feature=fet, label=self.label[i])
            if self.test:
                return fet, self.label[i]
                # break
        self._print_save_path(self._get_aug_save_path(self.ado_pths_list[0], aug='noise2'))

    def ex_pitch(self):
        for i, ado in tqdm(enumerate(self.ado_pths_list), total=len(self.ado_pths_list)):
            stereo, sr = librosa.load(ado, sr=None)
            n_step = np.random.uniform(-4, 4)
            y_pitched = librosa.effects.pitch_shift(stereo, sr, n_steps=n_step)
            length = len(stereo)
            stereo = y_pitched
            logmel_data = np.zeros((self.args.num_channel, self.args.num_freq_bin, self.args.num_time_bin), 'float32')
            logmel_data[0, :, :] = librosa.feature.melspectrogram(stereo[:], sr=self.args.sr, n_fft=self.args.num_fft,
                                                                  hop_length=self.args.hop_length,
                                                                  n_mels=self.args.num_freq_bin, fmin=0.0,
                                                                  fmax=self.args.sr / 2, htk=True, norm=None)
            logmel_data = np.log(logmel_data + 1e-8)
            feat_data = logmel_data
            fet = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
            self._save_feature(path=self._get_aug_save_path(ado, aug='pitch'),
                               feature=fet, label=self.label[i])
            if self.test:
                return fet, self.label[i]
                # break
        self._print_save_path(self._get_aug_save_path(self.ado_pths_list[0], aug='pitch'))

    def ex_time(self):
        for i, ado in tqdm(enumerate(self.ado_pths_list), total=len(self.ado_pths_list)):
            stereo, sr = librosa.load(ado, sr=None)
            time_factor = np.random.uniform(0.5, 2)
            length = len(stereo)
            y_stretch = librosa.effects.time_stretch(stereo, time_factor)
            if len(y_stretch) < length:
                y_stretch = np.concatenate((y_stretch, y_stretch))
                y_stretch = y_stretch[0:length]
            else:
                y_stretch = y_stretch[0:length]
            stereo = y_stretch
            logmel_data = np.zeros((self.args.num_channel,self.args.num_freq_bin, self.args.num_time_bin), 'float32')

            logmel_data[0, :, :] = librosa.feature.melspectrogram(stereo[:], sr=self.args.sr, n_fft=self.args.num_fft,
                                                                  hop_length=self.args.hop_length,
                                                                  n_mels=self.args.num_freq_bin, fmin=0.0,
                                                                  fmax=self.args.sr / 2, htk=True, norm=None)
            logmel_data = np.log(logmel_data + 1e-8)
            feat_data = logmel_data
            fet = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
            self._save_feature(path=self._get_aug_save_path(ado, aug='time2'),
                               feature=fet, label=self.label[i])
            if self.test:
                return fet, self.label[i]
                # break
        self._print_save_path(self._get_aug_save_path(self.ado_pths_list[0], aug='time2'))

    def _save_feature(self, path, feature, label, mode='w'):
        feature_path = pathlib.Path(path).parent
        if not os.path.exists(feature_path):
            os.mkdir(feature_path)
        with h5py.File(path, mode) as hf:
            hf.create_dataset('feature', data=feature)
            hf.attrs['label'] = np.int64(label)

    def _get_aug_save_path(self, ado, aug):
        ado_fn_name = ado.split('/')[-1].split('.')[0]
        save_path = os.path.join(self.dtpth, 'feature', 'logmel_aug', aug, ado_fn_name + ".h5")
        return save_path

    def _print_save_path(self, path):
        feature_path = pathlib.Path(path).parent
        print("=> save feature in {}".format(feature_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for extractor')
    parser.add_argument('--dtpth', metavar='DIR', help='dataset path')
    parser.add_argument('--sr', default=16000, type=int, metavar='N',
                        help='sampling rate (default: 48000)')
    parser.add_argument('--duration', default=10, type=int, metavar='N',
                        help='duration of audio recording')
    parser.add_argument('--num_fft', default=2048, type=int, metavar='N',
                        help='number of fft')
    parser.add_argument('--num_freq_bin', default=128, type=int, metavar='N',
                        help='number of filter banks')
    parser.add_argument('--num_channel', default=1, type=int,metavar='N',
                        help='channels')
    parser.add_argument('--hop_length', default=512, type=int, metavar='N',
                        help='hop length')
    # parser.add_argument('--num_freq_bin', default=128, type=int, metavar='N',
    #                     help='manual epoch number (useful on restarts)')
    args = parser.parse_args()
    isTest = True
    if isTest:
        extractor = Extractor(args.dtpth, args, test=True)
        raw_fet, raw_lbl = extractor.ex_raw()
        time_fet, time_lbl = extractor.ex_time()
        noise_fet, noise_lbl = extractor.ex_noise()
        pitch_fet, pitch_lbl = extractor.ex_pitch()
    else:
        extractor = Extractor(args.dtpth, args)
        extractor.ex_raw()

