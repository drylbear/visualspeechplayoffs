import csv
import numpy as np
import glob
import os.path
import threading
import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
import collections

# all the phonemes and visemes possible for our case
uniq_visemes = ["v01", "v02", "v03", "v04", "v05", "v06", "v07", "v08", "sil", "gar", "tcd"]
uniq_phonemes = ["p", "b", "m", "f", "v", "w", "r", "W", "t", "d", "n", "l", "th", "dh", "s", "z", "ch", "jh", "sh",
                 "zh", "j", "k", "g", "uh", "hh", "ee", "iy", "ay", "eh", "ah", "uw", "oo", "ax", "ua", "sil", "sp",
                 "ow", "az", "oh", "ey", "ao", "aa", "ia", "er", "oy", "ng", "aw", "ih", "hh", "ae", "y"]



class ThreadSaveIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def thread_safe_generator(func):
    def generator(*a, **kw):
        return ThreadSaveIterator(func(*a, **kw))

    return generator


class DataSet:
    """A class to wrap the data set including getting sequences with the corresponding ground truth

    Attributes:
        seq_length (int): Length all sequences are trimmed or padded to
        unit (str): The unit of detection. Can be one of (phonemes - visemes)
        data_type (str): The feature type. Can be one of (dct - hog - sift - aam - image)
        data (list): The data stored as a list. Each item is of form ['train', speaker, seq_name, num_frames, list_gt]
        classes (list): All possible classes for the given unit
    """

    def __init__(self, path, seq_length=120, data_type="dct", unit="phonemes"):
        """Load ground truth and lables. Clean the data and get speakers

        :param path: Path of the base directory of the project
        :param seq_length: The length all sequences should be trimmed or padded to
        :param data_type: The feature type. Can be one of (dct - hog - sift - aam - image)
        :param unit: The unit of detection. Can be one of (phonemes - visemes)
        """
        self.seq_length = seq_length
        self.__data_path = os.path.join(path, "data", "train")
        self.__sequence_path = os.path.join(path, "data", "sequence")
        self.__image_path = os.path.join(path, "data", "resnet")
        self.__path = path
        self.unit = unit
        self.data_type = data_type

        if self.data_type == "image":
            self.data = self.__get_data_image(self.__path, self.unit)
        else:
            self.data = self.__get_data_features(self.__path, self.data_type, self.unit)
        self.classes = self.__get_classes()
        self.data = self.__clean_data()
        # self.labels = [self.get_class_one_hot(i[4]) for i in self.data]

        self.__speakers = self.__get_speakers_from_data(self.data)

    def get_class_one_hot(self, lables):
        """For a given sequence label, get the most likely class as a index for each frame

        :param lables (list): A list of lables with size (seq_length, feature_length)
        :return List of length (seq_length) with class index as int
        """
        tmp = []
        for class_str in lables:
            label_encoded = self.classes.index(class_str)
            label_hot = to_categorical(label_encoded, len(self.classes))

            assert (len(label_hot) == len(self.classes))
            tmp.append(label_hot)
        return tmp

    def split_train_test(self, size_test):
        """Split data into train and test data

        :param size_test (int): number of speakers which will be used for testing
        :return: list of train, train labels, test and test labels
        """
        # set up speakers randomly
        test_speakers = np.random.choice(self.__speakers, size_test, replace=False)
        train_speakers = [s for s in self.__speakers if s not in test_speakers]

        # split sequences acoording to speakers
        train = []
        y_test = []
        y_train = []
        test = []
        for item in self.data:
            if item[1] in train_speakers:
                train.append(item)
                y_train.append(self.get_class_one_hot(item[4]))
            elif item[1] in test_speakers:
                test.append(item)
                y_test.append(self.get_class_one_hot(item[4]))
            else:
                ValueError("Something went wrong with speakers while splitting them up")
        return train, y_train, test, y_test

    def get_sequence(self, sample):
        """Load a feature/image for a given sample.

        :param sample: a single sample out of the data list
        :return: The loaded feature for the sequence
        """
        sequence = self.__get_extracted_sequence(sample)

        if sequence is None:
            raise ValueError("Can't find feature: {} for {}/{}. Did you generate them?\n"
                             .format(self.data_type, sample[1], sample[2]))

        assert len(sequence) == self.seq_length
        return sequence

    def get_class_from_prediction(self, predictions):
        """Return the most likely class for each predicted frame"""
        return (np.array(self.classes)[np.argmax(predictions, axis=1)])

    @staticmethod
    def split_train_valid(x_com, y_com, nb_seq):
        """Take randomly nb_seq sequences out of every speaker in x_train for validation, the rest is train

        :param x_com: The complete list of training data
        :param y_com: The complete list of labels for training data
        :param nb_seq: The number of sequences for each speaker should be used for validation
        :return: The split into train, train labels, validation and validation labels
        """
        x_train, x_valid, y_train, y_valid = [], [], [], []

        # split data by speaker
        speaker_data = {}
        speaker_label = {}
        for i, item in enumerate(x_com):
            speaker = item[1]
            if speaker not in speaker_data:
                speaker_data[speaker] = []
                speaker_label[speaker] = []
            speaker_data[speaker].append(item)
            speaker_label[speaker].append(y_com[i])

        # split data so that for each speaker nb_seq of sequences are included in validation
        for item in speaker_data:
            x_item = speaker_data[item]
            y_item = speaker_label[item]
            x_train_tmp, x_valid_tmp, y_train_tmp, y_valid_tmp = train_test_split(x_item, y_item, test_size=nb_seq)

            x_train.extend(x_train_tmp)
            y_train.extend(y_train_tmp)
            x_valid.extend(x_valid_tmp)
            y_valid.extend(y_valid_tmp)

        return x_train, y_train, x_valid, y_valid

    def load_to_memory(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        tmp_xtrain = []
        tmp_ytrain = []
        for X, y in zip(x_train, y_train):
            seq = self.get_sequence(X)
            tmp_xtrain.append(seq)
            tmp_ytrain.append(y)
        tmp_xtrain = np.asarray(tmp_xtrain, np.float32)
        tmp_ytrain = np.asarray(tmp_ytrain, np.float32)

        tmp_xval = []
        tmp_yval = []
        for X, y in zip(x_valid, y_valid):
            seq = self.get_sequence(X)
            tmp_xval.append(seq)
            tmp_yval.append(y)
        tmp_xval = np.asarray(tmp_xval, np.float32)
        tmp_yval = np.asarray(tmp_yval, np.float32)

        tmp_xtest = []
        tmp_ytest = []
        for X, y in zip(x_test, y_test):
            seq = self.get_sequence(X)
            tmp_xtest.append(seq)
            tmp_ytest.append(y)
        tmp_xtest = np.asarray(tmp_xtest, np.float32)
        tmp_ytest = np.asarray(tmp_ytest, np.float32)

        return tmp_xtrain, tmp_ytrain, tmp_xval, tmp_yval, tmp_xtest, tmp_ytest


    def __get_extracted_sequence(self, sample):
        speaker = sample[1]
        filename = sample[2]
        if self.data_type == "image":
            jpgs_dir = os.path.join(self.__image_path, speaker)
            jpgs = sorted(glob.glob(os.path.join(jpgs_dir, filename + "-*.jpg")))

            features = np.asarray([preprocess_input(img_to_array(cv2.imread(jpg))) for jpg in jpgs])
        else:
            feature_file = os.path.join(self.__sequence_path, speaker + "-" + filename + "-" + self.data_type + ".npy")
            features = np.load(feature_file)

        if sample[5] == "fit" or sample[5] == "trimEnd":
            return features[:sample[3]]
        elif sample[5] == "padded":
            if self.seq_length < features.shape[0]:
                raise ValueError(
                    "This should not happen feature.shape = {} for {}/{}".format(features.shape, speaker, filename))
            dif = self.seq_length - features.shape[0]
            return np.append(features, np.repeat(np.array([features[-1]]), dif, axis=0), axis=0)
        elif sample[5] == "trimFront":
            end = sample[6] + sample[3]
            return features[sample[6]:end]
        else:
            return None

    def __clean_data(self):
        data_clean = []
        lengths = self.__get_max_sequence_length(self.data)
        print("Max Sequence length found: ", np.max(lengths, axis=0)[2])
        for i, item in enumerate(self.data):  # item = [train_test, speaker, file_no_ext, nb_frames, labels]
            new_item = []
            if int(item[3]) == self.seq_length:  # lengths fits perfectly
                new_item = [item[0], item[1], item[2], int(item[3]), item[4], "fit"]
            elif int(item[3]) < self.seq_length:  # snipped is to short, pad it at the end!
                dif = self.seq_length - int(item[3])
                tmp = item[4]
                tmp.extend(["sil"] * dif)
                new_item = [item[0], item[1], item[2], self.seq_length, tmp, "padded", dif]
            elif int(item[3]) > self.seq_length:  # total lengths is longer than than sequence length
                if lengths[i][1] <= self.seq_length:  # check if end can be cut away
                    tmp = item[4][:self.seq_length]
                    new_item = [item[0], item[1], item[2], self.seq_length, tmp, "trimEnd"]
                else:  # check if we can cut front and back
                    if lengths[i][2] <= self.seq_length:
                        rest = lengths[i][1] - self.seq_length
                        tmp = item[4][rest:lengths[i][1]]
                        new_item = [item[0], item[1], item[2], self.seq_length, tmp, "trimFront", rest]
                    else:  # total speaking is to long for sequence
                        print(
                            "{}/{} could not be trimmed to {}, so remove it".format(item[1], item[2], self.seq_length))
                        new_item = []

            if len(new_item) > 0:
                data_clean.append(new_item)
        return sorted(data_clean)

    def __get_classes(self):
        if self.unit == "phonemes":
            return uniq_phonemes
        elif self.unit == "visemes":
            return uniq_visemes

    @staticmethod
    def __get_speakers_from_data(dataset):
        speakers = []
        aux1 = set()
        for sample in dataset:
            if sample[1] not in aux1:
                speakers.append(sample[1])
                aux1.add(sample[1])
        return sorted(speakers)

    @staticmethod
    def __get_max_sequence_length(dataset):
        lengths = []  # start, end, difference
        for item in dataset:
            start = item[4].index(list(filter(lambda x: x != "sil", item[4]))[0])
            end = len(item[4]) - 1 - item[4][::-1].index(list(filter(lambda x: x != "sil", reversed(item[4])))[0])
            lengths.append([start, end, end - start])
        return lengths

    @staticmethod
    def __get_ground_truth(path, unit="phonemes"):
        d = collections.defaultdict(dict)
        with open(os.path.join(path, "data", "frame_by_frame.txt"), "r") as fbf:
            for line in fbf:
                li = line.strip().split(" ")
                if not li[0] == "speaker":
                    if not li[0] in d.keys():
                        d[li[0]] = {}
                    if not li[1] in d[li[0]].keys():
                        d[li[0]][li[1]] = {}
                    if unit == "phonemes":
                        d[li[0]][li[1]][int(li[2])] = li[3]
                    elif unit == "visemes":
                        d[li[0]][li[1]][int(li[2])] = li[4]
                    else:
                        raise ValueError("Unknown unit type for ground truth")
        return dict(d)

    @staticmethod
    def __get_data_image(path, unit="phonemes"):
        dataset = []
        group = "resnet"
        image_dir = os.path.join(path, "data", group)
        gt = DataSet.__get_ground_truth(path, unit)  # [speaker][file_no_ext][frame]

        speakers_path = sorted(glob.glob(os.path.join(image_dir, '*')))
        for speaker_path in speakers_path:
            speaker = os.path.basename(speaker_path)
            jpgs_all = sorted(glob.glob(speaker_path + '/*'))
            files = list(set([jpg.split('-')[0] for jpg in jpgs_all]))
            for i, file_path in enumerate(files):
                seq_name = os.path.basename(file_path)

                gt_file = list(gt[speaker][seq_name].values())
                num_frames = len(gt_file)

                dataset.append([group, speaker, seq_name, num_frames, gt_file])

        return dataset

    @staticmethod
    def __get_data_features(path, data_type, unit="phonemes"):
        dataset = []
        sequence_dir = os.path.join(path, "data", "sequence")
        gt = DataSet.__get_ground_truth(path, unit)  # [speaker][file_no_ext][frame]

        features = sorted(glob.glob(os.path.join(sequence_dir, "*-" + data_type + ".npy")))
        group = "train"
        for feature_path in features:
            file = os.path.basename(feature_path)
            speaker, seq_name = file.split(".")[0].split("-")[:2]
            gt_file = list(gt[speaker][seq_name].values())
            num_frames = len(gt_file)

            dataset.append([group, speaker, seq_name, num_frames, gt_file])

        return dataset
