import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data.extract_files import extract_files
import numpy as np
import logging
from itertools import product
from train import train
from model import get_model
from dataset import DataSet
import csv
from keras.utils.vis_utils import plot_model

logger = logging.getLogger('VisuSpeech.Main')
logger.setLevel(logging.DEBUG)

file_logger = logging.FileHandler('data/logs/output.log')
file_logger.setLevel(logging.DEBUG)

console_logger = logging.StreamHandler()
console_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_logger.setFormatter(formatter)
console_logger.setFormatter(formatter)

logger.addHandler(file_logger)
logger.addHandler(console_logger)


def print_params(params, msg):
    logger.info("###################################################")
    logger.info("############## {} #################".format(msg))
    logger.info("###################################################")
    logger.info("# Current parameters")
    logger.info("# Ground truth units: {}\t Feature Type: {}".format(params["gt_unit"], params["data_type"]))
    logger.info(
        "# Chosen model: {}\t learning rate: {}, decay: {}".format(params["model_name"], params["learning_rate"],
                                                                   params["decay"]))
    logger.info("# Epochs: {}\t Batch size: {}".format(params["nb_epoch"], params["batch_size"]))
    logger.info("# Optimizer: {}".format(params["optimizer_name"]))
    logger.info("# LSTM Parameters:")
    logger.info("# Units: {}, Dropout: {}".format(params["lstm_units"], params["lstm_dropout"]))
    logger.info("###################################################")
    logger.info("###################################################")


def print_results(mean, std, training, valid):
    tmp = "\n###################################################\n#################### Results ######################\n" \
          "###################################################\n# Best results\n# Train mean {}, standard deviation {}\n" \
          "# Valid mean {}, standard deviation {}\n# Test mean {}, standard deviation {}\n" \
          "###################################################".format(np.mean(training), np.std(training),
                                                                       np.mean(valid), np.std(valid), mean, std)
    logger.info(tmp)


def get_feature_length(data_type):
    resnet = False
    if data_type is "dct":
        features_length = 10
    elif data_type is "sift":
        features_length = 20
    elif data_type is "hog":
        features_length = 324
    elif data_type is "aam":
        features_length = 42
    elif data_type is "cnn":
        features_length = 2048
    elif data_type is "image":
        features_length = 0
        resnet = True
    return features_length, resnet


path = os.path.dirname(os.path.realpath(__file__))
seq_length = 190
image_shape = (224, 224, 3)

test_speakers = 9  # for whole data should be 9
valid_seq = 8  # for whole data should be 8
folds = 5  # should be 5


def main_gridsearch():
    # add values, which should be searched for
    grid = {"batch_size": [25],
            "nb_epoch": [75],
            "gt_unit": ["visemes", "phonemes"],
            "learning_rate": [1e-2],
            "decay": [1e-6],
            "lstm_units": [16, 64, 128],
            "data_type": ["hog", "dct", "aam", "sift"],
            "model_name": ["three-layer"],
            "lstm_dropout": [0.25],
            "optimizer_name": ["rmsprop"],
            "dense_units": [512],
            "dense_dropout": [0.5]}

    items = sorted(grid.items())

    best_mean = 0.
    best_std = 0
    best_train = []
    best_valid = []
    best_params = {}
    counter = 1

    keys, values = zip(*items)
    for v in product(*values):
        logger.info("###################################################")
        logger.info(
            "###################### {}/{} ########################".format(counter, len(list(product(*values)))))
        logger.info("###################################################")
        param = dict(zip(keys, v))
        print_params(param, "Current Parameters")

        learning_rate = param["learning_rate"]
        decay = param["decay"]
        model_name = param["model_name"]
        gt_unit = param["gt_unit"]
        data_type = param["data_type"]
        lstm_units = param["lstm_units"]
        lstm_dropout = param["lstm_dropout"]
        dense_units = param["dense_units"]
        dense_dropout = param["dense_dropout"]
        optimizer_name = param["optimizer_name"]
        batch_size = param["batch_size"]
        nb_epoch = param["nb_epoch"]

        features_length, resnet = get_feature_length(data_type)

        data = DataSet(path, seq_length=seq_length, data_type=data_type, unit=gt_unit)

        model = get_model(features_length, image_shape, len(data.classes), seq_length, model_name,
                          optimizer_name, learning_rate, decay, lstm_units, lstm_dropout,
                          dense_dropout, dense_units, resnet)

        tr, vs, test_scores = train(model, data, nb_test_speakers=test_speakers, nb_valid_seq=valid_seq,
                                    n_fold=folds, batch_size=batch_size, nb_epoch=nb_epoch, logger=logger)
        counter += 1
        if np.mean(test_scores) > best_mean:
            best_train = tr
            best_valid = tr
            best_mean = np.mean(test_scores)
            best_std = np.std(test_scores)
            best_params = param
            print_params(best_params, "New Best Parameters")
            print_results(best_mean, best_std, best_train, best_valid)

        with open("data/logs/log.csv", 'a') as f:
            out_writer = csv.writer(f, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            out_writer.writerow([data_type, np.mean(tr), np.std(tr), np.mean(vs), np.std(vs), np.mean(test_scores),
                                 np.std(test_scores), nb_epoch, batch_size, seq_length, learning_rate, decay, folds,
                                 model.layer, model.direction, lstm_units, lstm_dropout, optimizer_name, gt_unit])

    print_params(best_params, "Best Parameters")

    print_results(best_mean, best_std, best_train, best_valid)


def main_best_params():
    data_types = ["cnn"]  # ["hog", "dct", "aam", "sift", "cnn"]
    units = ["visemes", "phonemes"]
    counter = 1

    for data_type in data_types:
        for unit in units:
            param = {"batch_size": 25,
                     "nb_epoch": 50,
                     "gt_unit": unit,
                     "learning_rate": 1e-2,
                     "decay": 1e-6,
                     "lstm_units": 128,
                     "data_type": data_type,
                     "model_name": "three-layer",
                     "lstm_dropout": 0.25,
                     "optimizer_name": "rmsprop",
                     "dense_units": 512,
                     "dense_dropout": 0.5}

            logger.info("###################################################")
            logger.info("###################### {}/{} ########################"
                        .format(counter, len(data_types) * len(units)))
            logger.info("###################################################")
            print_params(param, "Current Parameters")

            learning_rate = param["learning_rate"]
            decay = param["decay"]
            model_name = param["model_name"]
            gt_unit = param["gt_unit"]
            data_type = param["data_type"]
            lstm_units = param["lstm_units"]
            lstm_dropout = param["lstm_dropout"]
            dense_units = param["dense_units"]
            dense_dropout = param["dense_dropout"]
            optimizer_name = param["optimizer_name"]
            batch_size = param["batch_size"]
            nb_epoch = param["nb_epoch"]

            train_scores = []
            valid_scores = []
            test_scores = []

            features_length, resnet = get_feature_length(data_type)

            data = DataSet(path, seq_length=seq_length, data_type=data_type, unit=gt_unit)

            for fold in range(folds):
                print("Starting fold {}".format(fold + 1))
                model = get_model(features_length, image_shape, len(data.classes), seq_length, model_name,
                                  optimizer_name, learning_rate, decay, lstm_units, lstm_dropout, dense_dropout,
                                  dense_units, resnet)

                tr, vs, ts = train(model, data, nb_test_speakers=test_speakers, nb_valid_seq=valid_seq, n_fold=fold,
                                  batch_size=batch_size, nb_epoch=nb_epoch, logger=logger, load_to_memory=False)

                train_scores.append(tr)
                valid_scores.append(vs)
                test_scores.append(ts)

            print_results(np.mean(test_scores), np.std(test_scores), train_scores, valid_scores)

            with open("data/logs/log.csv", 'a') as f:
                out_writer = csv.writer(f, delimiter=';', quotechar='\"', quoting=csv.QUOTE_MINIMAL,
                                        lineterminator='\n')
                out_writer.writerow(
                    [data_type, np.mean(tr), np.std(tr), np.mean(vs), np.std(vs), np.mean(test_scores),
                     np.std(test_scores), nb_epoch, batch_size, seq_length, learning_rate, decay, folds,
                     model.layer, model.direction, lstm_units, lstm_dropout, optimizer_name, gt_unit])
            counter += 1


if __name__ == '__main__':
    # main_gridsearch()
    # main_best_params()
    extract_files()
