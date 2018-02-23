import numpy as np
import os
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN, EarlyStopping
from sklearn.metrics import f1_score
import logging
from time import time

from dataset import DataSet


class TimingCallback(Callback):
    def __init__(self):
        self.logs = []
        self.start_time = 0.

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time() - self.start_time)


def train(rm, data, nb_test_speakers=9, nb_valid_seq=8, batch_size=32, nb_epoch=100, n_fold=0,
          logger=logging.getLogger('VisuSpeech.Train'), load_to_memory=False):
    logs_dir = os.path.join("data", "logs")
    run = 'Units_' + str(data.unit) + '-Feature_' + str(data.data_type) + '-' + rm.model_name + '-Fold_' + str(n_fold)
    history_path = os.path.join(logs_dir, "history-" + run + ".npy")
    checkpoint_path = os.path.join(logs_dir, "inception-" + run + '.hdf5')

    # split into train, validation and test
    x_train_com, y_train_com, x_test, y_test = data.split_train_test(nb_test_speakers)
    x_train, y_train, x_valid, y_valid = DataSet.split_train_valid(x_train_com, y_train_com, nb_valid_seq)

    # print some info
    print("Number of Speakers in train set: ", len(set([s[1] for s in x_train])))
    print("Number of Speakers in test set: ", len(set([s[1] for s in x_test])))

    print('Number of sequences for train and valid: ', len(x_train_com))
    print('Number of sequences for train: ', len(x_train))
    print('Number of sequences for valid: ', len(x_valid))

    if load_to_memory:
        x_train, y_train, x_valid, y_valid, x_test, y_test = data.load_to_memory(x_train, y_train, x_valid,
                                                                                 y_valid, x_test, y_test)

    lr = None
    if os.path.exists(history_path):
        history_load = np.load(history_path)
        lr = history_load.item().get("lr")[-1]

    # get model and compile
    rm.compile(lr)
    model = rm.model

    # define generators for train, validation and testing
    def train_generator():
        while True:
            for start in range(0, len(x_train), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(x_train))
                train_batch = x_train[start:end]
                for sample in train_batch:
                    seq = data.get_sequence(sample)
                    x_batch.append(seq)
                    y_batch.append(data.get_class_one_hot(sample[4]))
                yield np.asarray(x_batch, np.float32), np.asarray(y_batch, np.uint8)

    def valid_generator():
        while True:
            for start in range(0, len(x_valid), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(x_valid))
                train_batch = x_valid[start:end]
                for sample in train_batch:
                    seq = data.get_sequence(sample)
                    x_batch.append(seq)
                    y_batch.append(data.get_class_one_hot(sample[4]))
                yield np.asarray(x_batch, np.float32), np.asarray(y_batch, np.uint8)

    def test_generator():
        while True:
            for start in range(0, len(x_test), batch_size):
                x_batch = []
                end = min(start + batch_size, len(x_test))

                test_batch = x_test[start:end]
                for sample in test_batch:
                    seq = data.get_sequence(sample)
                    x_batch.append(seq)
                yield np.asarray(x_batch, np.float32)

    # define callbacks for adaptive learning rate, saving the model, an early stopper and a terminate on NaN
    tc = TimingCallback()
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1, min_lr=1e-7),
                 ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True),
                 EarlyStopping(patience=10, verbose=1),
                 TerminateOnNaN(), tc]

    # calculate steps for train, validation and testing
    train_steps = len(x_train) // batch_size
    valid_steps = len(x_valid) // batch_size
    test_steps = len(x_test) // batch_size

    # Load weights if possible
    if os.path.exists(checkpoint_path):
        print("Loaded weights")
        model.load_weights(filepath=checkpoint_path)

    # Fit model
    if load_to_memory:
        history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_valid, y_valid),
                            epochs=nb_epoch, verbose=1, callbacks=callbacks)
    else:
        history = model.fit_generator(train_generator(), train_steps, epochs=nb_epoch, verbose=1, callbacks=callbacks,
                                      validation_data=valid_generator(), validation_steps=valid_steps)

    print("Mean timing for epoch: ", np.mean(tc.logs))
    # Run train, validation and test predictions
    if load_to_memory:
        print('Running train predictions on fold {}'.format(n_fold + 1))
        preds_train = model.predict(x_test, batch_size=batch_size, verbose=1)

        print('Running validation predictions on fold {}'.format(n_fold + 1))
        preds_valid = model.predict(x_test, batch_size=batch_size, verbose=1)

        print('Running test predictions on fold {}'.format(n_fold + 1))
        preds_test_fold = model.predict(x_test, batch_size=batch_size, verbose=1)
    else:
        print('Running train predictions on fold {}'.format(n_fold + 1))
        preds_train = model.predict_generator(generator=train_generator(), steps=train_steps, verbose=1)

        print('Running validation predictions on fold {}'.format(n_fold + 1))
        preds_valid = model.predict_generator(generator=valid_generator(), steps=valid_steps, verbose=1)

        print('Running test predictions on fold {}'.format(n_fold + 1))
        preds_test_fold = model.predict_generator(generator=test_generator(), steps=test_steps, verbose=1)

    # Calculate scores
    v_score = 0.
    t_score = 0.
    te_score = 0.
    for yt, yp in zip(np.argmax(y_valid, axis=2), np.argmax(preds_valid, axis=2)):
        v_score += f1_score(yt, yp, average='weighted')
    for yt, yp in zip(np.argmax(y_train, axis=2), np.argmax(preds_train, axis=2)):
        t_score += f1_score(yt, yp, average='weighted')
    for yt, yp in zip(np.argmax(y_test, axis=2), np.argmax(preds_test_fold, axis=2)):
        te_score += f1_score(yt, yp, average='weighted')

    valid_score = v_score / preds_valid.shape[0]
    train_score = t_score / preds_train.shape[0]
    test_score = te_score / preds_test_fold.shape[0]
    logger.info('Val Score:{} for fold {}'.format(valid_score, n_fold))
    logger.info('Train Score: {} for fold {}'.format(train_score, n_fold))
    logger.info('Test Score: {} for fold {}'.format(test_score, n_fold))

    logger.info('Avg Train Score:{} after {} folds'.format(valid_score, n_fold + 1))

    logger.info('Avg Val Score:{} after {} folds'.format(train_score, n_fold + 1))

    logger.info('Avg Test Score:{} after {} folds'.format(test_score, n_fold + 1))

    # Save history object
    if os.path.exists(history_path):
        history_load = np.load(history_path)
        history_load.item().get("lr").extend(history.history["lr"])
        history_load.item().get("val_loss").extend(history.history["val_loss"])
        history_load.item().get("loss").extend(history.history["loss"])
        history_load.item().get("acc").extend(history.history["acc"])
        history_load.item().get("val_acc").extend(history.history["val_acc"])
        history.history = history_load

    np.save(history_path, history.history)

    # res_seq = np.asarray([data.get_sequence(data, x_test[0])])
    # res = model.predict(res_seq, batch_size=1, verbose=1)[0]
    # log_prediction(data.get_class_from_prediction(np.asarray(y_test[0])),
    #               data.get_class_from_prediction(res), logger)

    return train_score, valid_score, test_score


def log_prediction(groundtruth, prediction, logger):
    tmp = "\n{0:^15} | {1:^15}\n".format("Ground truth", "Prediction")
    for gt, p in zip(groundtruth, prediction):
        tmp += "{0:^15} | {1:^15}\n".format(gt, p)

    logger.info(tmp)


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

# based on
# https://www.kaggle.com/jamesrequa/keras-k-fold-inception-v3-1st-place-lb-0-99770/code
# and https://github.com/harvitronix/five-video-classification-methods/blob/master/validate_cnn.py
