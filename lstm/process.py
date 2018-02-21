import dlib
import cv2
import numpy as np
from utils import rect_to_bb, shape_to_np, FACIAL_LANDMARKS_IDXS, image_resize
import os
from skimage.feature import hog
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import preprocess_input
import menpo.io as mio

import menpo
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter, WibergInverseCompositional
from utils import DebugPrinter
                                
log = DebugPrinter(False)

from bow import BagOfWords
import pickle
bwrds = pickle.load(open("bow_model.pkl", 'rb'))
BOW_LENGTH = 20

class FaceDetectionError(Exception):
    pass


class MouthExtractor:
    def __init__(self, path=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.detector = dlib.get_frontal_face_detector()
        if path is None:
            self.predictor = dlib.shape_predictor(
                os.path.join(dir_path, 'predictor/shape_predictor_68_face_landmarks.dat'))
        else:
            self.predictor = dlib.shape_predictor(
                os.path.join(path + 'predictor/shape_predictor_68_face_landmarks.dat'))

    def get_face(self, image, width=48, height=48):
        rect = self.detector(image, 1)[0]  # first face
        (x, y, w, h) = rect_to_bb(rect)
        face = image[y:y + h, x:x + w]
        return rect, cv2.resize(face, (width, height))

    def get_mouth(self, image, width=48, height=48, padding=10, jpg=''):
        rect = self.detector(image, 1)[0]  # first face

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(image, rect)
        shape = shape_to_np(shape)

        (i, j) = FACIAL_LANDMARKS_IDXS['mouth']

        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y - padding:y + h + padding, x - padding:x + w + padding]

        return cv2.resize(roi, (width, height))  # image_resize(roi, width, height, inter=cv2.INTER_CUBIC)


mouth_ext = MouthExtractor()
sift = cv2.xfeatures2d.SIFT_create()


def process_image(jpg, target_shape, region='mouth', padding=10, rec=False):
    ''' Crop image to region of interest

    :param jpg: the image to crop
    :param target_shape: the shape the image should have
    :param region: if we want face or mouth. Possible: 'mouth', 'face'
    :param padding:
    :return:
    '''
    h, w, c = target_shape
    if c == 1:
        image = cv2.imread(jpg, 0)  # load as grayscale
    elif c == 3:
        image = cv2.imread(jpg, 0) # load with 3 channels for resnet
    else:
        raise ValueError("Can't have image channel other than 1 or 3")

    if region is 'mouth':
        try:
            img = mouth_ext.get_mouth(image, h, w, padding, jpg)
        except IndexError:  # assume no face was found
            raise FaceDetectionError("Could not find face in {}".format(jpg))
    elif region is 'face':
        img = mouth_ext.get_face(image, h, w)
    else:
        raise ValueError('Unknown region, we want to crop to: {}, use \'mouth\' or \’face\’'.format(region))

    return img


def calc_cnn(jpg, model):
    img = load_img(jpg, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()


def calc_dct(jpg, coeffs=10):
    img = cv2.imread(jpg, 0)
    imf = np.float32(img) / 255.0
    dst = cv2.dct(imf)
    imgi = np.uint8(dst) * 255.0

    return zig(imgi, coeffs)


def calc_sift(jpg):
    img = cv2.imread(jpg, 0)

    # calculate SIFT
    img8 = cv2.convertScaleAbs(img)
    _, sift_vec = sift.detectAndCompute(img8, None)
    if (sift_vec is None):
        fvec = [0] * BOW_LENGTH
    else:
        fvec = bwrds.transform(sift_vec)
    return fvec


def calc_hog(jpg):
    img = cv2.imread(jpg, 0)
    hog_vec = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm="L2-Hys",
                  visualise=False, transform_sqrt=False, feature_vector=True)
    return hog_vec


def zig(matrix, n):
    ary = np.empty_like(matrix)
    ary[:] = matrix
    ary = ary.transpose()
    ary = np.flipud(ary)
    selection = np.concatenate(
        [np.diagonal(ary, k)[::(2 * (k % 2) - 1)] for k in range(1 - ary.shape[0], ary.shape[0])])
    return selection[:n]


def calc_aam(jpgs, speakername):
    # AAMs - have to be computed sequentially.
    aam_features = []
    aam_landmarks = []
    aam_path = os.path.join('data', 'aam', speakername + '-aam.pkl')
    aam = mio.import_pickle(aam_path)

    # Compute the first frame (no previous shape)
    aam_v, last_shape = calc_aam_single_jpg(jpgs[0], aam)
    aam_features.append(aam_v)
    landmarks = last_shape.tojson()["landmarks"]["points"]
    aam_landmarks.append(landmarks)

    # Compute the remaining frames using the previous shape.
    for i, jpg in enumerate(jpgs[1:]):
        # print("{}th jpg".format(i + 1))
        aam_v, last_shape = calc_aam_single_jpg(jpg, aam, last_shape)
        aam_features.append(aam_v)
        landmarks = last_shape.tojson()["landmarks"]["points"]
        aam_landmarks.append(landmarks)
    return np.asarray(aam_features), np.asarray(aam_landmarks)


def calc_aam_single_jpg(jpg, aam, shape_last_frame=None):
    fitter = LucasKanadeAAMFitter(aam, lk_algorithm_cls=WibergInverseCompositional, n_shape=None, n_appearance=None)
    log(fitter)

    image = menpo.io.import_image(jpg)
    if image.n_channels == 3:
        image = image.as_greyscale(mode='luminosity')

    # Fit AAM to Test Image
    initial_bbox = menpo.shape.bounding_box(*image.bounds())

    if shape_last_frame is None:
        result = fitter.fit_from_bb(image, initial_bbox, max_iters=80)
    else:
        result = fitter.fit_from_shape(image, shape_last_frame, max_iters=80)

    # Get the lip landmark positions.
    fs = result.final_shape
    log(result)
    log("Landmarks Parameters")
    log(fs.landmarks)
    
    log("Shape/Appearance Parameters")
    log(result.shape_parameters[-1])
    log(result.appearance_parameters[-1])
    fvec = np.concatenate((result.shape_parameters[-1], result.appearance_parameters[-1]))
    
    return (fvec, fs)
