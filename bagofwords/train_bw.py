import cv2
import numpy as np
import os
from bow import BagOfWords
from random import shuffle
import sys
import pickle

FPS = 20  # Frames Per Speaker

sift = cv2.xfeatures2d.SIFT_create()
bow = BagOfWords(20, l1_norm=False)

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, "..", "data", "train")

speakers = os.listdir(path)
flag = None

for speaker in speakers:
    frame_path = os.path.join(path, speaker)
    frames = os.listdir(frame_path)
    indices = list(range(len(frames)))
    shuffle(indices)
    random_frames = [frames[i] for i in indices[:FPS]]

    sift_features = []
    for image in random_frames:
        image_path = os.path.join(frame_path, image)
        img = cv2.imread(image_path, 0)
        img8 = cv2.convertScaleAbs(img)
        _, sift_vec = sift.detectAndCompute(img8, None)
        # sift_vec /= sift_vec.sum(axis=1).reshape(-1, 1)
        if sift_vec is not None:
            sift_features += sift_vec.tolist()

    bow.partial_fit(sift_features)
# f_bow = bow.transform(sift_features)
# print(f_bow)
# print(len(f_bow))
# input()

pickle.dump(bow, open('bow_model.pkl', 'wb'))
print("Saved model to bow_model.pkl")

bow_test = pickle.load(open("bow_model.pkl", 'rb'))
print(bow_test.transform(sift_features[0]))
