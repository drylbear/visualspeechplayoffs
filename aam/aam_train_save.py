"""
Train an AAM and save the resulting model to disk for later use.
"""
import numpy as np
from menpowidgets import visualize_aam, visualize_images, visualize_appearance_model # {?} visualize_shape_model
import menpo.io as mio
from menpo.landmark import labeller
from menpo.landmark import face_ibug_68_to_face_ibug_68_trimesh as ibug_face_66
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter, WibergInverseCompositional
from menpo.feature import hog, sparse_hog, igo
from menpodetect import load_dlib_frontal_face_detector
import os, glob
import dlib
import menpo
import json
import matplotlib.pyplot as plt
import sys

"""
To get a set of images to train on:
> ffmpeg -i sa1.mp4 -vf fps=2 train/out%d.png
where fps is the number of seconds between output frames.
"""

"""
Requires Menpo
> pip install menpo, menpofit, menpowidgets

To see the output visualisations you will need to run this in an ipython notebook and 
uncomment the line visualize_images(images) and add %matplotlib inline.
"""

"""
Uses only the lips, don't remove the other points from the landmarks array if you
want to track the whole face. You will then also have to change the feature extraction to 
get just the lip landmarks at the end.
"""

class DebugPrinter:
	def __init__(self, dbug=False):
		self.dbug = dbug
	def __call__(self, info):
		if self.dbug:
			print(info)

log = DebugPrinter(False)

dlib_detector = load_dlib_frontal_face_detector()

detector = dlib.get_frontal_face_detector()

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  

MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  
predictor = dlib.shape_predictor(PREDICTOR_PATH)

IMAGE_SIZE = (48,48)

path = os.path.dirname(os.path.realpath(__file__))
path_speakers = os.path.join(path, "..", "data", "aam")
path_save = os.path.join(path, "..", "data", "aam", "out")
speakers = os.listdir(path_speakers)

for speaker in speakers:
	print("Processing Speaker: ", speaker)
	images = []
	for i in mio.import_images(os.path.join(path_speakers, speaker),  max_images=20, verbose=False): # max_images = 10
		if i.n_channels == 3:
			i = i.as_greyscale(mode='luminosity')
		
		dlib_detector(i)
		
		rect = detector(i.as_imageio(), 1)[0]
		
		landmarks = np.matrix([[p.y, p.x] for p in predictor(i.as_imageio(), rect).parts()])
		
		landmarks = landmarks[MOUTH_OUTLINE_POINTS + MOUTH_INNER_POINTS]
		
		landmarks_pc = menpo.shape.PointCloud(landmarks)
		
		# The image will have two sets of lanmarks, PTS with the labeleld points and dlib_0 with the 
		# original face bounding box.
		i.landmarks['PTS'] = landmarks_pc #lm_lbl
		
		# Crop to lips 
		i = i.crop_to_landmarks(group='PTS', boundary=10) #dlib_0 for box
		i = i.resize(IMAGE_SIZE)
		
		images.append(i)
		
	#visualize_images(images)

	# IMPORTANT! Adjust these parameters, they are currently set to low to avoid MemoryError on my laptop
	# max_shape_components=20, max_appearance_components=150 features: mphog, sparse_hog, fast_dsift
	aam = HolisticAAM(images, diagonal=None,
					  scales=1, group='PTS', holistic_features=igo, verbose=True,
					  max_shape_components=20, max_appearance_components=150)   # 20, 150
	log(aam)
	outpath = os.path.join(path_save, speaker + "-aam.pkl")
	menpo.io.export_pickle(aam, outpath, overwrite=True, protocol=2)
	print("\n\nSaved AAM to: ", outpath)