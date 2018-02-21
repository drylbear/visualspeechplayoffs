import csv
import numpy as np
import pickle
import sys
import ast
import math
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from random import shuffle
from pprint import pprint
import os
import glob
from scipy import stats

uniq_visemes = ["v01", "v02", "v03", "v04", "v05", "v06", "v07", "v08", "sil", "gar", "tcd"]
uniq_phonemes = ["p", "b", "m", "f", "v", "w", "r", "W", "t", "d", "n", "l", "th", "dh", "s", "z", "ch", "jh", "sh", "zh", "j", "k", "g", "uh", "hh", "ee", "iy", "ay", "eh", "ah", "uw", "oo", "ax", "ua", "sil", "sp", "ow", "az", "oh", "ey", "ao", "aa", "ia", "er", "oy", "ng", "aw", "ih", "hh", "ae", "y"]

config = {
	"features": ["dct", "hog", "sift", "aam"], #dct, hog, sift, aam
	"train": 50,
	"test": 9,
	"folds": 5,
	"classifiers": ["nb", "rf", "svm"] # nb, svm, rf
}

class GroundTruth(object):
	
	def __init__(self, id):
		self.id = id
		self.phonemes = []
		self.visemes = []
		self.fvectors = []
		self.seq_order = []
		
class LClassifier(object):
	
	def __init__(self, pc, vc):
		self.c_phoneme = pc
		self.c_viseme = vc

d_cs = { "nb": LClassifier(GaussianNB(), GaussianNB()),
		"svm": LClassifier(svm.SVC(), svm.SVC()),
		"rf": LClassifier(RandomForestClassifier(max_depth=None, random_state=None), RandomForestClassifier(max_depth=None, random_state=None))
}

#output_dct, output_hog, output_sift
path = os.path.dirname(os.path.realpath(__file__))
path_frames = os.path.join(path, "..", "data", "sequence")
path_gt = os.path.join(path, "..", "data", "frame_by_frame.txt")

print("Started.")
for feature in config["features"]:
	ground_dict = {}
	files = sorted(glob.glob(os.path.join(path_frames, "*" + feature + "*.npy")))
	for file in files:
		#print("Processing: ", file)
		#features = open(os.path.join(path, file))
		spkr, seq, _ = file.split("\\")[-1].split("-")
		spkr = int(spkr.replace("M", "").replace("F", ""))
		#print("Speaker: ", spkr, "Sequence: ", seq)
		ground_dict[spkr] = GroundTruth(spkr)
		ground_dict[spkr].seq_order.append(seq)
		features = np.load(file).tolist()
		ground_dict[spkr].fvectors += features
			
	groundt = open(path_gt, "r")
	truth_imported = {}
	for line in groundt:
		try:
			gt = line.strip().split(" ")
			spkr = int(gt[0].replace("M", "").replace("F", ""))
			seq = gt[1]
			if spkr not in truth_imported.keys():
				truth_imported[spkr] = {}
			if seq not in truth_imported[spkr].keys():
				truth_imported[spkr][seq] = {"phonemes": [], "visemes": []}
			if spkr in ground_dict.keys():
				truth_imported[spkr][seq]["phonemes"].append(gt[3])
				truth_imported[spkr][seq]["visemes"].append(gt[4])
		except(ValueError):
			pass # Skip the header.
		
	for spkr in ground_dict.keys():
		for seq in ground_dict[spkr].seq_order:
			ground_dict[spkr].phonemes += truth_imported[spkr][seq]["phonemes"]
			ground_dict[spkr].visemes += truth_imported[spkr][seq]["visemes"]

	# Check we have enough speakers for the desired split.
	assert(config["train"] + config["test"] <= len(ground_dict.keys()))

	for classifier in config["classifiers"]:
		p_errors_fold = []
		v_errors_fold = []
		for fold in range(config["folds"]):
			# Shuffle so we get random results each time
			indices = list(ground_dict.keys())
			shuffle(indices)
			training = [indices[i] for i in range(config["train"])]
			test = [indices[i] for i in range(config["test"] + 1, config["test"] + config["train"])]
			#print("Training with:", training, " Testing on: ", test)

			# Assemble the data
			fs = []
			ps = []
			vs = []
			for speaker in training:
				fs += ground_dict[speaker].fvectors
				ps += ground_dict[speaker].phonemes
				vs += ground_dict[speaker].visemes
			
			fs_scaled = preprocessing.scale(fs)
			d_cs[classifier].c_phoneme.fit(fs_scaled, ps)
			d_cs[classifier].c_viseme.fit(fs_scaled, vs)

			p_errors_speaker = []
			v_errors_speaker = []
			for speaker in test:
				fs_spkr_scaled = preprocessing.scale(ground_dict[speaker].fvectors)
				
				pred_phonemes = d_cs[classifier].c_phoneme.predict(fs_spkr_scaled)
				phoneme_error = ((ground_dict[speaker].phonemes == pred_phonemes).sum() / len(pred_phonemes)) * 100
				p_errors_speaker.append(phoneme_error)

				pred_visemes = d_cs[classifier].c_viseme.predict(fs_spkr_scaled)
				viseme_error = ((ground_dict[speaker].visemes == pred_visemes).sum() / len(pred_visemes)) * 100
				v_errors_speaker.append(viseme_error)
			
			p_errors_fold.append(np.mean(p_errors_speaker))
			v_errors_fold.append(np.mean(v_errors_speaker))
			

		print("Results: ", feature, classifier, config["folds"])
		print("Unit: Mean Standard Error")
		print("Visemes: ", np.mean(v_errors_fold), stats.sem(v_errors_fold))
		print("Phonemes: ", np.mean(p_errors_fold), stats.sem(p_errors_fold))
		print("---")
