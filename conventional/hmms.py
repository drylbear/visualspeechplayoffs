import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)    
    import csv
    import numpy as np
    import pickle
    import sys
    import ast
    import math
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm, preprocessing
    from hmmlearn import hmm
    from sklearn.ensemble import RandomForestClassifier
    from random import shuffle
    from pprint import pprint
    import os
    import glob
    from scipy import stats
    from sklearn.exceptions import NotFittedError

    import hmmlearn # Need the latest from Github to fix errors in log-liklihood computation.
assert(hmmlearn.__version__ == "0.2.1")

uniq_visemes = ["v01", "v02", "v03", "v04", "v05", "v06", "v07", "v08", "sil", "gar", "tcd"]
uniq_phonemes = ["p", "b", "m", "f", "v", "w", "r", "W", "t", "d", "n", "l", "th", "dh", "s", "z", "ch", "jh", "sh", "zh", "j", "k", "g", "uh", "hh", "ee", "iy", "ay", "eh", "ah", "uw", "oo", "ax", "ua", "sil", "sp", "ow", "az", "oh", "ey", "ao", "aa", "ia", "er", "oy", "ng", "aw", "ih", "hh", "ae", "y"]
	
def speaker_to_hmminfo(fvs, ps, vs):
	# Visemes
	v_classes = {}
	v_counts = dict.fromkeys(uniq_visemes, 0)
	p_counts = dict.fromkeys(uniq_phonemes, 0)
	for i in range(len(fvs)):
		if not vs[i] in v_classes.keys():
			v_classes[vs[i]] = { "data": [], "lengths": [] }

		v_classes[vs[i]]["data"].append(fvs[i])
		v_counts[vs[i]] += 1
		#v_classes[vs[i]]["lengths"].append(len(fvs[i]))
			
	for v in v_classes.keys():
		v_classes[v]["lengths"].append(v_counts[v])
	
	# Phonemes
	p_classes = {}
	for i in range(len(fvs)):
		if not ps[i] in p_classes.keys():
			p_classes[ps[i]] = { "data": [], "lengths": [] }
			
		p_classes[ps[i]]["data"].append(fvs[i])
		p_counts[ps[i]] += 1
			
	for p in p_classes.keys():
		p_classes[p]["lengths"].append(p_counts[p])
			
	return v_classes, p_classes
	
config = {
	"features": ["dct", "hog", "sift", "aam"],
	"train": 50,
	"test": 9,
	"folds": 5,
	"classifiers": ["hmm"]  # nb, svm, rf, hmm
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


d_cs = {"nb": LClassifier(GaussianNB(), GaussianNB()),
		"svm": LClassifier(svm.SVC(), svm.SVC()),
		"rf": LClassifier(RandomForestClassifier(max_depth=None, random_state=None),
						  RandomForestClassifier(max_depth=None, random_state=None))}

# output_dct, output_hog, output_sift
path = os.path.dirname(os.path.realpath(__file__))
path_frames = os.path.join(path, "..", "data", "sequence")
path_gt = os.path.join(path, "..", "data", "frame_by_frame.txt")

bad_vs = set()
bad_ps = set()
def predict_features(fs, unit):
	predict_seq = []
	for f in fs:
		f = f.reshape(1, -1)
		score = []
		if unit == "viseme":
			for v in uniq_visemes:
				try:
					# log-likelihood higher is better
					score.append(v_hmms[v].score(f))
				except (NotFittedError):# , ValueError
					score.append(float('-inf'))
					bad_vs.add(v)
			prediction = score.index(max(score))
			#print(score)
			predict_seq.append(uniq_visemes[prediction])
		else:
			for p in uniq_phonemes:
				try:
					# log-likelihood higher is better
						score.append(p_hmms[p].score(f))
				except (NotFittedError): #, ValueError
					score.append(float('-inf'))
					bad_ps.add(p)
			prediction = score.index(max(score))
			predict_seq.append(uniq_phonemes[prediction])
	return np.array(predict_seq)

print("Started.")
for feature in config["features"]:
	v_hmms = {}
	p_hmms = {}
	N_HMM_COMPONENTS = 3
	for viseme in uniq_visemes:
		v_hmms[viseme] = hmm.GaussianHMM(n_components=N_HMM_COMPONENTS)
	for phoneme in uniq_phonemes:
		p_hmms[phoneme] = hmm.GaussianHMM(n_components=N_HMM_COMPONENTS)


	ground_dict = {}
	files = sorted(glob.glob(os.path.join(path_frames, "*" + feature + "*.npy")))
	for file in files:
		# print("Processing: ", file)
		# features = open(os.path.join(path, file))
		spkr, seq, _ = file.split("\\")[-1].split("-")
		spkr = int(os.path.basename(spkr.replace("M", "").replace("F", "")))
		# print("Speaker: ", spkr, "Sequence: ", seq)
		ground_dict[spkr] = GroundTruth(spkr)
		ground_dict[spkr].seq_order.append(seq)
		features = np.load(file).tolist()
		ground_dict[spkr].fvectors += features

	groundt = open(path_gt, "r")
	truth_imported = {}
	for line in groundt:
		try:
			gt = line.strip().split(" ")
			spkr = int(os.path.basename(gt[0].replace("M", "").replace("F", "")))
			seq = gt[1]
			if spkr not in truth_imported.keys():
				truth_imported[spkr] = {}
			if seq not in truth_imported[spkr].keys():
				truth_imported[spkr][seq] = {"phonemes": [], "visemes": []}
			if spkr in ground_dict.keys():
				truth_imported[spkr][seq]["phonemes"].append(gt[3])
				truth_imported[spkr][seq]["visemes"].append(gt[4])
		except(ValueError):
			pass  # Skip the header.

	for spkr in ground_dict.keys():
		for seq in ground_dict[spkr].seq_order:
			ground_dict[spkr].phonemes += truth_imported[spkr][seq]["phonemes"]
			ground_dict[spkr].visemes += truth_imported[spkr][seq]["visemes"]

	# Check we have enough speakers for the desired split.
	print(len(ground_dict.keys()))
	assert (config["train"] + config["test"] <= len(ground_dict.keys()))

	for classifier in config["classifiers"]:
		p_errors_fold = []
		v_errors_fold = []
		for fold in range(config["folds"]):
			# Shuffle so we get random results each time
			indices = list(ground_dict.keys())
			shuffle(indices)
			training = [indices[i] for i in range(config["train"])]
			test = [indices[i] for i in range(config["test"], config["test"] + config["train"])]
			#print("Training with:", training, " Testing on: ", test)
			#print("Training classifier: ",  classifier, " in fold: ", fold)

			# Assemble the data
			fs = []
			ps = []
			vs = []
			lengths = []
			# TODO: Need to incorporate lengths!!!!! - Should each speaker be an observation,
			# or each sequence?
			v_classes, p_classes = (None, None)
			for speaker in training:
				fs_scaled = preprocessing.scale(np.asarray(ground_dict[speaker].fvectors, dtype=np.float32))
				v_c, p_c = speaker_to_hmminfo(fs_scaled, ground_dict[speaker].phonemes, ground_dict[speaker].visemes)
				if (v_classes is None):
					v_classes = v_c
				else:
					for v in v_c.keys():
						if v not in v_classes.keys():
							v_classes[v] = {"data": v_c[v]["data"], "lengths": v_c[v]["lengths"] }
						else:
							v_classes[v]["lengths"] += v_c[v]["lengths"]
							v_classes[v]["data"] += v_c[v]["data"]
							
				if (p_classes is None):
					p_classes = p_c
				else:
					for p in p_c.keys():
						if p not in p_classes.keys():
							p_classes[p] = {"data": p_c[p]["data"], "lengths": p_c[p]["lengths"] }
						else:
							p_classes[p]["lengths"] += p_c[p]["lengths"]
							p_classes[p]["data"] += p_c[p]["data"]
							
				"""
				fs += ground_dict[speaker].fvectors
				ps += ground_dict[speaker].phonemes
				vs += ground_dict[speaker].visemes
				"""
			
			#pprint(p_classes)
			#sys.exit()
			"""
			fs_scaled = preprocessing.scale(np.asarray(fs, dtype=np.float32))
			v_classes, p_classes = speaker_to_hmminfo(fs_scaled, ps, vs)
			"""		
			for v in v_classes.keys():
				data = np.array(v_classes[v]["data"])
				# Can't train if not enough data in set 
				if not data.shape[0] < N_HMM_COMPONENTS:
					v_hmms[v].fit(np.array(v_classes[v]["data"]), lengths=v_classes[v]["lengths"])
					# v_hmms[v].fit(v_classes[v]["data"], lengths=v_classes[v]["lengths"])
			
					sane_transmat = np.sum(np.sum(v_hmms[v].transmat_, axis=1))
					if not np.isclose(sane_transmat,3.0):
						# Drop bad models... (leave unfitted, error caught later)
						v_hmms[v] = hmm.GaussianHMM(n_components=N_HMM_COMPONENTS)
						
			for p in p_classes.keys():
				data = np.array(p_classes[p]["data"])
				if not data.shape[0] < N_HMM_COMPONENTS:
					# p_hmms[p].fit(p_classes[p]["data"], lengths=p_classes[p]["lengths"])
					p_hmms[p].fit(np.array(p_classes[p]["data"]), lengths=p_classes[p]["lengths"])
					
					
					sane_transmat = np.sum(np.sum(p_hmms[p].transmat_, axis=1))
					if not np.isclose(sane_transmat,3.0):
						# Drop bad models... (leave unfitted, error caught later)
						p_hmms[p] = hmm.GaussianHMM(n_components=N_HMM_COMPONENTS)
						#print(p_hmms[p].transmat_)
						#print(np.sum(p_hmms[p].transmat_, axis=1))

			p_errors_speaker = []
			v_errors_speaker = []
			for speaker in test:
				fs_spkr_scaled = preprocessing.scale(ground_dict[speaker].fvectors)
				pred_phonemes = predict_features(fs_spkr_scaled, "phoneme")
				pred_visemes = predict_features(fs_spkr_scaled, "viseme")
				
				#print(pred_visemes)
				#print(ground_dict[speaker].visemes)
				#print(pred_phonemes)
				
				phoneme_error = ((ground_dict[speaker].phonemes == pred_phonemes).sum() / len(pred_phonemes)) * 100
				p_errors_speaker.append(phoneme_error)

				viseme_error = ((ground_dict[speaker].visemes == pred_visemes).sum() / len(pred_visemes)) * 100
				v_errors_speaker.append(viseme_error)

			p_errors_fold.append(np.mean(p_errors_speaker))
			v_errors_fold.append(np.mean(v_errors_speaker))

		print("Results: ", feature, classifier, config["folds"])
		print("Unit: Mean Standard Error")
		print("Visemes: ", np.mean(v_errors_fold), stats.sem(v_errors_fold))
		print("Phonemes: ", np.mean(p_errors_fold), stats.sem(p_errors_fold))
		print("---")
	
print("Unable to build HMM for phonemes: ", bad_ps)
print("Unable to build HMM for visemes: ", bad_vs)
