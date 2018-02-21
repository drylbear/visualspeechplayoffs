# Data Line [0-9]+ [0-9]+ [a-z]+
# File Name Line \".+\"
# Speaker ID: [0-9][0-9](M|F)
# Framerate: 29.97fps

import re
import math
import sys
import operator

fps = 29.97 

datapath = "..\\features\\data\\01M\\straightcam"

data_line = re.compile("[0-9]+ [0-9]+ [a-z]+")
file_name = re.compile("\\\".+\\\"")
speak_id = re.compile("[0-9][0-9][MF]{1}")

mapping = {}
with open('mapping_final.txt', 'r') as vmap: # 'woodward_d.vmap'
	for line in vmap:
		line = line.strip().split(" ")
		viseme = line[0]
		phonemes = line[1:]
		for phoneme in phonemes:
			mapping[phoneme] = viseme
		
#volunteer_labelfiles		
with open('volunteer_labelfiles.mlf', 'r') as labeldata:
	out = open("labelled_data3_newmapping.txt", "w")
	fbf = open("frame_by_frame3_newmapping.txt", "w")
	fbf.write("speaker file frame phoneme viseme\n")
	name = "NO_NAME"
	for line in labeldata:
		if file_name.match(line):
			rec_file_name = line.split("/")[-1].replace(".rec\"", "").strip()
			name = speak_id.findall(line)[0]
			"""
			word_script = datapath + "\\" + rec_file_name.upper() + ".txt"
			end_time_words = {}
			try:
				with open(word_script, 'r') as words:
					for l in words:
						l = l.strip().split(" ")
						end_time_words[(int(l[1]) + 1000)/ 10000] = l[2] 
			except FileNotFoundError:
				pass
			#print(end_time_words)
			#sys.exit()
			"""
			
		elif data_line.match(line):
			data = line.strip().split(" ")
			try:
				data.append(mapping[data[2]])
			except:
				print("Unable to map phoneme. " + data[2])
				exit()
			out.write(name + " " + " ".join(data) + "\n")
		
			start_time = int(data[0]) / 10E6
			end_time = int(data[1]) / 10E6
			start_frame = math.floor(start_time * fps)
			end_frame = math.floor(end_time * fps)# - 1
			#print(start_time, end_time, start_frame, end_frame)
			#sys.exit()
			
			"""
			# Lookup word.
			word_match = "sil"
			if (data[2] != "sil"):
				word_match = "gar"
				sorted_et = sorted(end_time_words.items(), key=operator.itemgetter(0))
				
				for el in sorted_et:
					if el[0] < end_time:
						word_match = el[1]
					else:
						print("Matching Time: ")
						print(start_time, end_time, start_frame, end_frame)
						print(sorted_et)
						#sys.exit()
						break
			#print(word_match)
			"""
			
			for i in range(start_frame, end_frame):
				fbf.write(name + " " + rec_file_name + " " + str(i) + " " + data[2] + " " + data[3] + "\n") # " " + word_match + 
			
print("Done")
			
