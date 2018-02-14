from __future__ import print_function
import numpy as np
import os
from cntk import load_model
from TransferLearning import *


# define base model location and characteristics
base_folder = os.path.dirname(os.path.abspath(__file__))
#base_model_file = os.path.join(base_folder, "..", "PretrainedModels", "ResNet_50.model")
#base_model_file = os.path.join(base_folder, "Output", "TransferLearning.model")
new_model_file = os.path.join(base_folder, "Output", "ResNet18_CNTK_transferedPN_0.model")
feature_node_name = "features"
last_hidden_node_name = "z.x"
image_height = 224
image_width = 224
num_channels = 3
# get class mapping and map files from train and test image folder
train_map_file = "D:/cntk/CNTK-2-0-beta11-0-Windows-64bit-GPU/cntk/Examples/Image/DataSets/NCL/train1.txt"
test_map_file = "C:/VSP/mapPNtestRandom.txt"
class_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19','20', '21', '22', '23', '24', '25', '26', '27', '28', '29','30', '31', '32', '33', '34', '35', '36', '37', '38', '39','40', '41', '42', '43', '44']

def format_output_line(img_name, true_class, probs, class_mapping, top_n=3):
    class_probs = np.column_stack((probs, class_mapping)).tolist()
    class_probs.sort(key=lambda x: float(x[0]), reverse=True)
    top_n = min(top_n, len(class_mapping)) if top_n > 0 else len(class_mapping)
    true_class_name = class_mapping[true_class] if true_class >= 0 else 'unknown'
    line = '[{"class": "%s", "predictions": {' % true_class_name
    for i in range(0, top_n):
        line = '%s"%s":%.3f, ' % (line, class_probs[i][1], float(class_probs[i][0]))
    line = '%s}, "image": "%s"}]\n' % (line[:-2], img_name.replace('\\', '/'))
    return line

# if __name__ == '__main__':
	# # evaluate test images
	# trained_model = load_model(new_model_file)
	# results_file = os.path.join(base_folder, "Output", "predictions2.txt")
	# with open(results_file, 'w') as output_file:
		# with open(test_map_file, "r") as input_file:
			# for line in input_file:
				# tokens = line.rstrip().split('\t')
				# img_file = tokens[0]
				# true_label = int(tokens[1])
				# probs = eval_single_image(trained_model, img_file, image_width, image_height)

				# formatted_line = format_output_line(img_file, true_label, probs, class_mapping)
				# output_file.write(formatted_line)

	# print("Done. Wrote output to %s" % results_file)
	
if __name__ == '__main__':
	# evaluate test images
	trained_model = load_model(new_model_file)
	output_file = os.path.join(base_folder, "Output", "predOutput1.txt")
		# Evaluate the test set
	eval_test_images(trained_model, output_file, test_map_file,image_width,image_height)
	print("Done. Wrote output to %s" % output_file)