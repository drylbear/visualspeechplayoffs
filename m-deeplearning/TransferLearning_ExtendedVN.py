# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os
from cntk import load_model
from TransferLearningVN import *


# define base model location and characteristics
base_folder = os.path.dirname(os.path.abspath(__file__))
base_model_file = os.path.join(base_folder,"ResNet18_ImageNet_CNTK.model")
feature_node_name = "features"
last_hidden_node_name = "z.x"
image_height = 224
image_width = 224
num_channels = 3

# define data location and characteristics
train_image_folder = os.path.join(base_folder)
test_image_folder = os.path.join(base_folder)
file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.VNg', '.VNG']

def create_map_file_from_folder(root_folder, class_mapping, include_unknown=False):
    map_file_name = os.path.join(root_folder, "map.txt")
    lines = []
    for class_id in range(0, len(class_mapping)):
        folder = os.path.join(root_folder, class_mapping[class_id])
        if os.path.exists(folder):
            for entry in os.listdir(folder):
                filename = os.path.join(folder, entry)
                if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                    lines.append("{0}\t{1}\n".format(filename, class_id))

    if include_unknown:
        for entry in os.listdir(root_folder):
            filename = os.path.join(root_folder, entry)
            if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                lines.append("{0}\t-1\n".format(filename))

    lines.sort()
    with open(map_file_name , 'w') as map_file:
        for line in lines:
            map_file.write(line)

    return map_file_name

def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    classes.sort()
    return np.asarray(classes)

def format_output_line(img_name, true_class, probs, class_mapping, top_n=3):
    class_probs = np.column_stack((probs, class_mapping)).tolist()
    class_probs.sort(key=lambda x: float(x[0]), reverse=True)
    top_n = min(top_n, len(class_mapping)) if top_n > 0 else len(class_mapping)
    true_class_name = class_mapping[true_class] if true_class >= 0 else 'unknown'
    line = '[{"class": "%s", "predictions": {' % true_class_name
    for i in range(0, top_n):
        line = '%s"%s":%.3f, ' % (line, class_probs[i][1], float(class_probs[i][0]))
    line = '%s}, "image": "%s"}]\n' % (line[:-2], img_name.replace('\\', '/').rsplit('/', 1)[1])
    return line

def train_and_eval(_base_model_file, _train_image_folder, _test_image_folder, _results_file, _new_model_file,fold,train_map_file,test_map_file, testing = False):
    # check for model and data existence
    # if not (os.path.exists(_base_model_file) and os.path.exists(_train_image_folder) and os.path.exists(_test_image_folder)):
    #     print("Please run 'python install_data_and_model.py' first to get the required data and model.")
    #     exit(0)

    # # get class mapping and map files from train and test image folder
    # class_mapping = create_class_mapping_from_folder(_train_image_folder)
    # train_map_file = create_map_file_from_folder(_train_image_folder, class_mapping)
    # test_map_file = create_map_file_from_folder(_test_image_folder, class_mapping, include_unknown=True)
        # get class mapping and map files from train and test image folder

   # train_map_file = "mapVNtrain.txt"
   # test_map_file = "mapVNtest.txt"
    class_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10']
   # map_test = {"bird": '0', "bond": '1', "can": '2', "cracker": '3', "house": '4', #"shoe": '5', "teapot": '6'}

    make_mode = False
    # train
    # Train only if no model exists yet or if make_mode is set to False
    if os.path.exists(tl_model_file) and make_mode:
        print("Loading existing model from %s" % tl_model_file)
        trained_model = load_model(tl_model_file)

    else:
        trained_model = train_model(base_model_file, feature_node_name, last_hidden_node_name,
                                    image_width, image_height, num_channels,
                                    len(class_mapping), train_map_file,learningRateMB=0.1, num_epochs=20, freeze=False)
        trained_model.save(_new_model_file)
        print("Stored trained model at %s" % tl_model_file)

    # if not testing:
    #     trained_model.save(_new_model_file)
    #     print("Stored trained model at %s" % tl_model_file)

    # evaluate test images
    eval_test_images(trained_model, _results_file, test_map_file,image_width,image_height)
    # with open(_results_file, 'w') as output_file:
        # with open(test_map_file, "r") as input_file:
            # for line in input_file:
                # tokens = line.rstrip().split('\t')
                # img_file = tokens[0]
                # #true_label = int(map_test[tokens[1]])
                # probs = eval_single_image(trained_model, img_file, image_width, image_height)
                # print(probs)

                # formatted_line = format_output_line(img_file, probs, class_mapping)
                # output_file.write(formatted_line)

    print("Done. Wrote output to %s" % _results_file)

if __name__ == '__main__':
    try_set_default_device(gpu(0))
    foldNumbers=5
    for fold in range(1, foldNumbers):
        results_file = os.path.join(base_folder, "Output", "predictions_VN"+str(fold)+".txt")
        new_model_file = os.path.join(base_folder, "Output", "TransferLearning_VN"+str(fold)+".model")
        train_and_eval(base_model_file, train_image_folder, test_image_folder, results_file, new_model_file,5,"mapVNtrainF"+str(fold)+".txt","mapVNtestF"+str(fold)+".txt")
		

	
