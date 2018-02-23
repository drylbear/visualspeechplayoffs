import glob
import os
import os.path
import re
from subprocess import call
import collections
import cv2
import numpy as np
from joblib import Parallel, delayed
from process import process_image, calc_dct, calc_aam, calc_sift, calc_hog, calc_cnn, FaceDetectionError
from keras.applications import ResNet50 as ResNet
from time import sleep

detection_problems = {
    "10M-si833": (0, 92),
    "30F-sx360": (0, 105),
    "56M-sx67": (0, 135),
    "40F-si1557": (10, -1),
    "05F-si1132": (10, -1),
    "05F-si1231": (26, -1),
    "05F-si1581": (15, -1),
    "05F-si1678": (0, 62),
    "05F-si1766": (25, -1),
    "05F-si970": (0, 156),
    "05F-sx231": (0, 74),
    "05F-sx241": (0, 110),
    "05F-sx327": (20, -1),
    "28M-si1417": (5, -1),
    "25M-si1113": (0, 180),
}


def remove_resnet_wrong():
    print("Calculate features for all images in data/train")

    path = os.path.dirname(os.path.realpath(__file__))
    frame_by_frame = get_frame_phonemes()
    group = 'train'
    speakers_path = sorted(glob.glob(os.path.join(path, "resnet", '*')))
    for speaker_path in speakers_path:
        speaker = os.path.basename(speaker_path)
        print("--> Speaker {}".format(speaker))
        jpgs_all = sorted(glob.glob(speaker_path + '/*'))
        files = list(set([jpg for jpg in jpgs_all if not "-" in jpg]))
        for i, file_path in enumerate(files):
            print("----> Features for File {}, {}/{}".format(file_path, i + 1, len(files)))
            # sleep(5)
            os.remove(file_path)


def calc_features_on_pictures(model=ResNet(include_top=False, pooling="avg")):
    print("Calculate features for all images in data/train")

    path = os.path.dirname(os.path.realpath(__file__))
    frame_by_frame = get_frame_phonemes()
    group = 'train'
    speakers_path = sorted(glob.glob(os.path.join(path, "resnet", '*')))
    for speaker_path in speakers_path:
        speaker = os.path.basename(speaker_path)
        print("--> Speaker {}".format(speaker))
        jpgs_all = sorted(glob.glob(speaker_path + '/*'))
        files = list(set([jpg.split('-')[0] for jpg in jpgs_all]))
        for i, file_path in enumerate(files):
            file = os.path.basename(file_path)
            filename_ext = file + '.mp4'
            print("----> Features for File {}, {}/{}".format(file, i + 1, len(files)))
            video_parts = group, speaker, file, filename_ext

            calc_features(video_parts, model, len(list(frame_by_frame[speaker][file].values())))


def extract_files(model=ResNet(include_top=False, pooling="avg")):
    data_file = []
    missing = []
    path = os.path.dirname(os.path.realpath(__file__))

    frame_by_frame = get_frame_phonemes()  # [speaker][file_no_ext][frame]

    files = get_train_test_lists()
    group = 'train'
    for i, video in enumerate(sorted(files)[98*30:]):
        # Get the parts.
        parts = video.split(os.path.sep)
        speakername = parts[-2]
        filename = parts[-1]
        filename_no_ext = filename.split('.')[0]
        src = os.path.join('original', speakername, filename)
        video_parts = group, speakername, filename_no_ext, filename

        num_frames_vid = get_nb_frames_for_video(video_parts)
        phonemes = list(frame_by_frame[speakername][filename_no_ext].values())
        num_frames_file = len(phonemes)

        print("Current: {}/{} Progress: {}/{}".format(speakername, filename_no_ext, i, len(files)))

        dest_base = os.path.join(path, group, speakername, filename_no_ext)
        dest_resnet = os.path.join(path, "resnet", speakername, filename_no_ext)

        if not check_already_extracted(video_parts, frame_by_frame):
            print("--> Generate jpgs from video")
            if not os.path.exists(os.path.dirname(dest_base)):
                os.makedirs(os.path.dirname(dest_base))
            dest = dest_base + '-%04d.jpg'
            call(['ffmpeg', '-loglevel', 'quiet', '-i', src, dest])

            jpgs = sorted(glob.glob(dest_base + '-*.jpg'))
            if speakername + "-" + filename_no_ext in detection_problems:
                start, end = detection_problems[speakername + "-" + filename_no_ext]

                if speakername == "10M" and filename_no_ext == "si833":
                    jpgs[91] = jpgs[90]
                if speakername == "56M" and filename_no_ext == "sx67":
                    jpgs[129] = jpgs[128]
                    jpgs[130] = jpgs[129]
                    jpgs[131] = jpgs[129]
                    jpgs[132] = jpgs[129]
                    jpgs[133] = jpgs[129]
                    jpgs[134] = jpgs[129]
                for j in range(start):
                    os.remove(dest_base + '-%04d.jpg' % (j + 1))
                if not end == -1:
                    for j in range(end, num_frames_file):
                        os.remove(dest_base + '-%04d.jpg' % (j + 1))

                jpgs = jpgs[start:end]

            print("--> Crop images and process them")
            try:
                # Order is important!! As the second command overwrites the jpg file and it's not usable anymore
                Parallel(n_jobs=-1)(delayed(parallel_process_image)(jpg, (224, 224, 3), dest_resnet) for jpg in jpgs)
                Parallel(n_jobs=-1)(delayed(parallel_process_image)(jpg) for jpg in jpgs)
            except FaceDetectionError as e:
                print(str(e))
                missing.append([group, speakername, filename_no_ext])
                continue

        data_file.append([group, speakername, filename_no_ext, num_frames_file, phonemes])

        if num_frames_file < num_frames_vid:
            for j in range(num_frames_file, num_frames_vid):
                rm_path = dest_base + '-%04d.jpg' % (j + 1)
                rm_resnet = dest_resnet + '-%04d.jpg' % (j + 1)
                call(['rm', '-rf', rm_path])
                call(['rm', '-rf', rm_resnet])

        if not check_already_extracted_feature(video_parts):
            print("--> Calculate features")
            if not os.path.exists(os.path.join('data', 'sequence')):
                os.makedirs(os.path.join('data', 'sequence'))
            calc_features(video_parts, model, num_frames_file)

        print("--> Generated %d frames for %s" % (num_frames_file, filename_no_ext))

    print("Extracted and wrote %d video files." % (len(data_file)))
    print("Still %d files left to calculate" % (len(missing)))


def parallel_process_image(jpg, target_shape=(48, 48, 1), save_to=None):
    if save_to is None:  # save it to the same place
        save_to = jpg
    else:  # keep the number to the jpg the same!!
        save_to += "-" + jpg.split("-")[1]
    img = process_image(jpg, target_shape, 'mouth')
    cv2.imwrite(save_to, img)


def calc_features(video_parts, model, true_length):
    train_or_test, speakername, filename_no_ext, filename = video_parts
    base_path = os.path.join('data', train_or_test, speakername, filename_no_ext)
    resnet_path = os.path.join("data", "resnet", speakername, filename_no_ext)
    sequence_path = os.path.join("data", "sequence", speakername + "-" + filename_no_ext)

    jpgs = sorted(glob.glob(base_path + '-*.jpg'))
    jpgs_resnet = sorted(glob.glob(resnet_path + "-*.jpg"))

    if len(glob.glob(sequence_path + "-*.npy")) >= 5:
        return

    # calculate features in parallel
    resnet_v = np.asarray([calc_cnn(jpg, model) for jpg in jpgs_resnet])
    dct_v = np.asarray(Parallel(n_jobs=-1)(delayed(calc_dct)(jpg) for jpg in jpgs))
    sift_v = np.asarray(Parallel(n_jobs=-1)(delayed(calc_sift)(jpg) for jpg in jpgs))
    hog_v = np.asarray(Parallel(n_jobs=-1)(delayed(calc_hog)(jpg) for jpg in jpgs))
    aam_v, aam_landmarks = calc_aam(jpgs, speakername)

    # Check that the length denoted in the ground truth file is equal to the number of frames
    nb_frames = get_nb_frames_for_video(video_parts)
    print(resnet_v.shape)
    # assert correct length and shape of all the features calculated
    assert dct_v.shape == (nb_frames, 10)
    assert sift_v.shape == (nb_frames, 20)
    assert hog_v.shape == (nb_frames, 324)
    assert aam_v.shape == (nb_frames, 42)
    assert resnet_v.shape == (nb_frames, 2048)

    #  write it to file
    for feature in ['dct', 'hog', 'sift', 'aam', 'landmarks', "cnn"]:
        feature_path = os.path.join('data', 'sequence', speakername + '-' + filename_no_ext + '-' + feature)
        if feature == 'dct':
            np.save(feature_path, dct_v)
        elif feature == 'hog':
            np.save(feature_path, hog_v)
        elif feature == 'sift':
            np.save(feature_path, sift_v)
        elif feature == 'aam':
            np.save(feature_path, aam_v)
        elif feature == 'landmarks':
            np.save(feature_path, aam_landmarks)
        elif feature == "cnn":
            np.save(feature_path, resnet_v)


def get_frame_phonemes():
    d = collections.defaultdict(dict)
    with open('data/frame_by_frame.txt', 'r') as fbf:
        for line in fbf:
            l = line.strip().split(' ')
            if not l[0] == 'speaker':
                if not l[0] in d.keys():
                    d[l[0]] = {}
                if not l[1] in d[l[0]].keys():
                    d[l[0]][l[1]] = {}
                d[l[0]][l[1]][int(l[2])] = l[3]
    return dict(d)


def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join('data', train_or_test, classname, filename_no_ext + '-*.jpg'))
    return len(generated_files)


def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    tmp = video_path.replace('data/', "")
    parts = tmp.split(os.path.sep)
    filename = parts[2]
    filename_no_ext = filename.split('.')[0]
    classname = parts[1]
    train_or_test = parts[0]

    return train_or_test, classname, filename_no_ext, filename


def check_already_extracted(video_parts, frame_by_frame):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, speakername, filename_no_ext, _ = video_parts

    jpgs = glob.glob(os.path.join('data', train_or_test, speakername, filename_no_ext + '-*.jpg'))
    jpgs_resnet = glob.glob(os.path.join('data', "resnet", speakername, filename_no_ext + '-*.jpg'))
    len_gt = len(list(frame_by_frame[speakername][filename_no_ext].values()))
    return len(jpgs_resnet) > 0 and len(jpgs) > 0  # len(jpgs_resnet) == len_gt and len(jpgs) > 0 and len(jpgs) == len_gt


def check_already_extracted_feature(video_parts):
    """Check to see if we created the feature of this file."""
    _, speakername, filename_no_ext, _ = video_parts
    features = glob.glob(os.path.join('data', 'sequence', speakername + '-' + filename_no_ext + '-*.npy'))
    return len(features) > 0


def get_train_test_lists():
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    train_list = []
    p = re.compile('[0-9][0-9][MF]')
    d = os.path.join('original')
    for speaker in sorted([f for f in os.listdir(d) if p.match(f)]):
        vids = glob.glob(os.path.join(d, speaker, '*'))
        for vid in vids:
            train_list.append(vid)

    return train_list


def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:
    [train|test], class, filename, nb frames
    """
    calc_features_on_pictures()


if __name__ == '__main__':
    main()
