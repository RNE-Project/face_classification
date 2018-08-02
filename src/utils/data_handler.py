from imageio import imread, imwrite
from sklearn.preprocessing import LabelEncoder
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from pathlib import Path
import re
from random import shuffle
import random
from numpy import newaxis, asarray
from skimage.color import rgb2gray
import constants

match = re.compile('(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)')
def process_path(path):
    groups = match.search(str(path)).groups()[1:]
    #print(groups)
    #speech, emotion, intensity, statement, repetition, actor =
    return groups

def emotion(group):
    emotion = [0, 0, 0, 0, 0, 0, 0, 0]
    emotion[int(group[1])-1] = 1
    return emotion

def gender(group):
    gender = [0, 0] #male female
    actor = int(group[5])
    if actor % 2 == 0:
        gender[1] = 1
    else:
        gender[0] = 1

    return gender

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def normalize(imgs):
    window = constants.lstm_window
    rest = len(imgs) % window
    if rest <= window/2 and rest != 0:
        for i in range(rest): imgs.pop(random.randint(0,len(imgs)-1))
    elif rest > window/2:
        for i in range(window/2-rest):
            pos = random.randint(0, len(imgs) - 1)
            imgs.insert(pos, imgs[pos])
    imgs = list(chunks(imgs, window))



seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Add((-20, 20)),
iaa.AddToHueAndSaturation((-20, 20)), iaa.Multiply((0.8, 1.2)),
iaa.GaussianBlur(sigma=(0, 0.1)), iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)),
iaa.SaltAndPepper(p=(0, 0.05)), iaa.ContrastNormalization((0.8, 1.2))])

def process(imgs):
    imgs = [str(path) for path in imgs]
    imgs = [imread(p) for p in imgs]

    seq.augment_images(imgs)
    return [rgb2gray(img)[:, :, newaxis] for img in imgs]

class TrainCNN:
    def __init__(self, target = 'emotion'):
        self.split_dataset()
        self.steps_per_epoch_train = int(len(self.train_set_paths)/Constants.batch_size)
        self.steps_per_epoch_valid = int(len(self.validation_set_paths)/Constants.batch_size)
        if target == 'emotion':
            self.y_func = emotion
        if target == 'gender':
            self.y_func = gender

    def split_dataset(self):
        self.train_set_paths = []
        self.validation_set_paths = []
        paths = []
        #rewrite this maybe
        for path in Path(constants.dataset_path).iterdir():
            for file in path.iterdir():
                paths.append(file)
        shuffle(paths)
        for path in paths:
            which = random.random()
            if which < 0.7:
                self.train_set_paths.append(path)
            else:
                self.validation_set_paths.append(path)

    def flow(self, mode = 'train'):
        flow_paths = []
        if mode == 'train':
            flow_paths = self.train_set_paths
        if mode == 'valid':
            flow_paths = self.validation_set_paths

        input = []
        targets = []
        while True:
            for path in flow_paths:

                y = self.y_func(process_path(path))
                input.append(path)
                targets.append(y)
                if len(input) == constants.cnn_batch_size:
                    input = process(input)
                    yield(asarray(input), asarray(targets))
                    input = []
                    targets = []
            shuffle(flow_paths)

class TrainLSTM:
    def __init__(self, target='emotion'):
        self.split_dataset()
        self.steps_per_epoch_train = int(len(self.train_set_paths)/Constants.batch_size)
        self.steps_per_epoch_valid = int(len(self.validation_set_paths)/Constants.batch_size)
        if target == 'emotion':
            self.y_func = emotion
        if target == 'gender':
            self.y_func = gender

    def split_dataset(self):
        self.train_set_paths = []
        self.validation_set_paths = []
        paths = []
        #rewrite this
        for path in Path(constants.dataset_path).iterdir():
            #have to check if this is Sequential
            files = list(sorted(path.iterdir()))
            normalize(files)
            which = random.random()
            if which < 0.7:
                # so now it is a list of lists of size lstm_window
                self.train_set_paths.append(files)
            else:
                self.validation_set_paths.append(files)
        shuffle(self.train_set_paths)
        shuffle(self.validation_set_paths)

    def flow(self, mode = 'train'):
        flow_paths = []
        if mode == 'train':
            flow_paths = self.train_set_paths
        if mode == 'valid':
            flow_paths = self.validation_set_paths

        input = []
        targets = []
        while True:
            for lstm_chunk in flow_paths:
                # lstm_chunk is a chunk of size lstm_window that contains some paths
                y = self.y_func(process_path(lstm_chunk[0]))
                x = process(lstm_chunk)
                input.append(x)
                targets.append(y)
                if len(input) == constants.lstm_batch_size:
                    yield(asarray(input), asarray(targets))
                    input = []
                    targets = []
            shuffle(flow_paths)
