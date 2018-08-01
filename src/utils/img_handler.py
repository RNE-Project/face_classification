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

match = re.compile('(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)')

class ImageHandler:

    def __init__(self, live = False):
        self.live = live
        aa = [iaa.Fliplr(0.5), iaa.Add((-20, 20)),
        iaa.AddToHueAndSaturation((-20, 20)), iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 0.1)), iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)),
        iaa.SaltAndPepper(p=(0, 0.05)), iaa.ContrastNormalization((0.8, 1.2))]
        self.seq = iaa.Sequential(aa)
        self.le = LabelEncoder()

    def process_path(path):
        groups = match.search(str(path)).groups()[1:]
        #print(groups)
        #speech, emotion, intensity, statement, repetition, actor =
        return groups

    def emotion_gender(self,group):
        emotion = [0, 0, 0, 0, 0, 0, 0, 0] #last two: male, female
        emotion[int(group[1])-1] = 1
        actor = int(group[5])
        #if actor % 2 == 0:
        #    emotion[9] = 1
        #else: emotion[8] = 1
        #return self.le.fit_transform(emotion)
        return emotion


    def process(self, imgs, lstm = True):
        if not self.live:
            imgs = [str(path) for path in imgs.iterdir()]
            if lstm: ImageHandler.intoten(imgs)
            imgs = [imread(p) for p in imgs]
        else:
            imgs = ia.imresize_many_images(imgs, (500, 500))

        self.seq.augment_images(imgs)
        return [rgb2gray(img)[:, :, newaxis] for img in imgs]

    def split_dataset(self, dir):
        #split by statement
        self.train_set_paths = []
        self.validation_set_paths = []
        train_set_dogs = 0
        train_set_kids = 0
        paths = list(Path(dir).iterdir())
        shuffle(paths)
        for path in paths:
            props = ImageHandler.process_path(path)
            statement = props[3]
            if statement == '01': #train_set_kids
                if train_set_kids < 613:
                    self.train_set_paths.append(path)
                    train_set_kids += 1
                else:
                    self.validation_set_paths.append(path)
            else:
                if train_set_dogs < 613:
                    self.train_set_paths.append(path)
                    train_set_dogs += 1
                else:
                    self.validation_set_paths.append(path)

    def intoten(imgs):
        rest = len(imgs) % 10
        if rest <= 5 and rest != 0:
            for i in range(rest): imgs.pop(random.randint(0,len(imgs)-1))
        elif rest > 5:
            for i in range(10-rest):
                pos = random.randint(0, len(imgs) - 1)
                imgs.insert(pos, imgs[pos])


    def flow(self, dataset_dir, mode1 = 'train', mode2='cnn'):
        self.split_dataset(dataset_dir)
        if mode2 == 'lstm':
            if mode1 == 'train':
                for path in self.train_set_paths:
                    x_val = self.process(path)
                    y_val = ImageHandler.emotion_gender(process_path(path))
                    no = len(x_val)/10
                    for i in range(1, no+1):
                        x = x_val[(i-1):i]
                        yield(x, y_val)
            if mode1 == 'valid':
                for path in self.validation_set_paths:
                    x_val = self.process(path)
                    y_val = ImageHandler.emotion_gender(process_path(path))
                    no = len(x_val)/10
                    for i in range(1, no+1):
                        x = x_val[(i-1):i]
                        yield(x, y_val)
        if mode2 == 'cnn':
            while True:
                if mode1 == 'train':
                    input = []
                    targets = []
                    for path in self.train_set_paths:
                        x_val = self.process(path, False)
                        y_val = self.emotion_gender(ImageHandler.process_path(path))
                        for x in x_val:
                            input.append(x)
                            targets.append(y_val)
                            if len(input) == 32:
                                yield (asarray(input), asarray(targets))
                                input = []
                                targets = []
                if mode1 == 'valid':
                    input = []
                    targets = []
                    for path in self.validation_set_paths:
                        x_val = self.process(path, False)
                        y_val = self.emotion_gender(ImageHandler.process_path(path))
                        for x in x_val:
                            input.append(x)
                            targets.append(y_val)
                            if len(input) == 32:
                                yield (asarray(input), asarray(targets))
                                input = []
                                targets = []
                shuffle(self.validation_set_paths)
                shuffle(self.train_set_paths)


#p = Path('E:/dataset')
#imghandler = ImageHandler()
#statement = [0, 0]
#for path in p.iterdir():
#    groups = ImageHandler.process_path(path)
#    if groups[3] == '01':
#        statement[0] += 1
#    else: statement[1] += 1

#print(statement)

#imghandler.split_dataset(p)
#print(len(imghandler.train_set_paths))


# pass list of dir paths to flow, one dir = one emotion and gender, one batch
#hard coded batch size for now
