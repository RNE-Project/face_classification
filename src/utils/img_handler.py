from imageio import imread
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from pathlib import Path
import re
from random import shuffle

match = re.compile('(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)')

class ImageHandler:

    def __init__(self, live = False):
        self.live = live
        aa = [iaa.Fliplr(0.5), iaa.Add((-20, 20)),
        iaa.AddToHueAndSaturation((-20, 20)), iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 0.1)), iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)),
        iaa.SaltAndPepper(p=(0, 0.05)), iaa.ContrastNormalization((0.8, 1.2)), iaa.Grayscale(alpha=1.0)]
        if live: aa.append(iaa.Grayscale(alpha=1.0))
        self.seq = iaa.Sequential(aa)

    def process_path(path):
        groups = match.search(str(path)).groups()[1:]
        #print(groups)
        #speech, emotion, intensity, statement, repetition, actor =
        return groups

    def emotion_gender(group):
        emotion = int(group[1])
        actor = int(group[5])
        female = False
        if actor % 2 == 0:
            female = True
        return (emotion, female)


    def process(self, imgs):
        if not self.live:
            imgs = ImageHandler.intoten(list(imgs.iterdir()))
            imgs = list(imread(str(path) for path in imgs))
        else:
            imgs = ia.imresize_many_images(imgs, (450, 450))

        return self.seq.augment_images(imgs)

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


    def flow(self, dataset_dir, mode = 'train'):
        self.split_dataset(dataset_dir)
        if mode == 'train':
            for path in self.train_set_paths:
                x_val = self.process(path)
                y_val = ImageHandler.emotion_gender(process_path(path))
                no = len(x_val)/10
                for i in range(1, no+1):
                    x = x_val[(i-1):i]
                    yield(x, y_val)
        if mode == 'valid':
            for path in self.validation_set_paths:
                x_val = self.process(path)
                y_val = ImageHandler.emotion_gender(process_path(path))
                no = len(x_val)/10
                for i in range(1, no+1):
                    x = x_val[(i-1):i]
                    yield(x, y_val)

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
