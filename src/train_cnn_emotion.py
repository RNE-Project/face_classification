from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
#from models.cnn import mini_XCEPTION
from keras.models import load_model
from utils.data_handler import TrainCNN, TestCNN
from utils.constants import Constants

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# parameters
#batch_size = 64
#num_epochs = 100
#validation_split = .2
#do_random_crop = False
patience = 50
num_classes = 7
#dataset_name = 'imdb'
input_shape = (64, 64, 1)
#if input_shape[2] == 1:
    #grayscale = True
#images_path = '../datasets/imdb_crop/'
log_file_path = Constants.log_dir + 'cnnemotion.txt'
trained_models_path = Constants.emotion_model_dir + 'cnn'

# model parameters/compilation
#model = mini_XCEPTION(input_shape, num_classes)
model = load_model('../fer2013_mini_XCEPTION.102-0.66.hdf5')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# model callbacks
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

img_hndl = TrainCNN('emotion')

callbacks.append(TestCNN(img_hndl))

model.fit_generator(img_hndl.flow(), steps_per_epoch=img_hndl.steps_per_epoch_train, epochs = Constants.epochs, verbose =1, callbacks = callbacks, validation_data = img_hndl.flow('valid'), validation_steps=img_hndl.steps_per_epoch_valid)
