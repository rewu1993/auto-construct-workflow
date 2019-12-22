import pandas as pd
import sys
import numpy as np
from keras import applications
from keras import layers
from keras.models import Sequential, Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray


ROWS, COLS, CHANNELS = 180, 320,3
NUM_CLASSES = 8
ROOT_PATH = '/home/rewu/Documents/research/auto-construct-workflow/'

class Classifer(object):
    def __init__(self,input_file_name):
        self.df = self._read_file(input_file_name)
        self.model = self._get_model((ROWS, COLS, CHANNELS),NUM_CLASSES)
        
        
    def _read_file(self,input_file_name):
        df = pd.read_csv(input_file_name)
        df['path'] = ROOT_PATH+df['path'] 
        df['label'] = df['label'].astype('str')
        return df
    
    
    def _get_model(self,input_shape, nb_classes):
        base_model = applications.Xception(input_shape=input_shape, weights='imagenet', include_top=False)

        # Top Model Block
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(nb_classes, activation='sigmoid')(x)

        # add your top layer block to your base model
        model = Model(base_model.input, predictions)
        return model
    
    def _rgb_gray(self,image):
        img = rgb2gray(image)
        return np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
        
        
    
class Trainer(Classifer):
    def __init__(self,input_file_name):
        super().__init__(input_file_name)
        self.train_data_gen = self._get_data_generator()
        self.val_data_gen = self._get_data_generator(True)
    
    
    def compile_train_model(self,lr = 1e-3):
        adam = optimizers.Adam(lr)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self,save_path,epochs=10):
        self.model.fit_generator(
                        self.train_data_gen,
                        steps_per_epoch=500,
                        epochs=epochs,
                        validation_data=self.val_data_gen,
                        validation_steps=20,
                        workers = 10,
                        use_multiprocessing = False)
        self.model.save(save_path)
    
    def _get_data_generator(self,val = False):
        train_df, val_df = self._get_train_val_df()
        
        datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            preprocessing_function = self._rgb_gray,
            horizontal_flip=True)
        
        df = train_df
        if val:
            df = val_df
            
        return datagen.flow_from_dataframe(
                            dataframe=df,
                            directory=None,
                            color_mode = "rgb",
                            x_col="path",
                            y_col="label",
                            target_size=(ROWS, COLS),
                            batch_size=32,
                            class_mode= 'categorical')
    
    def _get_train_val_df(self,train_ratio = 0.7):
        df = self.df.sample(frac=1).reset_index(drop=True)
        
        l = len(df)
        train_length = int (train_ratio * l)
    
        train_df = df.iloc[:train_length]
        val_df = df.iloc[train_length:]
    
        return train_df,val_df
    
class Predictor(Classifer):
    def __init__(self,input_file_name):
        super().__init__(input_file_name)
        self.test_data_gen = self._get_test_data_generator()

    def _get_test_data_generator(self):
        datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function = self._rgb_gray)
            
        return datagen.flow_from_dataframe(
                            dataframe=self.df,
                            directory=None,
                            color_mode = "rgb",
                            x_col="path",
                            y_col="label",
                            target_size=(ROWS, COLS),
                            batch_size=1,
                            shuffle=False,
                            class_mode= 'categorical')
    
    def load_weight(self,weight_path):
        self.model.load_weights(weight_path)
    
    def pred(self):
        n_files = len(self.test_data_gen.filenames)
        res = self.model.predict_generator(self.test_data_gen,steps = n_files,
                                          workers=10, use_multiprocessing=False, verbose=1)
        return res
        
