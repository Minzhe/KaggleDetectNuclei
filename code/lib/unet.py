##############################################################################
###                               unet.py                                  ###
##############################################################################
# U-Net keras implementation for kaggle unclei detection

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


class unet(object):

    def __init__(self, img_height, img_width, img_channels, metrics, learning_rate=1e-4):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.metrics = metrics
        self.model = self.init_model(metrics=metrics, learning_rate=learning_rate)

    def init_model(self, metrics, learning_rate):
        print('Initilizing U-Net model ... ', end='', flush=True)
        inputs = Input(shape=(self.img_height, self.img_width, self.img_channels))

        conv1 = Conv2D(16, (3,3), activation='relu', padding='same') (inputs)
        conv1 = Conv2D(16, (3,3), activation='relu', padding='same') (conv1)
        pool1 = MaxPool2D(pool_size=(2,2)) (conv1)

        conv2 = Conv2D(32, (3,3), activation='relu', padding='same') (pool1)
        conv2 = Conv2D(32, (3,3), activation='relu', padding='same') (conv2)
        pool2 = MaxPool2D(pool_size=(2,2)) (conv2)

        conv3 = Conv2D(64, (3,3), activation='relu', padding='same') (pool2)
        conv3 = Conv2D(64, (3,3), activation='relu', padding='same') (conv3)
        pool3 = MaxPool2D(pool_size=(2,2)) (conv3)

        conv4 = Conv2D(128, (3,3), activation='relu', padding='same') (pool3)
        conv4 = Conv2D(128, (3,3), activation='relu', padding='same') (conv4)
        pool4 = MaxPool2D(pool_size=(2,2)) (conv4)

        conv5 = Conv2D(256, (3,3), activation='relu', padding='same') (pool4)
        conv5 = Conv2D(256, (3,3), activation='relu', padding='same') (conv5)

        up6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same') (conv5)
        up6 = concatenate([up6, conv4])
        conv6 = Conv2D(128, (3,3), activation='relu', padding='same') (up6)
        conv6 = Conv2D(128, (3,3), activation='relu', padding='same') (conv6)

        up7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same') (conv6)
        up7 = concatenate([up7, conv3])
        conv7 = Conv2D(64, (3,3), activation='relu', padding='same') (up7)
        conv7 = Conv2D(64, (3,3), activation='relu', padding='same') (conv7)

        up8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same') (conv7)
        up8 = concatenate([up8, conv2])
        conv8 = Conv2D(32, (3,3), activation='relu', padding='same') (up8)
        conv8 = Conv2D(32, (3,3), activation='relu', padding='same') (conv8)

        up9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same') (conv8)
        up9 = concatenate([up9, conv1])
        conv9 = Conv2D(16, (3,3), activation='relu', padding='same') (up9)
        conv9 = Conv2D(16, (3,3), activation='relu', padding='same') (conv9)

        outputs = Conv2D(1, (1,1), activation='sigmoid') (conv9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=self.metrics)
        print('Done\nModel structure summary:', flush=True)
        print(model.summary())

        return model


    def train(self, X_train, y_train, model_name, validation_split=0.2, batch_size=32, epochs=30, verbose=1):
        print('Start training U-Net ... ', end='', flush=True)
        early_stopper = EarlyStopping(patience=5, verbose=1)
        check_pointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)
        result = self.model.fit(X_train, y_train, 
                                validation_split=validation_split, 
                                batch_size=batch_size, 
                                epochs=epochs, 
                                verbose=verbose,
                                shuffle=True,
                                callbacks=[early_stopper, check_pointer])
        self.model = load_model(model_name)
        print('Done')


    def load_Model(self, path):
        print('Loading model ... ', end='', flush=True)
        self.model = load_model(path)
        print('Done')
    

    def predict(self, X):
        y = self.model.predict(X, verbose=1)
        return(y)