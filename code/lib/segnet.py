#####################################################################
###                           segnet.py                           ###
#####################################################################
# This script is the implementation of SegNet model by Keras

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


class segnet(object):
    def __init__(self, img_height, img_width, img_channels, metrics, learning_rate=1e-4):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.metrics = metrics
        self.model = self.init_model(metrics=metrics, learning_rate=learning_rate)

    def init_model(self, metrics, learning_rate):
        print('Initilizing SegNet model ... ', end='', flush=True)
        inputs = Input(shape=(self.img_height, self.img_width, self.img_channels))

        ### encoding layers
        conv1 = Conv2D(16, (3,3), padding='same') (inputs)
        conv1 = BatchNormalization() (conv1)
        conv1 = Activation('relu') (conv1)
        conv1 = Conv2D(16, (3,3), padding='same') (conv1)
        conv1 = BatchNormalization() (conv1)
        conv1 = Activation('relu') (conv1)
        pool1 = MaxPool2D(pool_size=(2,2)) (conv1)

        conv2 = Conv2D(32, (3,3), padding='same') (pool1)
        conv2 = BatchNormalization() (conv2)
        conv2 = Activation('relu') (conv2)
        conv2 = Conv2D(32, (3,3), padding='same') (conv2)
        conv2 = BatchNormalization() (conv2)
        conv2 = Activation('relu') (conv2)
        pool2 = MaxPool2D(pool_size=(2,2)) (conv2)

        conv3 = Conv2D(64, (3,3), padding='same') (pool2)
        conv3 = BatchNormalization() (conv3)
        conv3 = Activation('relu') (conv3)
        conv3 = Conv2D(64, (3,3), padding='same') (conv3)
        conv3 = BatchNormalization() (conv3)
        conv3 = Activation('relu') (conv3)
        conv3 = Conv2D(64, (3,3), padding='same') (conv3)
        conv3 = BatchNormalization() (conv3)
        conv3 = Activation('relu') (conv3)
        pool3 = MaxPool2D(pool_size=(2,2)) (conv3)

        conv4 = Conv2D(128, (3,3), padding='same') (pool3)
        conv4 = BatchNormalization() (conv4)
        conv4 = Activation('relu') (conv4)
        conv4 = Conv2D(128, (3,3), padding='same') (conv4)
        conv4 = BatchNormalization() (conv4)
        conv4 = Activation('relu') (conv4)
        conv4 = Conv2D(128, (3,3), padding='same') (conv4)
        conv4 = BatchNormalization() (conv4)
        conv4 = Activation('relu') (conv4)
        pool4 = MaxPool2D(pool_size=(2,2)) (conv4)

        conv5 = Conv2D(128, (3,3), padding='same') (pool4)
        conv5 = BatchNormalization() (conv5)
        conv5 = Activation('relu') (conv5)
        conv5 = Conv2D(128, (3,3), padding='same') (conv5)
        conv5 = BatchNormalization() (conv5)
        conv5 = Activation('relu') (conv5)
        conv5 = Conv2D(128, (3,3), padding='same') (conv5)
        conv5 = BatchNormalization() (conv5)
        conv5 = Activation('relu') (conv5)
        pool5 = MaxPool2D(pool_size=(2,2)) (conv5)

        ### decoding layers
        up6 = UpSampling2D((2,2)) (pool5)
        conv6 = Conv2D(128, (3,3), padding='same') (up6)
        conv6 = BatchNormalization() (conv6)
        conv6 = Activation('relu') (conv6)
        conv6 = Conv2D(128, (3,3), padding='same') (conv6)
        conv6 = BatchNormalization() (conv6)
        conv6 = Activation('relu') (conv6)
        conv6 = Conv2D(128, (3,3), padding='same') (conv6)
        conv6 = BatchNormalization() (conv6)
        conv6 = Activation('relu') (conv6)

        up7 = UpSampling2D((2,2)) (conv6)
        conv7 = Conv2D(128, (3,3), padding='same') (up7)
        conv7 = BatchNormalization() (conv7)
        conv7 = Activation('relu') (conv7)
        conv7 = Conv2D(128, (3,3), padding='same') (conv7)
        conv7 = BatchNormalization() (conv7)
        conv7 = Activation('relu') (conv7)
        conv7 = Conv2D(64, (3,3), padding='same') (conv7)
        conv7 = BatchNormalization() (conv7)
        conv7 = Activation('relu') (conv7)

        up8 = UpSampling2D((2,2)) (conv7)
        conv8 = Conv2D(64, (3,3), padding='same') (up8)
        conv8 = BatchNormalization() (conv8)
        conv8 = Activation('relu') (conv8)
        conv8 = Conv2D(64, (3,3), padding='same') (conv8)
        conv8 = BatchNormalization() (conv8)
        conv8 = Activation('relu') (conv8)
        conv8 = Conv2D(32, (3,3), padding='same') (conv8)
        conv8 = BatchNormalization() (conv8)
        conv8 = Activation('relu') (conv8)

        up9 = UpSampling2D((2,2)) (conv8)
        conv9 = Conv2D(32, (3,3), padding='same') (up9)
        conv9 = BatchNormalization() (conv9)
        conv9 = Activation('relu') (conv9)
        conv9 = Conv2D(16, (3,3), padding='same') (conv9)
        conv9 = BatchNormalization() (conv9)
        conv9 = Activation('relu') (conv9)

        up10 = UpSampling2D((2,2)) (conv9)
        conv10 = Conv2D(16, (3,3), padding='same') (up10)
        conv10 = BatchNormalization() (conv10)
        conv10 = Activation('relu') (conv10)
        conv10 = Conv2D(1, (1,1)) (conv10)
        conv10 = BatchNormalization() (conv10)
        outputs = Activation('sigmoid') (conv10)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=self.metrics)
        print('Done\nModel structure summary:', flush=True)
        print(model.summary())

        return model


    def train(self, X_train, y_train, model_name, validation_split=0.2, batch_size=32, epochs=30, verbose=1):
        print('Start training Seg-Net ... ', end='', flush=True)
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

