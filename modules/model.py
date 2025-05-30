from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D, Flatten
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow import random_normal_initializer, tile

def conv_block(input_tensor, filters, size, strides=1, batchnorm=True, name="null", sequential = False, activation="relu"):
    'Base convolutional block'
    initializer = random_normal_initializer(0., 0.02)
    if activation == "leakyrelu":
        activation = LeakyReLU()
    if sequential:
        result = Sequential(name=name)
        result.add(Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False, name=name+"_Conv2D"))
        if batchnorm:
            result.add(BatchNormalization(name=name+"_BatchNorm"))
        result.add(Activation(activation, name=name+"_Activation"))
        return result
    else:
        x = Conv2D(filters, size, strides=strides, padding='same',
                   kernel_initializer=initializer, name=name)(input_tensor)
        if batchnorm:
            x = BatchNormalization(name=name+"_BatchNorm")(x)
        x = Activation(activation, name=name+"_Activation")(x)
        return x

def Generator(n_filters=64, dropout=0.5):
    # Inputs
    input_img = Input((256,256,1,2), name='G_InputSequences')
    input_quality = Input((1,), name="G_InputQuality")
    input_FS = Input((1,), name="G_InputFS")
        
    # contracting path
    input_reshaped = Reshape((256,256,2), name="G_InputSequencesShaped")(input_img)
    input_quality_reshaped = tile(Reshape((1,1,1))(input_quality), (1,256,256,1), name="G_InputQualityTiled")
    input_FS_reshaped = tile(Reshape((1,1,1))(input_FS), (1,256,256,1), name="G_InputFSTiled")
    input_concatenated = concatenate([input_reshaped, input_quality_reshaped, input_FS_reshaped], axis=3, name="G_InputConcatenated")
    
    c0 = conv_block(input_concatenated, n_filters*1, 3, name="G_Down_1a")
    c1 = conv_block(c0, n_filters*1, 3, 2, name="G_Down_1b")
    p1 = Dropout(dropout*0.5, name="G_Down_Drop1")(c1)

    c2 = conv_block(p1, n_filters*2, 3, name="G_Down_2a")
    c2 = conv_block(c2, n_filters*2, 3, 2, name="G_Down_2b")
    p2 = Dropout(dropout, name="G_Down_Drop2")(c2)

    c3 = conv_block(p2, n_filters*4, 3, name="G_Down_3a")
    c3 = conv_block(c3, n_filters*4, 3, 2, name="G_Down_3b")
    p3 = Dropout(dropout, name="G_Down_Drop3")(c3)

    c4 = conv_block(p3, n_filters*8, 3, name="G_Down_4a")
    c4 = conv_block(c4, n_filters*8, 3, 2, name="G_Down_4b")
    p4 = Dropout(dropout, name="G_Down_Drop4")(c4)
    
    c5 = conv_block(p4, n_filters*16, 3, name="G_Down_5a")
    c5 = conv_block(c5, n_filters*16, 3, 2, name="G_Down_5b")
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, 3, 2, padding='same', name="G_Up_1a")(c5)
    u6 = concatenate([u6, c4], name="G_Up_Skip1")
    u6 = Dropout(dropout, name="G_Up_Drop1")(u6)
    u6 = conv_block(u6, n_filters*8, 3, name="G_Up_1b")
    u6 = conv_block(u6, n_filters*8, 3, name="G_Up_1c")

    u7 = Conv2DTranspose(n_filters*4, 3, 2, padding='same', name="G_Up_2a")(u6)
    u7 = concatenate([u7, c3], name="G_Up_Skip2")
    u7 = Dropout(dropout, name="G_Up_Drop2")(u7)
    u7 = conv_block(u7, n_filters*4, 3, name="G_Up_2b")
    u7 = conv_block(u7, n_filters*4, 3, name="G_Up_2c")

    u8 = Conv2DTranspose(n_filters*2, 3, 2, padding='same', name="G_Up_3a")(u7)
    u8 = concatenate([u8, c2], name="G_Up_Skip3")
    u8 = Dropout(dropout, name="G_Up_Drop3")(u8)
    u8 = conv_block(u8, n_filters*2, 3, name="G_Up_3b")
    u8 = conv_block(u8, n_filters*2, 3, name="G_Up_3c")

    u9 = Conv2DTranspose(n_filters*1, 3, 2, padding='same', name="G_Up_4a")(u8)
    u9 = concatenate([u9, c1], axis=3, name="G_Up_Skip4")
    u9 = Dropout(dropout, name="G_Up_Drop4")(u9)
    u9 = conv_block(u9, n_filters*1, 3, name="G_Up_4b")
    u9 = conv_block(u9, n_filters*1, 3, name="G_Up_4c")

    u10 = Conv2DTranspose(n_filters*1, 3, 2, padding='same', name="G_Up_5a") (u9)
    u10 = concatenate([u10, c0], axis=3, name="G_Up_Skip5")
    u10 = Dropout(dropout, name="G_Up_Drop5")(u10)
    u10 = conv_block(u10, n_filters*1, 3, name="G_Up_5b")
    u10 = conv_block(u10, n_filters*1, 3, name="G_Up_5c")
    
    output = Conv2D(1, 1, activation='tanh', dtype='float32', name="G_End") (u10)
    model = Model(inputs=[input_img, input_quality, input_FS], outputs=[output], name="synthFlairGenerator")
    return model

def Discriminator(n_filters=128):
    initializer = random_normal_initializer(0., 0.02)
    inp = Input(shape=[256, 256, 1, 2], name='D_InputSequences') 
    inp_reshaped = Reshape((256,256,2), name="D_InputSequencesShaped")(inp)
    sobel = Input(shape=[256, 256, 1], name='D_SobelImage')#MAD 
    tar = Input(shape=[256, 256, 1], name='D_TargetImage')
    qual = Input(shape=[1,], name='D_Quality')
    qual_tile = tile(Reshape((1,1,1))(qual), (1,256,256,1), name="D_Quality_Tiled")
    fs = Input(shape=[1,], name='D_FS')
    fs_tile = tile(Reshape((1,1,1))(qual), (1,256,256,1), name="D_FS_Tiled")

    x = concatenate([inp_reshaped, sobel, tar, qual_tile, fs_tile], name="D_InputVector") # (bs, 256, 256, channels*2)
    down1 = conv_block(x, n_filters, 4, 2, batchnorm=False, name="D_Down1", sequential=True, activation="leakyrelu")(x) # (bs, 128, 128, 64)
    down2 = conv_block(down1, n_filters*2, 4, 2, name="D_Down2", sequential=True, activation="leakyrelu")(down1) # (bs, 64, 64, 128)
    down3 = conv_block(down2, n_filters*4, 4, 2, name="D_Down3", sequential=True, activation="leakyrelu")(down2) # (bs, 32, 32, 256)
    down4 = conv_block(down3, n_filters*8, 4, 2, name="D_Down4", sequential=True, activation="leakyrelu")(down3) # (bs, 16, 16, 512)

    last = Conv2D(1, 2, strides=2, activation="sigmoid",
                 kernel_initializer=initializer, use_bias=True, name="D_End")(down4) # (bs, 4, 4, 1)
    
    return Model(inputs=[inp, sobel, qual, tar], outputs=last)#MAD

