# -*- coding: utf-8 -*-
import tensorflow as tf
import itertools
import numpy as np
import random
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import itertools
import math 
from skimage.transform import AffineTransform
from scipy.ndimage import affine_transform
from random import randrange
import tensorflow.keras.backend as K
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

# Utils for augmentation
def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image
def random_crop(input_image, real_image, orig_height, orig_width):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image =  tf.image.random_crop(stacked_image, size=[2, orig_height, orig_width, 1])
    return cropped_image[0], cropped_image[1]


# CPU transform
def transform_output(item, trIMGX, trMASKX, trIMGY, trMIRROR, 
                     trINPUTSCALE, scaleINPUTVALUE, scaleINPUTCLIP, 
                     trOUTPUTSCALE, scaleOUTPUTVALUE, scaleOUTPUTCLIP, 
                     trMNI, mniVALUE, shuffle, shapeaugm, flipaugm, augment, actbrightaugm, 
                     augment_brightness, augment_contrast,
                     augment_scalefactor, augment_translatepixel, 
                     augment_rotateangle, augment_shearangle,
                     flatten, batch_size):
    # Augmentation via flipping
    if flipaugm:
        flip = slice(None,None,1) if random.randrange(2) else slice(None,None,-1)
    else:
        flip = slice(None,None,1)
    
    # Signal normalization
    if trINPUTSCALE:
        scaleINPUTVALUE = np.array(scaleINPUTVALUE)
        scalerange = np.tile(np.reshape(scaleINPUTVALUE[:,1] - scaleINPUTVALUE[:,0], (1,1,1,1,scaleINPUTVALUE.shape[0])), 
                             (item[0]["img"].shape[0],item[0]["img"].shape[1],item[0]["img"].shape[2],item[0]["img"].shape[3],1))
        scalebase = np.tile(np.reshape(scaleINPUTVALUE[:,0], (1,1,1,1,scaleINPUTVALUE.shape[0])), 
                            (item[0]["img"].shape[0],item[0]["img"].shape[1],item[0]["img"].shape[2],item[0]["img"].shape[3],1))
        item[0]["img"] = ((item[0]["img"][...] - scalebase) / scalerange)
        
    
    if trOUTPUTSCALE:
        scaleOUTPUTVALUE = np.array(scaleOUTPUTVALUE)
        scalerange = np.tile(np.reshape(scaleOUTPUTVALUE[:,1] - scaleOUTPUTVALUE[:,0], (1,1,1,scaleOUTPUTVALUE.shape[0])), 
                             (item[1].shape[0],item[1].shape[1],item[1].shape[2],1))
        scalebase = np.tile(np.reshape(scaleOUTPUTVALUE[:,0], (1,1,1,scaleOUTPUTVALUE.shape[0])), 
                            (item[1].shape[0],item[1].shape[1],item[1].shape[2],1))
        item[1] = ((item[1][...] - scalebase) / scalerange)
        
    peripheralX = np.percentile(np.concatenate([item[0]["img"][:,0,:,:,:], item[0]["img"][:,-1,:,:,:], item[0]["img"][:,:,0,:,:], item[0]["img"][:,:,-1,:,:]], axis=1), 
                               50, axis=1, interpolation="midpoint")
    peripheralbaseX = np.tile(np.reshape(peripheralX, (item[0]["img"].shape[0], 1, 1, item[0]["img"].shape[3], item[0]["img"].shape[4])), 
                             (1,item[0]["img"].shape[1],item[0]["img"].shape[2],1,1))
    item[0]["img"] = item[0]["img"] - peripheralbaseX
    
    peripheralY = np.percentile(np.concatenate([item[1][:,0,:,:], item[1][:,-1,:,:], item[1][:,:,0,:], item[1][:,:,-1,:]], axis=1), 
                               50, axis=1, interpolation="midpoint")
    peripheralbaseY = np.tile(np.reshape(peripheralY, (item[1].shape[0], 1, 1, item[1].shape[3])), 
                             (1,item[1].shape[1],item[1].shape[2],1))
    item[1] = item[1] - peripheralbaseY
    
    # augmentation by shape transformation
    if shapeaugm:
        rotateangle = random.uniform(-augment_rotateangle,+augment_rotateangle)
        shearangle = random.uniform(-augment_shearangle,+augment_shearangle)
        scalex = random.uniform(1-augment_scalefactor,1+augment_scalefactor)
        scaley = random.uniform(1-augment_scalefactor,1+augment_scalefactor)
        tx = random.uniform(-augment_translatepixel,+augment_translatepixel)
        ty = random.uniform(-augment_translatepixel,+augment_translatepixel)
        matrix = AffineTransform(scale=(scalex, scaley), rotation=rotateangle, shear=shearangle, translation=(tx,ty)).params
        for sl in range(batch_size):
            if trIMGX:
                for i in range(item[0]["img"].shape[-2]):
                    for j in range(item[0]["img"].shape[-1]):
                        item[0]["img"][sl,...,i,j] = affine_transform(item[0]["img"][sl,flip,...,i,j], matrix, mode="nearest")
            if trMASKX:
                for i in range(item[0]["mask"].shape[-1]):
                    item[0]["mask"][sl,...,i] = affine_transform(item[0]["mask"][sl,flip,...,i], matrix, mode="nearest", order=0)
            for i in range(item[1].shape[-1]):
                item[1][sl,...,i] = affine_transform(item[1][sl,flip,...,i], matrix, mode="nearest", order=0)
    elif flipaugm:
        if trIMGX: item[0]["img"] = item[0]["img"][:,flip]
        if trMASKX: item[0]["mask"] = item[0]["mask"][:,flip]
        item[1] = item[1][:,flip]

    # augmentation by brightness/contrast
    if trIMGX:
        for i in range(item[0]["img"].shape[-1]):
            if(actbrightaugm[i]):
                brightplus = random.uniform(-augment_brightness,+augment_brightness)
                brightmult = random.uniform(1-augment_contrast,1+augment_contrast)
                item[0]["img"][...,i] = brightmult*(brightplus+item[0]["img"][...,i])
                
    item[0]["img"] = item[0]["img"] + peripheralbaseX
    
    # Signal clipping
    if trINPUTSCALE:
        item[0]["img"] = item[0]["img"]*2-1
        scaleINPUTCLIP = np.array(scaleINPUTCLIP)
        if np.sum(scaleINPUTCLIP) > 0:
            scaleINPUTCLIP = np.tile(scaleINPUTCLIP, (item[0]["img"].shape[0], item[0]["img"].shape[1], item[0]["img"].shape[2], item[0]["img"].shape[3], 1))
            item[0]["img"][np.logical_and(item[0]["img"] > 1,scaleINPUTCLIP)] = 1
            item[0]["img"][np.logical_and(item[0]["img"] < -1,scaleINPUTCLIP)] = -1
    
    item[1] = item[1] + peripheralbaseY
    
    if trOUTPUTSCALE:
        item[1] = item[1]*2-1
        scaleOUTPUTCLIP = np.array(scaleOUTPUTCLIP)
        if np.sum(scaleOUTPUTCLIP) > 0:
            scaleOUTPUTCLIP = np.tile(scaleOUTPUTCLIP, (item[1].shape[0], item[1].shape[1], item[1].shape[2], 1))
            item[1][np.logical_and(item[1] > 1,scaleOUTPUTCLIP)] = 1
            item[1][np.logical_and(item[1] < -1,scaleOUTPUTCLIP)] = -1
        
    # flattening
    if flatten == True:
        item[1] = item[1].reshape((item[1].shape[0], -1))
    return item

# GPU Transform
@tf.function
def gpu_transform(imgX, maskX, imgY, trIMGX, trMASKX, trIMGY, trMIRROR, 
                  trINPUTSCALE, scaleINPUTVALUE, scaleINPUTCLIP, 
                  trOUTPUTSCALE, scaleOUTPUTVALUE, scaleOUTPUTCLIP, 
                  trMNI, mniVALUE, shuffle, shapeaugm, flipaugm, augment, actbrightaugm, 
                  augment_brightness, augment_contrast,
                  augment_scalefactor, augment_translatepixel, 
                  augment_rotateangle, augment_shearangle, flatten):
    
    gpu_dtype = imgX.dtype
    
    # Signal normalization
    if trINPUTSCALE:
        scaleXrange = tf.tile(tf.reshape(scaleINPUTVALUE[:,1] - scaleINPUTVALUE[:,0], (1,1,1,1,scaleINPUTVALUE.shape[0])), 
                             (imgX.shape[0],imgX.shape[1],imgX.shape[2],imgX.shape[3],1))
        scaleXbase = tf.tile(tf.reshape(scaleINPUTVALUE[:,0], (1,1,1,1,scaleINPUTVALUE.shape[0])), 
                            (imgX.shape[0],imgX.shape[1],imgX.shape[2],imgX.shape[3],1))
        imgX = ((imgX[...] - scaleXbase) / scaleXrange)
        
    if trOUTPUTSCALE:
        scaleYrange = tf.tile(tf.reshape(scaleOUTPUTVALUE[:,1] - scaleOUTPUTVALUE[:,0], (1,1,1,scaleOUTPUTVALUE.shape[0])), 
                             (imgY.shape[0],imgY.shape[1],imgY.shape[2],1))
        scaleYbase = tf.tile(tf.reshape(scaleOUTPUTVALUE[:,0], (1,1,1,scaleOUTPUTVALUE.shape[0])), 
                            (imgY.shape[0],imgY.shape[1],imgY.shape[2],1))
        imgY = ((imgY[...] - scaleYbase) / scaleYrange)
        
    
    peripheralX = tfp.stats.percentile(tf.concat([imgX[:,0,:,:,:], imgX[:,-1,:,:,:], imgX[:,:,0,:,:], imgX[:,:,-1,:,:]], axis=1), 
                                       50, axis=1, interpolation="midpoint")
    peripheralbaseX = tf.tile(tf.reshape(peripheralX, (imgX.shape[0], 1, 1, imgX.shape[3], imgX.shape[4])), (1,imgX.shape[1],imgX.shape[2],1,1))
    imgX = imgX - peripheralbaseX
    
    
    peripheralY = tfp.stats.percentile(tf.concat([imgY[:,0,:,:], imgY[:,-1,:,:], imgY[:,:,0,:], imgY[:,:,-1,:]], axis=1), 
                                       50, axis=1, interpolation="midpoint")
    peripheralbaseY = tf.tile(tf.reshape(peripheralY, (imgY.shape[0], 1, 1, imgY.shape[3])), (1,imgY.shape[1],imgY.shape[2],1))
    imgY = imgY - peripheralbaseY
    
    # Mirrorring
    if trMIRROR:
        imgX = tf.concat((imgX, tf.reverse(imgX, [1])), axis=4)
    if trMNI:
        imgX = tf.concat((imgX, tf.tile(mniVALUE, (1,1,1,imgX.shape[3],1))), axis=4)

    origshapeX = imgX.shape[3:5]
    imgX = tf.reshape(imgX, (imgX.shape[0],imgX.shape[1],imgX.shape[2],imgX.shape[3]*imgX.shape[4]))
    
    # Mirror flipping
    if flipaugm:
        if shuffle:
            flip = tf.greater(tf.random.uniform((imgX.shape[0],1,1,1), dtype=gpu_dtype), 0.5)
            imgX = tf.where(flip, tf.reverse(imgX, [1]), imgX)
            if trMASKX:
                maskX = tf.where(flip, tf.reverse(maskX, [1]), maskX)
            if trIMGY:
                imgY = tf.where(flip, tf.reverse(imgY, [1]), imgY)
        else:
            flip = tf.random.uniform([], dtype=gpu_dtype)
            if flip > 0.5:
                imgX = tf.reverse(imgX, [1])
                if trMASKX:
                    maskX = tf.reverse(maskX, [1])
                if trIMGY:
                    imgY = tf.reverse(imgY, [1])
    
    # augmentation by shape transformation
    if shapeaugm:
        if shuffle:
            trshape = (imgX.shape[0],)
        else:
            trshape = (1,)
        rotateangle = tf.random.uniform(trshape,-augment_rotateangle,+augment_rotateangle)
        shearangle = tf.random.uniform(trshape,-augment_shearangle,+augment_shearangle)
        sx = tf.random.uniform(trshape,1-augment_scalefactor,1+augment_scalefactor)
        sy = tf.random.uniform(trshape,1-augment_scalefactor,1+augment_scalefactor)
        tx = tf.random.uniform(trshape,-augment_translatepixel,+augment_translatepixel)
        ty = tf.random.uniform(trshape,-augment_translatepixel,+augment_translatepixel)
        t = tf.math.cos(rotateangle)
        matrix = tf.stack(
                    (tf.stack((sx * tf.math.cos(rotateangle), -sy * tf.math.sin(rotateangle + shearangle), tx), axis=1),
                     tf.stack((sx * tf.math.sin(rotateangle), sy * tf.math.cos(rotateangle + shearangle), ty), axis=1),
                     tf.stack((tf.constant(0.0, shape=trshape), tf.constant(0.0, shape=trshape), tf.constant(1.0, shape=trshape)), axis=1)), axis=2)
        matrix = tf.transpose(matrix, (0,2,1))
        flattransform = tfa.image.transform_ops.matrices_to_flat_transforms(matrix)
        imgX = tfa.image.transform(imgX, flattransform, interpolation="BILINEAR")
        if trIMGY:
            imgY = tfa.image.transform(imgY, flattransform, interpolation="BILINEAR")
        if trMASKX:
            maskX = tfa.image.transform(maskX, flattransform)
            
    imgX = tf.reshape(imgX, (imgX.shape[0],imgX.shape[1],imgX.shape[2])+origshapeX)
    
    # augmentation by brightness/contrast
    if shapeaugm:
        actbrightaugm = tf.reshape(actbrightaugm, (1,1,1,1,imgX.shape[4]))
        if shuffle:
            brightplus = tf.random.uniform((imgX.shape[0],1,1,1,imgX.shape[4]), -augment_brightness, +augment_brightness, dtype=gpu_dtype)
            brightmult = tf.random.uniform((imgX.shape[0],1,1,1,imgX.shape[4]), 1-augment_contrast,1+augment_contrast, dtype=gpu_dtype)
        else:
            brightplus = tf.random.uniform((1,1,1,1,imgX.shape[4]), -augment_brightness, augment_brightness, dtype=gpu_dtype)
            brightmult = tf.random.uniform((1,1,1,1,imgX.shape[4]), 1-augment_contrast,1+augment_contrast, dtype=gpu_dtype)
        imgX = tf.where(actbrightaugm, (imgX + brightplus)*brightmult, imgX)
    
    # Signal clipping
    imgX = imgX + peripheralbaseX
    
    if trINPUTSCALE:
        imgX = imgX*2-1
        scaleINPUTCLIP = tf.reshape(scaleINPUTCLIP, (1, 1, 1, 1, imgX.shape[4]))
        imgX = tf.where(scaleINPUTCLIP, tf.clip_by_value(imgX, -1, 1), imgX)
        
    imgY = imgY + peripheralbaseY
    
    if trOUTPUTSCALE:
        imgY = imgY*2-1
        scaleOUTPUTCLIP = tf.reshape(scaleOUTPUTCLIP, (1, 1, 1, imgY.shape[3]))
        imgY = tf.where(scaleOUTPUTCLIP, tf.clip_by_value(imgY, -1, 1), imgY)
    
    # flattening 
    if flatten:
        imgY = tf.reshape(imgY, (imgY.shape[0], -1))

    return imgX, maskX, imgY


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 datax,
                 datay,
                 datac,
                 mask="maskmni", # mask can be a volume or "maskmni" : it will search a mnimaskdata
                 indices=None, # if needed, a list of indices corresponding to the first dimension of the data arrays
                 pat_names=None, # if needed, a list of patient names
                 batch_size=1, # not recommended for deep learning, prefer tensorflow data management tools 
                 only_stroke=False, # filter by stroke mask value
                 dim_z=1, # gets supplementary slices in Z axis
                 shuffle=True, # shuffle slices
                 augment=True, brightaugm=True, flipaugm=True, shapeaugm=True, gpu_augment=False,# data augmentation
                 augment_scalefactor=0.05, augment_translatepixel=10, # data augmentation parameters
                 augment_rotateangle=10, augment_shearangle=5,
                 augment_brightness=0.1, augment_contrast=0.1,
                 augment_mirror=False, 
                 give_img=True, give_z=False, give_mask=False, give_meta=False, give_mni=False, give_patient_index=False, # generator outputs
                 x_lim=0, y_lim=0, # symmetrically cut the volume in X and Y axis
                 flatten_output=True, # flattens Y
                 # scaling X and Y, scale_lim can be a tuple or a list of tuples / clip corresponds to value clipping values
                 scale_input=True, scale_input_lim=(-5.0,12.0), scale_input_clip=None,
                 scale_output=False, scale_output_lim=(-5.0,12.0), scale_output_clip=False,
                 isfake=None, fakeratio=1,
                 give_latent=0, give_noise=0, 
                 gpu_float=None):
        'Initialization'
        self.datax = datax
        self.datay = datay
        self.datac = datac
        self.mask = mask
        self.isfake = isfake
        self.num_pat = datax.shape[0]
        self.dimxy = (datax.shape[1]-x_lim*2, datax.shape[2]-y_lim*2)
        self.dimz = datax.shape[3] - 2
        self.dimzinput = dim_z
        self.n_channels = datax.shape[4]
        self.augment_mirror = augment_mirror
        if self.augment_mirror:
            self.n_channels *= 2
        self.give_mni = give_mni
        if self.give_mni:
            self.n_channels += 3
        self.diminput = self.dimxy + (self.dimzinput, self.n_channels)
        self.reset_indices(indices)
        self.pat_names = pat_names
        if self.pat_names is None:
            self.pat_names = [str(i) for i in range(self.num_pat)]
        self.shuffle = shuffle
        if x_lim > 0:
            self.x_lim = slice(x_lim, -x_lim)
        else:
            self.x_lim = slice(0, None)
        if y_lim > 0:
            self.y_lim = slice(y_lim, -y_lim)
        else:
            self.y_lim = slice(0, None)
        self.flatten_output = flatten_output
        
        self.scale_input = scale_input
        if isinstance(scale_input_lim, list):
            self.scale_input_lim = scale_input_lim
        else:
            self.scale_input_lim = [scale_input_lim]
        if scale_input_clip is None:
            self.scale_input_clip = [False]*len(self.scale_input_lim)
        elif isinstance(scale_input_clip, list):
            self.scale_input_clip = scale_input_clip
        else:
            self.scale_input_clip = [scale_input_clip]
        self.scale_output = scale_output
        if isinstance(scale_output_lim, list):
            self.scale_output_lim = scale_output_lim
        else:
            self.scale_output_lim = [scale_output_lim]
        if scale_output_clip is None:
            self.scale_output_clip = [False]*len(self.scale_output_clip)
        elif isinstance(scale_output_clip, list):
            self.scale_output_clip = scale_output_clip
        else:
            self.scale_output_clip = [scale_output_clip]
            
        self.batch_size = batch_size
        self.gpu_augment = gpu_augment
        self.augment = augment
        if isinstance(brightaugm, list):
            self.brightaugm = brightaugm
        else:
            self.brightaugm = [brightaugm]*self.n_channels
        self.flipaugm = flipaugm
        self.shapeaugm = shapeaugm
        self.augment_scalefactor = augment_scalefactor
        self.augment_translatepixel = augment_translatepixel
        self.augment_rotateangle = math.radians(augment_rotateangle)
        self.augment_shearangle = math.radians(augment_shearangle)
        self.augment_brightness = augment_brightness
        self.augment_contrast = augment_contrast
        self.give_img = give_img
        self.give_z = give_z
        self.give_mask = give_mask
        self.give_meta = give_meta
        self.give_latent = give_latent
        self.give_noise = give_noise
        self.give_patient_index = give_patient_index
        self.only_stroke = only_stroke
        if gpu_float == None:
            self.gpu_float = tf.float32
        else:
            self.gpu_float = gpu_float
        self.on_epoch_end()
        
    def reset_indices(self, indices):
        self.origindices = indices
        if self.isfake is not None:
            fakelen = self.isfake[0].shape[0]
            self.list_IDs_true = [(i,j) for i in self.origindices for j in range(self.dimz)]
            self.list_IDs_fake = [(i,0) for i in range(1000,1000+fakelen)]
            if fakeratio == "all":
                self.fakenumber = "all"
            else:
                self.fakenumber = int(fakeratio*len(self.list_IDs_true))
            #(multiplyer, len(self.list_IDs_true), len(self.list_IDs_fake))
            #self.list_IDs = list_IDs_true*multiplyer + list_IDs_fake
        else:
            self.list_IDs = [(i,j) for i in self.origindices for j in range(self.dimz)]
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def getnext(self):
        # Generate indexes of the batch
        self.curbatch = 0
        batchtosend = None
        for i in range(len(self.list_IDs)):
            nextitem = self.getitem(self.list_IDs[self.indexes[i]])
            #if self.give_img and self.shuffle==True:
            #    if nextitem[0]["img"].std() < 0.02:
            #        continue
            if self.only_stroke and self.shuffle==True and np.sum(nextitem[1]) == 0:
                continue

            lenitem = len(nextitem)
            if batchtosend is None:
                batchtosend = [[] for _ in range(len(nextitem))]
                batchtosend[0] = {}
                if self.give_img :
                     batchtosend[0]["img"] = []
                if self.give_patient_index :
                     batchtosend[0]["patindex"] = []
                if self.give_meta :
                     batchtosend[0]["meta"] = []
                if self.give_mask :
                     batchtosend[0]["mask"] = []
                else:
                     batchtosend[0]["mask"] = None
                if self.give_mni :
                     batchtosend[0]["mnicoords"] = []
                else:
                     batchtosend[0]["mnicoords"] = None
                if self.give_z :
                     batchtosend[0]["zindex"] = []
            for j in range(lenitem):
                if j == 0:
                    for k in nextitem[j].keys():
                        if batchtosend[j][k] is not None:
                            batchtosend[j][k].append(nextitem[j][k])
                else:
                    batchtosend[j].append(nextitem[j])
            self.curbatch += 1
            if self.curbatch == self.batch_size:
                self.curbatch = 0
                batchtosend0 = {}
                for key in batchtosend[0].keys():
                    if batchtosend[0][key] is None:
                        batchtosend0[key] = None
                    else:
                        batchtosend0[key] = np.stack(batchtosend[0][key], axis=0)
                batchtosend1 = [np.stack(batchtosend[j], axis=0) for j in range(1,len(batchtosend))]
                batchtosend = [batchtosend0] + batchtosend1
                batchtosend = self.transform_output(batchtosend, self.list_IDs[self.indexes[i]][1])
                for key in list(batchtosend[0]):
                    if batchtosend[0][key] is None:
                        del batchtosend[0][key]
                if self.only_stroke:
                    if np.sum(batchtosend[0]["mask"] == 2) < 5:
                        batchtosend = None
                        continue
                yield tuple(batchtosend)
                batchtosend = None
    
    def transform_output(self, item, zindex):
        'WRAPPER for CPU and GPU augment'
        if item[0]["img"] is None:
            trIMGX = False
            if self.gpu_augment:
                imgX = tf.constant(0)
        else:
            trIMGX = True
            if self.gpu_augment:
                imgX = tf.convert_to_tensor(item[0]["img"], dtype=self.gpu_float)

        if item[0]["mask"] is None:
            trMASKX = False
            if self.gpu_augment:
                maskX = tf.constant(0)
        else:
            trMASKX = True
            if self.gpu_augment:
                maskX = tf.convert_to_tensor(item[0]["mask"], dtype=self.gpu_float)

        if item[1] is None:
            trIMGY = False
            if self.gpu_augment:
                imgY = tf.constant(0)
        else:
            trIMGY = True
            if self.gpu_augment:
                imgY = tf.convert_to_tensor(item[1], dtype=self.gpu_float)

        if self.scale_input:
            if self.gpu_augment:
                scaleINPUTVALUE = tf.constant(self.scale_input_lim, dtype=self.gpu_float)
                scaleINPUTCLIP = tf.constant(self.scale_input_clip, dtype=tf.bool)
        elif self.gpu_augment:
                scaleINPUTVALUE = tf.constant(0, dtype=self.gpu_float)
                scaleINPUTCLIP = tf.constant(0, dtype=tf.bool)
                
        if self.scale_output:
            if self.gpu_augment:
                scaleOUTPUTVALUE = tf.constant(self.scale_output_lim, dtype=self.gpu_float)
                scaleOUTPUTCLIP = tf.constant(self.scale_output_clip, dtype=tf.bool)
        elif self.gpu_augment:
                scaleOUTPUTVALUE = tf.constant(0, dtype=self.gpu_float)
                scaleOUTPUTCLIP = tf.constant(0, dtype=tf.bool)
                
        if self.gpu_augment:
            actBrightAugm = tf.convert_to_tensor(self.brightaugm)
        else:
            actBrightAugm = self.brightaugm

        if self.give_mni:
            mniVALUE = tf.convert_to_tensor(item[0]["mnicoords"])
        elif self.gpu_augment:
            mniVALUE = tf.constant(0)

        if self.gpu_augment:
            imgX, maskX, imgY = gpu_transform(imgX, maskX, imgY, trIMGX, trMASKX, trIMGY, self.augment_mirror, 
                                                  self.scale_input, scaleINPUTVALUE, scaleINPUTCLIP, 
                                                  self.scale_output, scaleOUTPUTVALUE, scaleOUTPUTCLIP, 
                                                  self.give_mni, mniVALUE,
                                                  self.shuffle==True, self.shapeaugm, self.flipaugm, self.augment, actBrightAugm, 
                                                  self.augment_brightness, self.augment_contrast,
                                                  self.augment_scalefactor, self.augment_translatepixel, 
                                                  self.augment_rotateangle, self.augment_shearangle,
                                                  self.flatten_output)
            if trIMGX:
                item[0]["img"] = imgX.numpy()
                if self.batch_size == 1:
                    item[0]["img"] = item[0]["img"][0]
            if trMASKX:
                item[0]["mask"] = maskX.numpy()
                if self.batch_size == 1:
                    item[0]["mask"] = item[0]["mask"][0]
            if trIMGY:
                item[1] = imgY.numpy()
                if self.batch_size == 1:
                    item[1] = item[1][0]
        else:
            item = transform_output(item, trIMGX, trMASKX, trIMGY, self.augment_mirror, 
                                    self.scale_input, self.scale_input_lim, self.scale_input_clip,
                                    self.scale_output, self.scale_output_lim, self.scale_output_clip,
                                    self.give_mni, item[0]["mnicoords"],
                                    self.shuffle==True, self.shapeaugm, self.flipaugm, self.augment, actBrightAugm, 
                                    self.augment_brightness, self.augment_contrast, 
                                    self.augment_scalefactor, self.augment_translatepixel, 
                                    self.augment_rotateangle, self.augment_shearangle,
                                    self.flatten_output, self.batch_size)
            if self.batch_size == 1:
                if item[0]["img"] is not None:
                    item[0]["img"] = item[0]["img"][0]
                if item[0]["mask"] is not None:
                    item[0]["mask"] = item[0]["mask"][0]
                item[1] = item[1][0]
                
        if self.give_mni:
            del item[0]["mnicoords"]
        if self.give_latent:
            item[0]["latent"] = np.random.normal(size=(self.give_latent,)).astype(np.float32) 
        if self.give_noise:
            item[0]["noise"] = np.random.normal(size=(self.give_noise,)).astype(np.float32) 
            
        return item

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.isfake:
            if self.fakenumber == "all":
                self.list_IDs = random.sample(self.list_IDs_fake, len(self.list_IDs_true))     
            else:
                self.list_IDs = self.list_IDs_true + random.sample(self.list_IDs_fake, self.fakenumber)        
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == "patient":
            mixedpatients = np.arange(len(self.list_IDs)//self.batch_size)
            np.random.shuffle(mixedpatients)
            self.indexes = np.array([mixedpatients[j]*self.batch_size+i for j in range(len(mixedpatients)) for i in range(self.batch_size)])
        elif self.shuffle == True:
            np.random.shuffle(self.indexes)

    def getitem(self, item):
        'Generates data containing samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ID, Z = item
        if self.isfake is not None and ID >= 1000:
            dataz = 0
            thisX = np.array(self.isfake[0][ID-1000,self.x_lim,self.y_lim,dataz,np.newaxis,0])
            thisY = np.array(self.isfake[1][ID-1000,self.x_lim,self.y_lim,dataz,:])
        else:
            zshift = int((self.dimzinput-1)/2)
            dataz = Z+zshift
            datazX = slice(dataz-zshift,dataz+zshift+1)
            thisX = {}
            thisY = np.array(self.datay[ID,self.x_lim,self.y_lim,dataz,:])
            if self.give_img:
                thisX["img"] = np.array(self.datax[ID,self.x_lim,self.y_lim,datazX,:])
                if self.give_mni:
                    thisX["mnicoords"] = mnicoordsdata[self.x_lim,self.y_lim,Z:Z+1]
            #brainmask = mnimaskdata[...,Z,:] # datay[ID,self.x_lim,self.y_lim,Z,1:2],
            #thisY = np.logical_and(brainmask, thisY) + brainmask
        if self.give_meta :
            thisX["meta"] = np.array(self.datac[ID])
        if self.give_patient_index :
            thisX["patindex"] = ID
        thisX["mask"] = None
        if self.give_mask and self.isfake is None:
            if type(self.mask) is str:
                if self.mask == "maskmni":
                    # mnimaskdata must be defined in principal script
                    thisX["mask"] =  mnimaskdata[self.x_lim,self.y_lim,Z]
            else:
                thisX["mask"] =  self.mask[ID,self.x_lim,self.y_lim,dataz]
        if self.give_z and self.isfake is None:
            thisX["zindex"] = Z
        retarray = [thisX, thisY]
        return retarray
