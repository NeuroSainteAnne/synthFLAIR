import os
import pydicom
import glob
import numpy as np
import nibabel as nib
from skimage import filters, morphology
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import label
from dipy.segment.mask import median_otsu

def padvolume(volume):
    "Applies a padding/cropping to a volume in order to hav 256x256 size"
    padx1 = padx2 = pady1 = pady2 = 0
    orig_shape = volume.shape
    if orig_shape[0] < 256 or orig_shape[1] < 256:
        if orig_shape[0] < 256:
            padx1 = int((256.0 - orig_shape[0])/2)
            padx2 = 256 - orig_shape[0] - padx1
        if orig_shape[1] < 256:
            pady1 = int((256.0 - orig_shape[1])/2)
            pady2 = 256 - orig_shape[1] - pady1
        volume = np.pad(volume, ((padx1, padx2),(pady1,pady2),(0,0)), mode="edge")
    cutx1 = cutx2 = cuty1 = cuty2 = 0
    if orig_shape[0] > 256 or orig_shape[1] > 256:
        if orig_shape[0] > 256:
            cutx1 = int((orig_shape[0]-256.0)/2)
            cutx2 = orig_shape[0] - 256 - cutx1
            volume = volume[cutx1:-cutx2,:,:]
        if orig_shape[1] > 256:
            cuty1 = int((orig_shape[1]-256.0)/2)
            cuty2 = orig_shape[1] - 256 - cuty1
            volume = volume[:,cuty1:-cuty2,:]
    return volume, (padx1, padx2, pady1, pady2, cutx1, cutx2, cuty1, cuty2)

def zpadding(volume, zpad):
    orig_shape = volume.shape
    if orig_shape[2] < zpad:
        padz1 = int((zpad - orig_shape[2])/2)
        padz2 = zpad - orig_shape[2] - padz1
        volume = np.pad(volume, ((0,0),(0,0),(padz1,padz2)), mode="minimum")
    elif orig_shape[2] > zpad:
        cutz1 = int((orig_shape[2] - zpad)/2)
        cutz2 = orig_shape[2] - zpad - cutz2
        volume = volume[:,:,cutz1:cutz2]
    return volume
        

def reversepad(volume, padspecs):
    "Reserves a previously applied padding"
    padx1 = padspecs[0]
    padx2 = padspecs[1]
    pady1 = padspecs[2]
    pady2 = padspecs[3]
    cutx1 = padspecs[4]
    cutx2 = padspecs[5]
    cuty1 = padspecs[6]
    cuty2 = padspecs[7]
    
    if cutx1>0 or cutx2>0:
        volume = np.pad(volume, ((cutx1, cutx2),(0,0),(0,0)), mode="edge")
    if cuty1>0 or cuty2>0:
        volume = np.pad(volume, ((0,0),(cuty1,cuty2),(0,0)), mode="edge")
    if padx1>0 or padx2>0:
        volume = volume[padx1:-padx2,:,:]
    if pady1>0 or pady2>0:
        volume = volume[:,pady1:-pady2,:]
        
    return volume

def brain_component(vol):
    "Select the largest component in a mask (brain)"
    label_im, nb_labels = label(vol)
    label_count = np.bincount(label_im.ravel().astype(np.int))
    label_count[0] = 0
    return label_im == label_count.argmax()

def normalize(vol, mask):
    "Normalization of a volume"
    masked_vol = vol[mask]
    mean_val, sd_val = np.mean(masked_vol), np.std(masked_vol)
    vol = (vol - mean_val) / sd_val
    return vol

def adccompute(b0, b1000):
    "Computes ADC map"
    crudemask = (b0 >= 1) & (b1000 >= 1) # exclude zeros for ADC calculation
    adc = np.zeros(b0.shape, b1000.dtype)
    adc[crudemask] = -1. * float(1000) * np.log(b1000[crudemask] / b0[crudemask])
    adc[adc < 0] = 0
    return adc

def maskcompute(b0, b1000):
    "Computes a brain mask using otsu method"
    b0_mask, mask0 = median_otsu(b0, 1, 1)
    b1000_mask, mask1000 = median_otsu(b1000, 1, 1)
    mask = binary_fill_holes(morphology.binary_dilation(brain_component(mask0 & mask1000)))
    mask = mask & (b0 >= 1) & (b1000 >= 1)
    return mask
    
def splitdiffusion(diffdata):
    "Splits b0 and b1000 based on value mean"
    vol1 = diffdata[...,0]
    vol2 = diffdata[...,1]
    if vol1.mean() > vol2.mean():
        b0 = vol1
        b1000 = vol2
    else:
        b0 = vol2
        b1000 = vol1
    return b0, b1000

def nifti2array(diffpath):
    # load diffusion
    diffnib = nib.load(diffpath)
    diffdata = diffnib.get_fdata()
    
    # differenciate b1000 from b0
    b0, b1000 = splitdiffusion(diffdata)
    
    stacked, mask, padspecs = splitdwi2array(b0,b1000,adjust=True,z_pad=False)
    stacked = stacked.transpose([3,2,1,0])[:,:,::-1,np.newaxis,:]
    qualarr = np.tile(2, (stacked.shape[0],1))
    return stacked, qualarr, padspecs, diffnib.affine
    
def twoniftis2array(b0path, b1000path,z_pad=None):
    # load niftis
    diff0nib = nib.load(b0path)
    diff0data = diff0nib.get_fdata()
    diff1000nib = nib.load(b1000path)
    diff1000data = diff1000nib.get_fdata()
    return splitdwi2array(diff0data,diff1000data,adjust=False,z_pad=z_pad)

def flairnifti2array(flairpath, mask, z_pad=None):
    # load nifti
    flairnib = nib.load(flairpath)
    flair = flairnib.get_fdata()
    
    # pad
    flair, padspecs = padvolume(flair)
    if z_pad is not None:
        flair = zpadding(flair, z_pad)
        
    # normalisation
    flair = normalize(flair, mask)
    
    return flair

def splitdwi2array(b0,b1000,adjust=False,z_pad=None):
        
    # pad/crop volumes to 256x256
    b0, _ = padvolume(b0)
    b1000, padspecs = padvolume(b1000)
    
    #Z-pad
    if z_pad is not None:
        b0 = zpadding(b0, z_pad)
        b1000 = zpadding(b1000, z_pad)
    
    # ADC calculation
    adc = adccompute(b0, b1000)
    
    # mask computation
    mask = maskcompute(b0, b1000)
    
    # normalisation
    b0 = normalize(b0, mask)
    b1000 = normalize(b1000, mask)
    
    # adjust for model input
    if adjust:
        b0 = ((b0 + 5) / (12 + 5))*2-1
        b1000 = ((b1000 + 5) / (12 + 5))*2-1
        adc = ((adc) / (7500))*2-1
    
    # export for model input
    stacked = np.stack([b0,b1000,adc])
    
    return stacked, mask, padspecs
    
def array2nifti(savearray, padspecs, affine, outpath):
    synthflair = savearray[:,:,::-1,0].transpose(2,1,0)
    synthflair = synthflair - synthflair.min()
    synthflair = 256*(synthflair / synthflair.max())
    synthflair = reversepad(synthflair, padspecs)
    synthflairnib = nib.Nifti1Image(synthflair, affine=affine)
    nib.save(synthflairnib, outpath)