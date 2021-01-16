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

def reversepad(volume, padspecs):
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
    label_im, nb_labels = label(vol)
    label_count = np.bincount(label_im.ravel().astype(np.int))
    label_count[0] = 0
    return label_im == label_count.argmax()

def normalize(vol, mask):
    masked_vol = vol[mask]
    mean_val, sd_val = np.mean(masked_vol), np.std(masked_vol)
    vol = (vol - mean_val) / sd_val
    return vol

def nifti2array(diffpath):
    # load diffusion
    diffnib = nib.load(diffpath)
    diffdata = diffnib.get_fdata()
    
    # differenciate b1000 from b0
    vol1 = diffdata[...,0]
    vol2 = diffdata[...,1]
    if vol1.mean() > vol2.mean():
        b0 = vol1
        b1000 = vol2
    else:
        b0 = vol2
        b1000 = vol1
        
    b0, _ = padvolume(b0)
    b1000, padspecs = padvolume(b1000)
    
    # ADC calculation
    crudemask = (b0 >= 1) & (b1000 >= 1) # exclude zeros for ADC calculation
    adc = np.zeros(b0.shape, b1000.dtype)
    adc[crudemask] = -1. * float(1000) * np.log(b1000[crudemask] / b0[crudemask])
    adc[adc < 0] = 0
    
    # mask computation
    b0_mask, mask0 = median_otsu(b0, 1, 1)
    b1000_mask, mask1000 = median_otsu(b1000, 1, 1)
    mask = binary_fill_holes(morphology.binary_dilation(brain_component(mask0 & mask1000)))
    mask = mask & (b0 >= 1) & (b1000 >= 1)
    
    # normalisation
    b0 = normalize(b0, mask)
    b1000 = normalize(b1000, mask)
    
    # adjust for model input
    b0 = ((b0 + 5) / (12 + 5))*2-1
    b1000 = ((b1000 + 5) / (12 + 5))*2-1
    adc = ((adc) / (7500))*2-1
    
    # export for model input
    stacked = np.stack([b0,b1000,adc]).transpose([3,2,1,0])[:,:,::-1,np.newaxis,:]
    qualarr = np.tile(2, (stacked.shape[0],1))
    
    return stacked, qualarr, padspecs, diffnib.affine
    
def array2nifti(savearray, padspecs, affine, outpath):
    synthflair = savearray[:,:,::-1,0].transpose(2,1,0)
    synthflair = synthflair - synthflair.min()
    synthflair = 256*(synthflair / synthflair.max())
    synthflair = reversepad(synthflair, padspecs)
    synthflairnib = nib.Nifti1Image(synthflair, affine=affine)
    nib.save(synthflairnib, outpath)