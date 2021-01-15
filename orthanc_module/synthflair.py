import sys

import os
import pydicom
import glob
import numpy as np
from skimage import filters, morphology
from scipy.ndimage.morphology import binary_fill_holes
from dipy.segment.mask import median_otsu
from nipy.labs.mask import largest_cc
import tensorflow as tf
generator = tf.keras.models.load_model("/path/to/saved_generator")

import orthanc
import io
import json

def OnChange(changeType, level, resourceId):
    if changeType == orthanc.ChangeType.STABLE_SERIES:
        instances = json.loads(orthanc.RestApiGet("/series/"+resourceId))["Instances"]
        files = []
        for instanceId in instances:
            f = orthanc.GetDicomForInstance(instanceId)
            file = pydicom.dcmread(io.BytesIO(f))
            if "SynthFLAIR" in file.SeriesDescription:
                return
            files.append(file)
        metadata = json.loads(orthanc.RestApiGet("/instances/"+instances[0]+"/metadata?expand"))
        
        slices = []
        for f in files:
            if hasattr(f, 'SliceLocation'):
                slices.append(f)
        b0 = []
        b1000 = []
        for s in slices:
            if str(s[0x0043, 0x1039][0]) == "0":
                b0.append(s)
            if "1000" in str(s[0x0043, 0x1039][0]):
                b1000.append(s)
        b0 = sorted(b0, key=lambda s: s.SliceLocation)
        b1000 = sorted(b1000, key=lambda s: s.SliceLocation)
        print(len(b0), len(b1000))

        orig_shape = (b0[0].pixel_array.shape) + (len(b0),)
        b0_src = np.zeros(orig_shape)
        for i, s in enumerate(b0):
            orig_dtype = s.pixel_array.dtype
            b0_src[:,:,i] = s.pixel_array
        b1000_src = np.zeros(orig_shape)
        for i, s in enumerate(b1000):
            b1000_src[:,:,i] = s.pixel_array

        #b0_src = np.pad(b0_src, ((10,10),(0,0),(0,0)))[:,20:,:]
        #b1000_src = np.pad(b1000_src, ((10,10),(0,0),(0,0)))[:,20:,:]
        #orig_shape = b0_src.shape

        b0_padded = b0_src.copy()
        b1000_padded = b1000_src.copy()
        padx1 = padx2 = pady1 = pady2 = 0
        if orig_shape[0] < 256 or orig_shape[1] < 256:
            if orig_shape[0] < 256:
                padx1 = int((256.0 - orig_shape[0])/2)
                padx2 = 256 - orig_shape[0] - padx1
            if orig_shape[1] < 256:
                pady1 = int((256.0 - orig_shape[1])/2)
                pady2 = 256 - orig_shape[1] - pady1
            b0_padded = np.pad(b0_padded, ((padx1, padx2),(pady1,pady2),(0,0)), mode="edge")
            b1000_padded = np.pad(b1000_padded, ((padx1, padx2),(pady1,pady2),(0,0)), mode="edge")
        cutx1 = cutx2 = cuty1 = cuty2 = 0
        if orig_shape[0] > 256 or orig_shape[1] > 256:
            if orig_shape[0] > 256:
                cutx1 = int((orig_shape[0]-256.0)/2)
                cutx2 = orig_shape[0] - 256 - cutx1
                b0_padded = b0_padded[cutx1:-cutx2,:,:]
                b1000_padded = b1000_padded[cutx1:-cutx2,:,:]
            if orig_shape[1] > 256:
                cuty1 = int((orig_shape[1]-256.0)/2)
                cuty2 = orig_shape[1] - 256 - cuty1
                b0_padded = b0_padded[:,cuty1:-cuty2,:]
                b1000_padded = b1000_padded[:,cuty1:-cuty2,:]

        maskdata = (b0_padded >= 1) & (b1000_padded >= 1) # exclude zeros for ADC calculation
        adc_padded = np.zeros(b0_padded.shape, b0_padded.dtype)
        adc_padded[maskdata] = -1. * float(1000) * np.log(b1000_padded[maskdata] / b0_padded[maskdata])
        adc_padded[adc_padded < 0] = 0

        b0_mask, mask = median_otsu(b0_padded, 1, 1)
        b1000_mask, mask1000 = median_otsu(b1000_padded, 1, 1)
        mask_padded = binary_fill_holes(morphology.binary_dilation(largest_cc(mask & mask1000)))
        mask_padded = mask_padded & (b0_padded >= 1) & (b1000_padded >= 1)

        masked_b0 = b0_padded[mask_padded]
        mean_b0, sd_b0 = np.mean(masked_b0), np.std(masked_b0)
        masked_b1000 = b1000_padded[mask_padded]
        mean_b1000, sd_b1000 = np.mean(masked_b1000), np.std(masked_b1000)

        b0_padded = (b0_padded - mean_b0) / sd_b0
        b1000_padded = (b1000_padded - mean_b1000) / sd_b1000

        b0_padded = ((b0_padded + 5) / (12 + 5))*2-1
        b1000_padded = ((b1000_padded + 5) / (12 + 5))*2-1
        adc_padded = ((adc_padded) / (7500))*2-1
        b0_padded[b0_padded > 1] = 1
        b0_padded[b0_padded < -1] = -1
        b1000_padded[b1000_padded > 1] = 1
        b1000_padded[b1000_padded < -1] = -1

        stacked = np.stack([b0_padded,b1000_padded,adc_padded]).transpose([3,2,1,0])[:,:,::-1,np.newaxis,:]
        qualarr = np.tile(2, (stacked.shape[0],1))

        synthflair = generator.predict([stacked, qualarr])[:,:,::-1,0].transpose(2,1,0)
        synthflair = (np.max(b1000_src) - np.min(b1000_src))*((synthflair-np.min(synthflair))/(np.max(synthflair) - np.min(synthflair)))

        sflair = []
        flairinstances = []
        uidStudy = pydicom.uid.generate_uid()
        uidPrefix = pydicom.uid.generate_uid()[:-3]
        if not os.path.exists("/tmp/dicom/"+uidStudy):
            os.makedirs("/tmp/dicom/"+uidStudy)
        for i, s in enumerate(b1000):
            news = s.copy()
            news.SeriesDescription = "SynthFLAIR - " + news.SeriesDescription
            news.PixelData = synthflair[:,:,i].astype(orig_dtype).tobytes()
            news.SOPInstanceUID = uidPrefix + "." + str(i)
            news.file_meta.MediaStorageSOPInstanceUID = uidPrefix + "." + str(i)
            news.SeriesInstanceUID = uidStudy
            news.SeriesNumber = news.SeriesNumber + 99
            news.InstanceNumber = i
            #news.save_as("/tmp/dicom/"+uidStudy+"/"+str(i)+".dcm")
            bytesnews = io.BytesIO()
            news.save_as(bytesnews)
            instanceinfo = json.loads(orthanc.RestApiPost("/instances", bytesnews.getvalue()))
            if i == 0:
                patientId = instanceinfo["ParentPatient"]
                seriesId = instanceinfo["ParentSeries"]
            flairinstances.append(instanceinfo["ID"])
            #print(bytesnews)
            #t = orthanc.CreateDicomInstance(bytesnews.getvalue())
        #print(flairinstances)
        
        request = {
            "Resources" : flairinstances
          }
        print(metadata["RemoteAET"])
        print(metadata)
        orthanc.RestApiPost("/modalities/" + metadata["RemoteAET"] + "/store", json.dumps(request));
            
        #orthanc.RestApiDelete("/series/"+resourceId)
        #orthanc.RestApiDelete("/series/"+seriesId)
        orthanc.RestApiDelete("/patients/"+patientId)

orthanc.RegisterOnChangeCallback(OnChange)

