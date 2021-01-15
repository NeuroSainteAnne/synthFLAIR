import tensorflow as tf
import tensorflow.keras.backend as K
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from modules.generator import DataGenerator
import os
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap, colorConverter

def figure(datax, datay, datac, datamask,
           generator, 
           train_names, this_valid_index,
           epoch=0, n_patients=5,
           save=False, multiquality=False, maxquality=True, 
           show=False, output="figures", savestr="validation_at_epoch_%%%",
           slices=None, show_outline=False):
    
    savestr = savestr.replace("%%%",str(epoch))
    
    if slices is None:
        slices = (5,20)
        
    from generator import DataGenerator
    random.shuffle(this_valid_index)
    valid_generator_ordered = DataGenerator(datax,
                                            datay,
                                            datac=datac.astype(np.uint8),
                                            mask=datamask,
                                            indices=this_valid_index, shuffle=False, 
                                    flatten_output=False, batch_size=1, dim_z=1,
                                    augment=False, shapeaugm=False, brightaugm=False, flipaugm=False, gpu_augment=False, 
                                    scale_input=True, scale_input_lim=[(-5,12),(-5,12),(0,7500.0)], scale_input_clip=[False,False,False],
                                    scale_output=True, scale_output_lim=(-5,10), scale_output_clip=True,
                                    only_stroke=False, give_mask=True, give_meta=True, give_patient_index=True)
    dsVO = tf.data.Dataset.from_generator(valid_generator_ordered.getnext, 
                                          ({"img":K.floatx(),"mask":K.floatx(),"meta":K.floatx(),"patindex":K.floatx()}, K.floatx()), 
                                          ({"img":(256,256,1,3), "mask":(256,256,1),"meta":(1,2),"patindex":(1,)}, 
                                           (256,256,1))).repeat().batch(25).prefetch(16)
    
    n_cols = 4
    qualities = [0,2]
    if multiquality:
        n_cols += len(qualities)-1
    
        
    if not show:
        plt.ioff()
        
    patients = []
    for i in dsVO.take(n_patients):
        patients.append(i)
        
    if n_patients > 1:
        n_slices = n_patients
    else:
        print_slices = list(np.argwhere(np.max(patients[0][0]["mask"][...,0].numpy(), axis=(1,2)) == 2).flatten())
        n_slices = len(slices)
        
    color1 = colorConverter.to_rgba('red',alpha=0.0)
    color2 = colorConverter.to_rgba('red',alpha=0.8)
    cmap1 = LinearSegmentedColormap.from_list('my_cmap',[color1,color2],256)
    plt.rcParams['figure.figsize'] = [5*4, 5*n_slices]
    n = 0
    for j in range(len(patients)):
        i = patients[j]
        
        if n_patients > 1:
            print_slices = [random.randrange(slices[0],slices[1])]
        
        for z in print_slices:
            maskcut = np.ones_like(np.flipud(i[0]["mask"][z,...,0].numpy().T)>=1)
            plt.subplot(n_slices,n_cols,n*n_cols+1)
            plt.title('Diffusion imaging (b0)')
            mri="h0"
            if i[0]["meta"][0,0,1] == 1:
                mri="h24"
            plt.ylabel(train_names[int(i[0]["patindex"].numpy()[0,0])]+"_"+mri+"_q"+str(int(i[0]["meta"].numpy()[0,0,0])))
            plt.imshow(maskcut*(1+np.flipud(i[0]["img"][z,...,0,0].numpy().T)),cmap='gray',vmin=0.2,vmax=1.4)
            plt.subplot(n_slices,n_cols,n*n_cols+2)
            plt.title('Diffusion imaging (b1000)')
            plt.imshow(maskcut*(1+np.flipud(i[0]["img"][z,...,0,1].numpy().T)),cmap='gray',vmin=0.2,vmax=1.4)
            roi = ndimage.laplace(ndimage.binary_dilation(np.flipud(i[0]["mask"][z,...,0].numpy().T)>1.5, iterations=4))
            plt.imshow(roi, cmap=cmap1, alpha=0.5)
            plt.subplot(n_slices,n_cols,n*n_cols+3)
            plt.title('FLAIR imaging')        
            plt.imshow(maskcut*(1+np.flipud(i[1][z].numpy()[...,0].T)),cmap='gray',vmin=0.2,vmax=1.4)

            if not multiquality:
                if maxquality:
                    qualities = [2]
                else:
                    qualities = [i[0]["meta"][...,0][0].numpy()[0]]
            for q in range(len(qualities)):
                plt.subplot(n_slices,n_cols,n*n_cols+4+q)
                qualarr = np.tile(qualities[q], (i[0]["img"].shape[0],1))
                prediction = generator.predict([i[0]["img"], qualarr])
                predictionT = np.reshape(prediction,(prediction.shape[0],256,256))
                syntext = 'Synthetic FLAIR (model created)'
                if multiquality:
                    syntext = 'Synthetic FLAIR (quality'+str(qualities[q])+')'
                plt.title(syntext)
                plt.imshow(maskcut*(1+np.flipud(predictionT[z].T)),cmap='gray',vmin=0.2,vmax=1.4)
            n += 1
    fig1 = plt.gcf()
    if show:
        plt.show()
    if save:
        fig1.savefig(os.path.join(output,savestr+'.png'))
    if not show:
        plt.close()
    if save:
        return "Saved to " + os.path.join(output,savestr+'.png')