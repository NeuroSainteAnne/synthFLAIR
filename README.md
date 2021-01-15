# synthFLAIR
Software designed to convert MRI diffusion-weighted sequences into FLAIR sequences

### Prerequisites
Python 3.8

### Usage
First, clone the git directory :

```
git clone https://github.com/NeuroSainteAnne/synthFLAIR.git
cd synthFLAIR/
```

Then install required modules:

```
pip install -r requirements.txt
pip install jupyterlab
```

Open the jupyter notebook and follow instructions to train your model

```
jupyter notebook
```


### Orthanc module

Once the model is trained and saved, you can use the Orthanc module to automatically convert DWI volumes into FLAIR using a lightweight DICOM server.
Follow the intructions at [Orthanc Module Readme](orthanc_module/README.md)
