## Installation :
- python 3.7 or later
- orthanc with python plugin

## Configuration :
- set absolute path to orthanc plugins directory in orthanc.json, line 56
- client name and corresponding AET name should be defined in orthanc.json, line 213. Client name and AET should be identical
- set absolute path to synthflair.py in orthanc.json, line 614
- set absolute path to model in synthflair.py, line 12

## Usage :
- launch Orthanc with following command : Orthanc /path/to/orthanc.json
- send DWI sequences to the Orthanc server - synthFLAIR should be automatically sent back
