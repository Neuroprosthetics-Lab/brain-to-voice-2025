# An instantaneous voice synthesis neuroprosthesis

Maitreyee Wairagkar, Nicholas S. Card, Tyler Singer-Clark, Xianda Hou, Carrina Iacobacci, Lee M. Miller, Leigh R. Hochberg, David M. Brandman*, Sergey D. Stavisky*    
**Co-senior authors*

preprint: https://doi.org/10.1101/2024.08.14.607690

## Overview

This repository contains the code for the offline implementation of the brain-to-voice synthesis methods described in the paper *"An instantaneous voice synthesis neuroprosthesis"*. A demo to synthesize voice from intracortical neural signals during speech task is provided. 

## Installation

* **Python environment:** Set up a new conda environment `b2voice` using the provided `requirements.yml` as follows:

```
conda env create --file requirements.yml
conda activate b2voice
```
This will install all the required Python packages to run this code (installation will take around 15-20 minutes). 

* **System requirements:** This code runs on the GPU for optimal performance. It has been tested on Linux (20.04) with RTX 3090 GPU and RTX A6000 GPU as well as on MacOS (13.5) with M1 Pro processor. 

* **Install LPCNet:** Brain-to-voice synthesis uses a pre-trained LPCNet vocoder which can be installed from here https://github.com/xiph/LPCNet. Install this into the `dependencies` folder. This installation will take around 7-10 mins. 

## Data setup

The dataset contains individual trials of neural features extracted recorded during the speech task as well as the pre-trained brain-to-voice decoders to synthesize voice from these trials. Neural data is divided into multiple blocks containing several speech trials. Save the data folder with neural data blocks in the `data` folder provided in this repository. Place the brain-to-voice decoders in the `data/trained_models` folder.

## Demo for synthesizing voice from intracortical neural activity

A demo to synthesize voice from neural activity using the brain-to-voice decoder is provded in the `inference.ipynb` notebook. Run the notebook to load the data, the decoder and then run the inference on each trial individually to synthesize intelligible voice. In the offline implementation, output audio files are generated and saved in the `aud` folder. 

The README files within the folders provide further information.