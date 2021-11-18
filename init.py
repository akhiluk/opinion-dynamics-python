import os
import subprocess
from os.path import exists, join, basename, splittext

if not exists('deepspeech-0.6.1-models'):
    cmdString = "pip install -q deepspeech-gpu==0.6.1 youtube-dl"
    subprocess.call(cmdString, shell=True)
    cmdString = "wget https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz"
    subprocess.call(cmdString, shell=True)
    cmdString = "tar xvfz deepspeech-0.6.1-models.tar.gz"
    subprocess.call(cmdString, shell=True)