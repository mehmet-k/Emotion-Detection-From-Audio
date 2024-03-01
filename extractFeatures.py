import time

import numpy as np
import pandas

import util.mergeDataFrames as mDF

import util.ExtractFeatures


#util.ExtractFeatures.extract_english_features(11,19,"CHROMA_VQT")

mfcc_start_time = time.time()
util.ExtractFeatures.extract_mandarin_features(1,11,"MFCC")
mfcc_end_time = time.time()

mel_start_time = time.time()
util.ExtractFeatures.extract_mandarin_features(1,11,"MEL")
mel_end_time = time.time()

rms_start_time = time.time()
util.ExtractFeatures.extract_mandarin_features(1,11,"RMS")
rms_end_time = time.time()

vqt_start_time = time.time()
util.ExtractFeatures.extract_mandarin_features(1,11,"CHROMA_VQT")
vqt_end_time = time.time()

stft_start_time = time.time()
util.ExtractFeatures.extract_mandarin_features(1,11,"CHROMA_STFT")
stft_end_time = time.time()
#util.ExtractFeatures.extract_english_features(11,19,"TEMPO")


print("MFCC: ", mfcc_end_time-mfcc_start_time)
print("MEL: ", mfcc_end_time- mfcc_start_time)
print("RMS: ",rms_end_time-rms_start_time)
print("CHROMA_VQT: ",vqt_end_time-vqt_start_time)
print("CHROMA_sTFT: ",stft_end_time - stft_start_time)
#util.ExtractFeatures.extract_english_features(11,19,"MFCC")
"""
util.ExtractFeatures.extract_english_features(11,19,"MEL")
util.ExtractFeatures.extract_english_features(11,19,"RMS")
util.ExtractFeatures.extract_english_features(11,19,"CHROMA_STFT")
util.ExtractFeatures.extract_english_features(11,19,"TEMPO")
"""
