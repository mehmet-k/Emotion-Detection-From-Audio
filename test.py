#this file is only to test some utility
import numpy as np
import pandas

import util.mergeDataFrames as mDF

import util.ExtractFeatures


#util.ExtractFeatures.extract_english_features(11,19,"CHROMA_VQT")

util.ExtractFeatures.extract_mandarin_features(1,11,"MFCC")
util.ExtractFeatures.extract_mandarin_features(1,11,"MEL")
util.ExtractFeatures.extract_mandarin_features(1,11,"RMS")
util.ExtractFeatures.extract_mandarin_features(1,11,"CHROMA_VQT")
util.ExtractFeatures.extract_mandarin_features(1,11,"CHROMA_STFT")
util.ExtractFeatures.extract_mandarin_features(1,11,"TEMPO")

#util.ExtractFeatures.extract_english_features(11,19,"TEMPO")