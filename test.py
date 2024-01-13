#this file is only to test some utility
import numpy as np
import pandas

import util.mergeDataFrames as mDF

df = mDF.getMergedDataFrames("11","english",["MFCC","MEL","TEMPO"])
df = pandas.DataFrame(df)
#df = pandas.DataFrame(df)
df.to_csv("testMerge.csv",mode='w', header=False, index=False)

import util.ExtractFeatures as ef

#ef.extract_english_features(11,16,"CHROMA_STFT")