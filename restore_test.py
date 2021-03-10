import warnings
import os
import logging
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')
warnings.filterwarnings('ignore')
def warn(*args, **kwargs):
    pass
warnings.warn = warn
#warnings.filterwarnings("error")
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
tic = time.perf_counter()
from Utils_test import PreProcessing
pp = PreProcessing()
pp.preprocess()
pp.restore(os.path.join("SavedArg128/pb"),os.path.join("SavedPred128/pb"),17)
toc = time.perf_counter()
print(f"The inference time is {toc - tic:0.4f} seconds")
print("end")