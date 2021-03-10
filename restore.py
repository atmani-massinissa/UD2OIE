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

from UM import PreProcessing
#tic = time.perf_counter()
pp = PreProcessing()
pp.preprocess()
#path needs to be updated if trained a new model with a different path
pp.restore(os.path.join("UD2OIE_Arg_saved/pb"),os.path.join("UD2OIE_pred_saved/pb"),29)
#toc = time.perf_counter()
#t = pp.toc - pp.tic
#print(f"Inference time {t:0.4f} seconds")

