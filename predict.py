import warnings
import os
import logging
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
pp = PreProcessing()
pp.preprocess()
#path needs to be updated if trained a new model with a different path
pp.vald_and_test(os.path.join("UD2OIE_Arg_saved/pb"),os.path.join("UD2OIE_pred_saved/pb"))