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
logging.getLogger('tensorflow').setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)
from UM import PreProcessing
pp = PreProcessing()
pp.preprocess()
pp.train()
#pp.predict(os.path.join("Multi_en_Arg/pb"),os.path.join("Multi_en_Pred/pb"))
