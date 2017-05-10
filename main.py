import os

from models import *

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = AttentionSumReader()
    model.execute()
