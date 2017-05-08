import os

from models.attention_over_attention_reader import AoAReader

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = AoAReader()
    model.execute()
