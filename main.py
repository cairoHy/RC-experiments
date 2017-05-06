import os

from models.attention_sum_reader import AttentionSumReader

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = AttentionSumReader()
    model.execute()