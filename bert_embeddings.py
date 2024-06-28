# todo
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt
import csv

def get_BERT(utterances):
    tf.get_logger().setLevel('ERROR')
    bert_model = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1"
    bert_preprocess =  "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_preprocess_model = hub.KerasLayer(bert_preprocess)
    bert_model = hub.KerasLayer(bert_model)
    all_bert_reps = []
    count = 0
    for utterance in utterances:
        utterancet = utterancet.replace("\n", '')
        utterance = utterance.replace(",", '')
        utterance = utterance.replace('.', '')
        utterance = [utterance]
        print(utterance)
        asr_preprocessed = bert_preprocess_model(utterance)
        bert_results = bert_model(asr_preprocessed)
        all_bert_reps.append([count,bert_results["pooled_output"]])
        # print(bert_results["pooled_output"])
        count += 1
    return all_bert_reps


