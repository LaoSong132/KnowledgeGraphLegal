# import packages
import pickle

from flair.models import SequenceTagger
from flair.models import RelationExtractor
from flair.data import Sentence, Corpus
from flair.datasets.relation_extraction import CoNLLUCorpus
from flair.trainers import ModelTrainer
from flair.embeddings import TransformerWordEmbeddings
import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional, Sequence, Dict, Tuple, Any

import conllu

from flair.data import Sentence, Corpus, Token, FlairDataset, Span, RelationLabel, SpanLabel
from flair.datasets.base import find_train_dev_test_files

print(os.getcwd())
print(os.listdir('.'))

ner_model_dir = 'models/ner'
re_model_dir = 'models/re'
initial_model = 'models/final-model.pt'
ner_data_dir = 'PATTERNS/CORPUS_GROUPS(NER)/'
re_data_dir = 'PATTERNS/CORPUS_GROUPS/'
ERPATTERN_FILE = 'PATTERNS/CASE_LAW_COUNSEL/cl_counsel_re_patterns_DO_NOT_DELETE_4_annotation_Afsar_v4.xlsx'

if __name__ == '__main__':
    Sentence_TEXT = pd.read_excel(ERPATTERN_FILE, sheet_name='ER_PATTERNS')
    sents = []
    loaded_model_ner = SequenceTagger.load(ner_model_dir + '/final-model.pt')
    loaded_model_re = RelationExtractor.load(re_model_dir + '/final-model.pt')
    for index, row in Sentence_TEXT.iterrows():
        sents.append(row['Sentence_TEXT'])
    print(len(sents))

    entity_dict = {}
    index = 1
    ke_list = []

    for i in range(len(sents)):
        if i > 200:
            break
        print(str(i) + ' ' + '*' * 20)
        sentence = Sentence(sents[i])
        loaded_model_ner.predict(sentence)
        loaded_model_re.predict(sentence)
        relations = sentence.get_labels('relation')
        for relation in relations:
            head = relation.head.text
            tail = relation.tail.text

            if entity_dict.get(head) is None:
                entity_dict[head] = index
                head_int = index
                index += 1
            else:
                head_int = entity_dict[head]

            if entity_dict.get(tail) is None:
                entity_dict[tail] = index
                tail_int = index
                index += 1
            else:
                tail_int = entity_dict[tail]

            ke_list.append([head_int, tail_int])

    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(entity_dict, f)

    # with open('saved_dictionary.pkl', 'rb') as f:
    #     load_entity_dict = pickle.load(f)

    with open('relation.pkl', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(ke_list, filehandle)
    #
    # with open('relation.pkl', 'rb') as filehandle:
    #     # store the data as binary data stream
    #     load_ke_list = pickle.load(filehandle)
