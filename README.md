# CSC591 GRAPH DATA MINING Capstone

## Group members & Contributions
|Name           |unity-ID   |contributions  |
|---------------|-----------|---------------|
|Haoze Du       |hdu5       |fake sentences, Flair NER and RE, Prediction|
|Yuanming Song  |ysong33    |fake sentences, LegalBERT|

## Running Requirements
We use anaconda to manage the requirements. Here the point is only pytorch 1.7.1 is valid to use flair. The installation of pytorch 1.7.1 is:
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
```
And the related packages mentioned in the jupyter notebook are still needed to be installed. For convenience, we will not list all of them at here.
## Source files
|File name          |Description    |
|-------------------|---------------|
|prediction.ipynb   |Prediction     |
|relations.ipynb    |Graph generation   |
|ERPatternToFakeLegalSentence.ipynb| fake sentences|
|PatternToFakeLegalSentence.ipynb|  fake sentences|
|cl_counsel_re_patterns_DO_NOT_DELETE_4_annotation_Afsar_v4.xlsx|   ER data|
|*.xlsx| NER data|
|relation.pkl|KG|
|*.pt| models|
|test.ipynb, RE_Flair.ipynb| Attempts using Legal-BERT and Flair|
|PATTERNS| Generated fake sentences data|

## Description
Our project goal is to build a knowledge graph from the given Law documents, using the NER model and RE model to pull out entities from the legal cases text, using the graph embedding methods to predict links on the knowledge.

We picked node2vec as our Graph Embedding Method. Node2vec is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. The node2vec framework learns low-dimensional representations for nodes in a graph by optimizing a neighborhood preserving objective. The objective is flexible, and the algorithm accomodates for various definitions of network neighborhoods by simulating biased random walks.

### Using Flair pretrained model to extract relations from xlsx files
(Haoze)

So in our final work, we used the pattern from cl_counsel_re_patterns_DO_NOT_DELETE_4_annotation_Afsar_v4.xlsx for training the NER and RE model, instead of doing the tagging stuff (as we attempted to do so and attached in this readme).
In relations.ipynb, we tried to use the pre-trained models (ner-fast, relations-fast) from flair webpage to generate the extraction of relations, and store the head and tail of relations as edges to form a KG and store this KG as relation.pkl. The details are stated in the jupyter notebook.
### Using node2vec and sckit-learn for prediction.
(Haoze)

In prediction.ipynb, we used the stellar graph to form a KG, applied node2vec to generate the random walk word2vec model, and used sckit-learn to train the best model. ROC_AUC is used as metric to evaluate the performance of these models. The details are stated in the jupyter notebook.

## Attempts

### Generate Fake Sentence using NER and RE
Here these 2 jupyter notebooks are used to generate the fake sentence: ERPatternToFakeLegalSentence.ipynb(NER), PatternToFakeLegalSentence.ipynb(RE)

### Train our NER model using legalBERT
(Yuanming)

Our team first trained our NER model using the fake legal sentences based on the professor's code. The sentences generated are in IOB format(CONLL format, every token in a separate line, O represent other, B-begin I-inside E-end of the entity). We use the data to create our own corpus for model training, validation, and testing.

```python
# The sentence objects holds a sentence that we may want to embed or tag
# Created flair's column corpus
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.datasets.relation_extraction import CoNLLUCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
```

As Dr. Samatova introduced in lecture, the fake sentence generator, patterns, dictionaries. For NER BERT and Flair Model, it allows us to stack Legal BERT Word Embeddings for Sequence Modeling. Therefore, we import Word Embedding and Legal-BERT as pre-trained model.

```python
# From 10/13 lecture and announcement
# Legal BERT Word Embeddings: Stacked
PATH_TO_EMBEDDINGS = Path("./LegalBERTEmbeddings")
embedding_types: List[TokenEmbeddings] = [
    FlairEmbeddings(os.path.join(PATH_TO_EMBEDDINGS, 'Fullset_Forward', 'best-lm.pt'), chars_per_chunk=64),
    FlairEmbeddings(os.path.join(PATH_TO_EMBEDDINGS, 'Fullset_Backward', 'best-lm.pt'), chars_per_chunk=64),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# CRF and LSTM NER Sequence Taggers: Stacked
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=TAG_TYPE,                                 
                                        #CRF   
                                        use_crf=True,
                                        use_rnn=True,
                                        rnn_layers=1,                                 
                                        #LSTM  
                                        rnn_type='LSTM',
                                        word_dropout=0.05,
                                        locked_dropout=0.5,
                                        train_initial_hidden_state=False)
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# CPU or GPU Model Trainer
MODEL_PATH = Path('./TrainedNERModel/')
trainer.train(MODEL_PATH,
              learning_rate=0.1,
              mini_batch_size=16,
              max_epochs=1,
              monitor_train=False,
              monitor_test=False,
              checkpoint=True,
              train_with_dev=True,
              embeddings_storage_mode='cpu')
```

Our team use legal-bert-base-uncased from Hugging Face as our Word Embedding Source, English legal text from several fields (e.g., legislation, court cases, contracts) scraped from publicly available resources, and uses the original pre-trained BERT configuration.
 
Flair allows us to apply our state-of-the-art natural language processing (NLP) models to your text. Flair has simple interfaces that allow us to use and combine different word and document embeddings. And for now, most Flair sequence tagging models (named entity recognition, part-of-speech tagging etc.) are now hosted on the HuggingFace model hub.

We also made changes to flair's source code, conllu.py file, based on professor's code:
```python
def parse_relation_tuple_list(key: str,
                              value: Optional[str] = None,
                              list_sep: str = "|",
                              value_sep: str = ";") -> Optional[Tuple[str, List[Tuple[int, int, int, int, str]]]]:
    if value is None:
        return value


    relation_tuples: List[Tuple[int, int, int, int, str]] = []
    for relation in value.split(list_sep):
        if len(relation.split(value_sep)) == 5: # NAGIZA: ADDED: if there are no relations
            head_start, head_end, tail_start, tail_end, label = relation.split(value_sep)
            relation_tuples.append((int(head_start), int(head_end), int(tail_start), int(tail_end), label))

    return key, relation_tuples
```
Those implementations were attempted and stored as RE_Flair.ipynb and test.ipynb.

## Reference

    https://github.com/flairNLP/flair
    https://github.com/flairNLP/flair/releases
    https://huggingface.co/nlpaueb/legal-bert-base-uncased
    http://nlpprogress.com/english/relationship_extraction.html
    https://medium.com/thecyphy/training-custom-ner-model-using-flair-df1f9ea9c762
    https://pytorch.org/get-started/previous-versions/
    https://stellargraph.readthedocs.io/en/v0.8.3/quickstart.html
    https://github.com/eliorc/node2vec


