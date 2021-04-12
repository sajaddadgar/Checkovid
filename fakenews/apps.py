import os
import pickle
from django.apps import AppConfig
from tensorflow.keras.models import load_model


def get_paragraph_base_models(type):
    path = 'models/' + type + '/'
    ml_model_list = os.listdir(path + 'ml/')
    dl_model_list = os.listdir(path + 'dl/')
    model_dict = {}
    for model in ml_model_list:
        model_name = model.split('.')[0]
        with open(path + 'ml/' + model, 'rb') as file:
            ml_model = pickle.load(file)
        model_dict[model_name] = ml_model

    for model in dl_model_list:
        model_name = model.split('.')[0]
        dl_model = load_model(path + 'dl/' + model)
        model_dict[model_name] = dl_model
    return model_dict


class FakenewsConfig(AppConfig):
    name = 'fakenews'

    paragraph_max_token = 39
    sentence_max_token = 19

    with open('./models/paragraph_base/tokenizer/words.pickle', 'rb') as handle:
        paragraph_base_tokenizer = pickle.load(handle)
    with open('./models/paragraph_base/tokenizer/finalized_tfidfvectorizer.pickle', 'rb') as handle:
        paragraph_base_tfidfvectorizer = pickle.load(handle)

    with open('./models/sentence_base/tokenizer/words.pickle', 'rb') as handle:
        sentence_base_tokenizer = pickle.load(handle)
    with open('./models/sentence_base/tokenizer/finalized_tfidfvectorizer.pickle', 'rb') as handle:
        sentence_base_tfidfvectorizer = pickle.load(handle)

    with open('./models/network_base/tokenizer/labelEncoder.pickle', 'rb') as handle:
        network_base_labelencoder = pickle.load(handle)
    with open('./models/network_base/tokenizer/standardScaler.pickle', 'rb') as handle:
        network_base_standardScaler = pickle.load(handle)

    with open('./models/similarity/cord19_word_embeddings.pickle', 'rb') as handle:
        word_embedding = pickle.load(handle)

    with open('./models/similarity/similarity.pickle', 'rb') as handle:
        similarity = pickle.load(handle)

    network_base_model = load_model('models/network_base/dl/ann.h5')

    paragraph_base_model_dict = get_paragraph_base_models(type='paragraph_base')
    sentence_base_model_dict = get_paragraph_base_models(type='sentence_base')


