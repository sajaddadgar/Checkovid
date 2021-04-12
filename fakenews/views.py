from django.shortcuts import render
from django.http import HttpResponse

from fakenews.models import Claim
from .apps import FakenewsConfig
import os
import pickle
from .utils import get_all_verdict, final_verdict, get_sentence_verdict, extract_tweets_feature, netword_predict, \
    get_vector, get_top_5_similar, top5_similarities
from django.http import HttpResponseRedirect
from django.urls import reverse


def claim(request):
    text = str(request.POST.get('text'))

    if (text != 'None') and (text != ''):
        naive_bayes_verdict, naive_bayes_label, logistic_regression_verdict, logistic_regression_label, \
        svm_verdict, svm_label, decision_tree_verdict, decision_tree_label, random_forest_verdict, \
        random_forest_label, stacking_verdict, stacking_label, lstm_verdict, lstm_label, \
        bidirectional_lstm_verdict, bidirectional_lstm_label, cnn_verdict, cnn_label, cnn_lstm_verdict, \
        cnn_lstm_label = get_all_verdict(text)

        verdict_dict = {
            'naive_bayes': naive_bayes_verdict,
            'logistic_regression': logistic_regression_verdict,
            'svm': svm_verdict,
            'decision_tree': decision_tree_verdict,
            'random_forest': random_forest_verdict,
            'stacking': stacking_verdict,
            'lstm': lstm_verdict,
            'bidirectional_lstm': bidirectional_lstm_verdict,
            'cnn': cnn_verdict,
            'cnn_lstm': cnn_lstm_verdict
        }

        final_decision, image_form, verdict_percent = final_verdict(verdict_dict)
        stuff = {
            'text': text,
            'show_verdict': True,
            'naive_bayes_label': naive_bayes_label,
            'logistic_regression_label': logistic_regression_label,
            'svm_label': svm_label,
            'decision_tree_label': decision_tree_label,
            'random_forest_label': random_forest_label,
            'stacking_label': stacking_label,
            'lstm_label': lstm_label,
            'bidirectional_lstm_label': bidirectional_lstm_label,
            'cnn_label': cnn_label,
            'cnn_lstm_label': cnn_lstm_label,
            'final_decision': final_decision,
            'image_form': image_form,
            'verdict_percent': verdict_percent
        }

    else:
        stuff = {
            'show_verdict': False,
        }

    return render(request, 'Page-1.html', stuff)


def sentence(request):
    text = str(request.POST.get('text'))
    selected_model = str(request.POST.get('selected_model'))

    if (text != 'None') and (text != ''):
        verdicts = get_sentence_verdict(text.strip(), selected_model)

        stuff = {
            'show_verdict': True,
            'verdicts': verdicts,
            'selected_model': selected_model,
            'verdicts_length': len(verdicts)
        }

    else:
        stuff = {
            'show_verdict': False
        }

    return render(request, 'Page-2.html', stuff)


def tweet(request):
    tweet_url = str(request.POST.get('tweet_url'))
    if (tweet_url != 'None') and (tweet_url != ''):
        tweet_info, lang, text, text_standard, mention_reliable_user = extract_tweets_feature(tweet_url)
        tweet_prediction, text_prediction, label = netword_predict(tweet_info, text)
        tweet_prediction = '{:.2f}'.format(tweet_prediction * 100)
        prediction = [text_prediction, tweet_prediction]
        if lang != 'en':
            correct_lang = True
        else:
            correct_lang = False

        stuff = {
            'show_verdict': True,
            'tweet_info': tweet_info,
            'correct_lang': correct_lang,
            'prediction': prediction,
            'text_standard': text_standard,
            'label': label,
            'text': text,
            'mention_reliable_user': mention_reliable_user
        }
    else:
        stuff = {
            'show_verdict': False
        }

    return render(request, 'Page-3.html', stuff)


def home(request):
    return render(request, 'homepage.html')

def similarity2(request):
    text = str(request.POST.get('text'))

    if (text != 'None') and (text != ''):
        top5 = get_top_5_similar(text)
        stuff = {
            'show_verdict': True,
            'top5': top5
        }
    else:
        stuff = {
            'show_verdict': False,
        }



    return render(request, 'Page-4.html', stuff)


def similarity(request):
    text = str(request.POST.get('text'))
    if (text != 'None') and (text != ''):
        top5 = top5_similarities(text)
        stuff = {
            'show_verdict': True,
            'top5': top5
        }
    else:
        stuff = {
            'show_verdict': False,
        }

    return render(request, 'Page-4.html', stuff)

# def save_data_to_database(request):
#     with open('./models/network_base/tokenizer/data_list.pickle', 'rb') as handle:
#         data_list = pickle.load(handle)
#     print(len(data_list))
#     for text, verdict in data_list:
#         document_to_vector_json = get_vector(text)
#         claim = Claim(text=text, verdict=verdict, vector=document_to_vector_json)
#         claim.save()
#     return render(request, 'Page-4.html')


def error_404(request, exception):
    return render(request, '404.html')

