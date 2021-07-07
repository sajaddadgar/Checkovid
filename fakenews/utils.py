import re
import string
from datetime import datetime
import numpy as np
import heapq
import textstat
# from google_trans_new import google_translator
from .google_trans import google_translator
import operator
from fakenews.models import Claim
from .twitter import get_tweets_by_id, get_user_by_id
import spacy
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.tag import pos_tag
from collections import Counter
from .apps import FakenewsConfig
from lexicalrichness import LexicalRichness
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot, hashing_trick, text_to_word_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from json import JSONEncoder
import numpy


# nltk.download('stopwords')
# nltk.download('punkt')

reliable_users = ['HelenBranswell', 'mlipsitch', 'WHO', 'JeremyFarrar', 'trvrb', 'MackayIM', 'kakape', 'DrTedros', 'cmyeaton', 'CDCDirector', 'T_Inglesby', 'MarionKoopmans', 'edyong209', 'AdamJKucharski', 'neil_ferguson', 'Laurie_Garrett', 'aetiology', 'maiamajumder', 'richardhorton1', 'CEPIvaccines', 'statnews', 'ChristoPhraser', 'arambaut', 'amymaxmen', 'BarackObama', 'marynmck', 'sciencecohen', 'jennifergardy', 'onisillos', 'PeterHotez', 'Chikwe_I', 'DrTomFrieden', 'juliaoftoronto', 'alexandraphelan', 'SCBriand', 'CDCgov', 'mvankerkhove', 'gmleunghku', 'GaviSeth', 'DrMikeRyan', 'michaelmina_lab', 'martinenserink', 'ScottGottliebMD', 'angie_rasmussen', 'CT_Bergstrom', 'CIDRAP', 'simonihay', 'BillGates', 'gatesfoundation', 'gregggonsalves', 'TheLancetInfDis', 'johnbrownstein', 'trevormundel', 'NEJM', 'TheLancet', 'carlzimmer', 'florian_krammer', 'MRC_Outbreak', 'RebeccaKatz5', 'EricTopol', 'devisridhar', 'BogochIsaac', 'nataliexdean', 'BillHanage', 'PeterDaszak', 'LawrenceGostin', 'Atul_Gawande', 'AfricaCDC', 'ProMED_mail', 'CDCGlobal', 'chngin_the_wrld', 'ScienceMagazine', 'wellcometrust', 'gavi', 'JeremyKonyndyk', 'doctorsoumya', 'CarlosdelRio7', 'paimadhu', 'BhadeliaMD', 'edwardcholmes', 'NatureNews', 'LSHTM', 'pathogenomenick', 'JustinLessler', 'lmadoff', 'NIHDirector', 'mugecevik', 'AmeshAA', 'SRileyIDD', 'melindagates', 'healthmap', 'PeterASinger', 'JenniferNuzzo', 'joshmich', 'V2019N', 'phylogenomics', 'ECDC_EU', 'LancetGH', 'InfectiousDz', 'nature', 'c_drosten', 'GlobalFund', 'IlonaKickbusch', 'K_G_Andersen', 'CCDD_HSPH', 'bmj_latest', 'JAMA_current', 'HarvardChanSPH', 'PLOS', 'ashishkjha', 'NPRGoatsandSoda', 'C_Althaus', 'IDEpiPhD', 'OWMorgan', 'WHOAFRO', 'DFisman', 'PardisSabeti', 'Crof', 'KindrachukJason', 'NathanGrubaugh', 'nicolamlow', 'nytimes', 'NickKristof', 'USAIDGH', 'RonaldKlain', 'PATHtweets', 'EpiEllie', 'PLOSPathogens', 'firefoxx66', 'marcelsalathe', 'ASTMH', 'MOUGK', 'dylanbgeorge', 'picardonhealth', 'PLOSMedicine', 'FluTrackers', 'MoetiTshidi', 'DreJoanneLiu', 'JNkengasong', 'yhgrad', 'TheEconomist', 'CDCemergency', 'larrybrilliant', 'profvrr', 'DrNancyM_CDC', 'MSFsci', 'IsabelOtt', 'SaadOmer3', 'EvolveDotZoo', 'TeebzR', 'TAlexPerkins', 'sdwfrost', 'gateshealth', 'LizSzabo', 'SueDHellmann', 'VirusesImmunity', 'mbeisen', 'alexvespi', 'icddr_b', 'glassmanamanda', 'bylenasun', 'SaskiaPopescu', 'KarenGrepin', 'OutbreakJake', 'EpsteinJon', 'jw132', 'DrJayVarma', 'Declan_M_Butler', 'NateSilver538', 'pahowho', 'DoctorYasmin', 'UKAMREnvoy', 'Pezzapezzi', 'RSTMH', 'curefinder', 'gabbystern', 'ChrisJElias', 'EdWhiting1', 'helleringer143', 'megan_b_murray', 'bansallab', 'mpkieny', 'UN', 'UNICEF', 'NIH', 'JeffDSachs', 'MicrobesInfect', 'maggiemfox', 'ghn_news', 'KrutikaKuppalli', 'EckerleIsabella', 'DrJudyStone', 'richardneher', 'eliowa', 'OrinLevine', 'PeterASands', 'bencowling88', 'rozeggo', 'soniashah', 'sarahcobey', 'rd_blueprint', 'BBCBreaking', 'HillaryClinton', 'washingtonpost', 'ChelseaClinton', 'NPRHealth', 'CDCFlu', 'JohnsHopkinsSPH', 'HansRosling', 'NAChristakis', 'womeninGH', 'thelonevirologi', 'BethCameron_DC', 'GlobalBioD', 'cmmid_lshtm', 'EIDGeek', 'igoodfel', 'SunKaiyuan', 'CIDIDteam', 'nmrfaria', 'betzhallo', 'antonioguterres', 'PHE_uk', 'MSF', 'JHSPH_CHS', 'IHME_UW', 'globalhlthtwit', 'HarvardGH', 'VignuzziLab', 'clarewenham', 'epi_michael', 'AOC', 'wef', 'NCDCgov', 'WHOWPRO', 'WHO_Europe', 'syramadad', 'sethmnookin', 'francetim', 'cbpolis', 'DavidQuammen', 'Eurosurveillanc', 'ProfMattFox', 'OlyIlunga', 'stefanswartpet', 'HartlGA', 'petrakle', 'Outbreaks101', 'MorrisonCSIS', 'gail_carson', 'jd_mathbio', 'PFormenty', 'DanielBausch2', 'BBCWorld', 'NYTHealth', 'USAID', 'ASlavitt', 'PIH', 'CMO_England', 'CGDev', 'Craig_A_Spencer', 'matthewherper', 'DrSenait', 'DrLeanaWen', 'GHS', 'DrRichBesser', 'IDSAInfo', 'VirusWhisperer', 'martinmckee', 'davidnabarro', 'sherifink', 'ECDC_Outbreaks', 'RELenski', 'PLOSNTDs', 'CUGHnews', 'EcoHealthNYC', 'arimoin', 'LaurenWeberHP', 'CHGlobalHealth', 'reichlab', 'jocalynclark', 'DokteCoffee', 'svscarpino', '_b_meyer', 'jbloom_lab', 'GermsAndNumbers', 'joel_mossong', 'pam_das', 'lisaschnirring', 'MMKavanagh', 'wtaylor1', 'jLewnard', 'datcummings', 'sciam', 'MaxCRoser', 'laurahelmuth', 'Fogarty_NIH', 'CDDEP', 'FLAHAULT', 'betswrites', 'asoucat', 'AniShakari', 'greg_folkers', 'TomBollyky', 'GYamey', 'jenkatesdc', 'jmayer0716', 'ppenttin', 'sbfnk', 'guardian', 'GretaThunberg', 'newscientist', 'WorldBank', 'guardianscience', 'AcademicsSay', 'royalsociety', 'MicrobiomDigest', 'KulikovUNIATF', 'CDC_NCEZID', 'JATetro', 'baym', 'Wellcome_AMR', 'MartenRobert', 'MCBazacoPhD', 'ZaminIqbal', 'FINDdx', 'DrMartinCDC', 'twpiggott', 'AshTuite', 'neva9257', 'TheMenacheryLab', 'isabelrodbar', 'hayesluk', 'nycbat', 'CJEMetcalf', 'rreithinger', 'kelle569', 'AndyTatem', 'MichelleObama', 'bbchealth', 'WIREDScience', 'MSF_USA', 'PublicHealth', 'CFR_org', 'DFID_UK', 'NatureMedicine', 'choo_ek', 'CDCMMWR', 'CNN', 'DHSCgovuk']

translator = google_translator()
nlp = spacy.load("en_core_web_sm")
embedding = FakenewsConfig.word_embedding
stopwords_english = stopwords.words('english')
power_word = ['improve', 'trust', 'immediately', 'discover', 'profit', 'learn', 'know', 'understand', 'powerful',
              'best', 'win', 'more', 'bonus', 'exclusive', 'extra', 'you', 'free', 'health', 'guarantee', 'new',
              'proven', 'safety', 'money', 'now', 'today', 'results', 'protect', 'help', 'easy', 'amazing', 'latest',
              'extraordinary', 'how to', 'worst', 'ultimate', 'hot', 'first', 'big', 'anniversary', 'premiere', 'basic',
              'complete', 'save', 'plus', 'create']


def one_hot_text(token2id, input_text):
    return [token2id[word] for word in word_tokenize(input_text) if word in token2id]


# def preprocessing(text):
#     stemmer = PorterStemmer()
#     stopwords_english = stopwords.words('english')
#     stopwords_english.remove('not')
#     text = re.sub(r'$', '', str(text))
#     text = re.sub(r'https?:\/\/.*[\r\n]*', '', str(text))
#     text = re.sub(r'^RT[\s]+', '', str(text))
#     text = re.sub(r'#', '', str(text))
#     text = re.sub(r'\@\w*', '', str(text))
#     text = re.sub(r'WHO', 'world health organization', str(text))
#     text = re.sub(r"&", ' and ', str(text))
#     text = text.replace('&amp', ' ')
#     text = re.sub(r"[^0-9a-zA-Z]+", ' ', str(text))
#     tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
#     text_tokens = tokenizer.tokenize(text)
#     clean_text = []
#     for word in text_tokens:
#         if (word not in stopwords_english and word not in string.punctuation):
#             stem_word = stemmer.stem(word)
#             clean_text.append(stem_word)
#     return ' '.join(clean_text)

def preprocessing(text):
    negate_dict = {
        "couldn't": "could not",
        "didn't": "did not",
        "won't": "will not",
        "don't": "do not",
        "aren't": "are not",
        "doesn't": "does not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "isn't": "is not",
        "mightn't": "might not",
        "mustn't": "must not",
        "needn't": "need not",
        "shan't": "shall not",
        "shouldn't": "should not",
        "wasn't": "was not",
        "weren't": "were not",
        "wouldn't": "would not"
    }
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    stopwords_english.remove('not')
    text = re.sub(r'$', '', str(text))
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', str(text))
    text = re.sub(r'^RT[\s]+', '', str(text))
    text = re.sub(r'#', '', str(text))
    text = re.sub(r'\@\w*', '', str(text))
    text = re.sub(r'WHO', 'world health organization', str(text))
    for negate_word in negate_dict.keys():
        text = re.sub(negate_word, negate_dict[negate_word], str(text))
    text = re.sub(r"&", ' and ', str(text))
    text = text.replace('&amp', ' ')
    text = re.sub(r"[^0-9a-zA-Z]+", ' ', str(text))
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    clean_text = []
    for word in text_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            clean_text.append(stem_word)
    return ' '.join(clean_text)


def cleantext(string):
    stops = set(stopwords.words("english"))
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+", ' ', text)
    text = re.sub(r"www(\S)+", ' ', text)
    text = re.sub(r"&", ' and ', text)
    text = text.replace('&amp', ' ')
    text = re.sub(r"[^0-9a-zA-Z]+", ' ', text)
    text = text.split()
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def preprocessing_claim(text):
    verdict_list = ['fake', 'true', 'mostly false', 'mostly true', 'false', 'partly false', 'partly true']
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    text = re.sub(r'$', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\@\w*', '', text)
    text = re.sub(r'WHO', 'world health organization', text)
    text = re.sub(r"&", ' and ', text)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    clean_text = []
    for word in text_tokens:
        if ((word not in stopwords_english) and (word not in string.punctuation) and (word not in verdict_list)):
            stem_word = stemmer.stem(word)
            clean_text.append(stem_word)
    return ' '.join(clean_text)


def get_verdict_with_token2id(model, token2id, input_text, maxlen):
    input_text_preproc = preprocessing(input_text)
    input_text_onehot = one_hot_text(token2id, input_text_preproc)
    input_text_embedded_docs = pad_sequences([input_text_onehot], padding='pre', maxlen=maxlen)
    output_verdict = model.predict(input_text_embedded_docs)
    return output_verdict[0][0]


def get_all_verdict(text):

    lang = translator.detect(text)[0]
    if lang != 'en':
        text = translator.translate(text, lang_tgt='en')

    models = FakenewsConfig.paragraph_base_model_dict
    tfidfvectorizer = FakenewsConfig.paragraph_base_tfidfvectorizer
    tokenizer = FakenewsConfig.paragraph_base_tokenizer
    max_token = FakenewsConfig.paragraph_max_token

    naive_bayes_verdict, naive_bayes_label = get_ml_verdict(model=models['naive_bayes'],
                                                            countvectorizer=tfidfvectorizer, input_text=text)
    logistic_regression_verdict, logistic_regression_label = get_ml_verdict(model=models['logistic_regression'],
                                                                            countvectorizer=tfidfvectorizer,
                                                                            input_text=text)
    svm_verdict, svm_label = get_ml_verdict(model=models['svm'], countvectorizer=tfidfvectorizer, input_text=text)
    decision_tree_verdict, decision_tree_label = get_ml_verdict(model=models['decision_tree'],
                                                                countvectorizer=tfidfvectorizer, input_text=text)
    random_forest_verdict, random_forest_label = get_ml_verdict(model=models['random_forest'],
                                                                countvectorizer=tfidfvectorizer, input_text=text)
    stacking_verdict, stacking_label = get_ml_verdict(model=models['stacking'], countvectorizer=tfidfvectorizer,
                                                      input_text=text)

    lstm_verdict, lstm_label = get_dl_verdict(model=models['lstm'], tokenizer=tokenizer, input_text=text,
                                              maxlen=max_token)
    bidirectional_lstm_verdict, bidirectional_lstm_label = get_dl_verdict(model=models['bidirectional_lstm'],
                                                                          tokenizer=tokenizer, input_text=text,
                                                                          maxlen=max_token)
    cnn_verdict, cnn_label = get_dl_verdict(model=models['cnn'], tokenizer=tokenizer, input_text=text, maxlen=max_token)
    cnn_lstm_verdict, cnn_lstm_label = get_dl_verdict(model=models['cnn_lstm'], tokenizer=tokenizer, input_text=text,
                                                      maxlen=max_token)

    return naive_bayes_verdict, naive_bayes_label, logistic_regression_verdict, logistic_regression_label, svm_verdict, svm_label, decision_tree_verdict, decision_tree_label, random_forest_verdict, random_forest_label, stacking_verdict, stacking_label, lstm_verdict, lstm_label, bidirectional_lstm_verdict, bidirectional_lstm_label, cnn_verdict, cnn_label, cnn_lstm_verdict, cnn_lstm_label


def check_model_type(modelname):
    ml_list = ['naive_bayes', 'logistic_regression', 'svm', 'decision_tree', 'random_forest', 'stacking']
    dl_list = ['lstm', 'bidirectional_lstm', 'cnn', 'cnn_lstm']
    if modelname in ml_list:
        return 'ml'
    elif modelname in dl_list:
        return 'dl'


def get_dl_verdict(model, tokenizer, input_text, maxlen):
    input_text_preproc = preprocessing(input_text)
    onehot_train = hashing_trick(input_text_preproc, round(len(tokenizer) * 1.3), hash_function='md5')
    input_text_embedded_docs = pad_sequences([onehot_train], padding='pre', maxlen=maxlen)
    output_verdict = model.predict(input_text_embedded_docs)
    final_verdict = output_verdict[0][0]
    if final_verdict >= 0.5:
        label = 'real'
    else:
        label = 'fake'
    return final_verdict, label


def get_ml_verdict(model, countvectorizer, input_text):
    docs_new = [input_text]
    X_new_counts = countvectorizer.transform([preprocessing(i) for i in docs_new])
    probability = list(model.predict_proba(X_new_counts)[0])
    final = probability.index(max(probability))

    if final == 1:
        label = 'real'
    else:
        label = 'fake'
        probability[final] = 1 - probability[final]
    return probability[final], label


def final_verdict(verdict_dict):
    acc_dict = {
        'naive_bayes': 0.8834162520729685,
        'logistic_regression': 0.8986733001658375,
        'svm': 0.8913764510779436,
        'decision_tree': 0.8490878938640133,
        'random_forest': 0.8980099502487562,
        'stacking': 0.9102819237147596,
        'lstm': 0.9038142620232172,
        'bidirectional_lstm': 0.9009950248756219,
        'cnn': 0.9087893864013267,
        'cnn_lstm': 0.8933665008291874
    }

    verdict_sum = sum(acc_dict.values())

    result = 0
    for model in acc_dict:
        result += (acc_dict[model] / verdict_sum) * verdict_dict[model]

    result_decimal = '{:.2f}'.format(result * 100)
    if (result >= 0.9):
        return 'We are certain that this is real news.', 'real', result_decimal
    elif (result >= 0.75):
        return 'We are almost certain that this news is real.', 'real', result_decimal
    elif (result >= 0.6):
        return 'We think that this is real news.', 'real', result_decimal
    elif (result >= 0.4):
        return 'we are not sure that this news is fake or real.', 'real', result_decimal
    elif (result >= 0.25):
        return 'We think that this is fake news.', 'fake', result_decimal
    elif (result >= 0.1):
        return 'We are almost certain that this news is fake', 'fake', result_decimal
    else:
        return 'We are certain that this is fake news.', 'fake', result_decimal


def get_sentence_verdict(text, selected_model):
    models = FakenewsConfig.sentence_base_model_dict
    tfidfvectorizer = FakenewsConfig.sentence_base_tfidfvectorizer
    tokenizer = FakenewsConfig.sentence_base_tokenizer
    max_token = FakenewsConfig.sentence_max_token

    model_name = {
        'Naive Bayes': 'naive_bayes',
        'Logistic Regression': 'logistic_regression',
        'Support Vector Machine (SVM)': 'svm',
        'Decision Tree': 'decision_tree',
        'Random Forest': 'random_forest',
        'Stacking': 'stacking',
        'Long short-term memory (LSTM)': 'lstm',
        'Bidirectional LSTM': 'bidirectional_lstm',
        'Convolutional Neural Networks (CNN)': 'cnn',
        'CNN + LSTM': 'cnn_lstm'
    }

    lang = translator.detect(text)[0]
    if lang != 'en':
        text = translator.translate(text, lang_tgt='en')

    doc = nlp(text)
    sentence = [str(sent).strip() for sent in doc.sents]

    model_type = check_model_type(model_name[selected_model])

    verdicts = []
    if model_type == 'ml':
        for sent in sentence:
            verdict, label = get_ml_verdict(models[model_name[selected_model]], tfidfvectorizer, sent)
            verdicts.append((sent, '{:.1f}'.format(verdict * 100), label, verdict))
    elif model_type == 'dl':
        for sent in sentence:
            verdict, label = get_dl_verdict(models[model_name[selected_model]], tokenizer, sent, max_token)
            verdicts.append((sent, '{:.1f}'.format(verdict * 100), label, verdict))
    return verdicts


def month_to_num(month):
    month = month.lower()
    num = ''
    if month == 'jan':
        num = '01'
    elif month == 'feb':
        num = '02'
    elif month == 'mar':
        num = '03'
    elif month == 'apr':
        num = '04'
    elif month == 'may':
        num = '05'
    elif month == 'jun':
        num = '06'
    elif month == 'jul':
        num = '07'
    elif month == 'aug':
        num = '08'
    elif month == 'sep':
        num = '09'
    elif month == 'oct':
        num = '10'
    elif month == 'nov':
        num = '11'
    elif month == 'dec':
        num = '12'
    return num


def join_punctuation(seq, characters='.,;?!'):
    characters = set(characters)

    try:
        seq = iter(seq)
        current = next(seq)
    except StopIteration:
        return

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt
    yield current


def preprocessing_network(text):
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', str(text))
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'\@\w*', '', text)
    text = re.sub(r'#', '', text)
    tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    return ' '.join(join_punctuation(text_tokens))


def extract_datetime(t):
    t = str(t)
    year = t[-4:]
    month = month_to_num(t[4:7])
    day = t[8:10]
    date = year + '-' + month + '-' + day
    return datetime.strptime(date, "%Y-%m-%d").timestamp() * 1000


def calculate_tweet_features(tweet_json):
    tweet_info = {}

    text = tweet_json['full_text']

    aux_features = {}
    punctuation_count = len([a for a in str(text) if a in string.punctuation])
    aux_features['punctuation_percent'] = punctuation_count / len(text)



    if tweet_json['lang'] != 'en':
        text = translator.translate(text, lang_tgt='en')

    preproc_text = preprocessing_network(text)
    tweet_token = word_tokenize(preproc_text)

    # Calculate the rate of stop word in the text
    stop_word_count = len([word for word in tweet_token if word.strip() in stopwords_english])
    tweet_info['stop_words_percent'] = stop_word_count / len(str(preproc_text).split())

    # Calculate noun count in the text
    tag_num = Counter([tag[1] for tag in pos_tag(tweet_token)])
    tweet_info['noun_count'] = tag_num.get('NN', 0) + tag_num.get('NNS', 0)

    # Calculate type token ratio of the text
    tweet_info['type_token_ratio'] = LexicalRichness(text).ttr

    # Calculate verb count in the text
    tweet_info['verb_count'] = tag_num.get('VBZ', 0) + tag_num.get('VBG', 0) + tag_num.get('VBP', 0) + tag_num.get(
        'VBN', 0) + tag_num.get('VBD', 0) + tag_num.get('VB', 0)

    # Calculate text standard
    text_standard = textstat.textstat.text_standard(text)
    tweet_info['text_standard'] = FakenewsConfig.network_base_labelencoder.transform([text_standard])[0]

    # Check if the user have url in his or her tweets or not
    tweet_info['has_url'] = int(len(tweet_json['entities']['urls']) > 0)

    # Calculate adjective count in the text
    tweet_info['adjective_count'] = tag_num.get('JJ', 0)

    # Calculate coleman liau index
    tweet_info['coleman_liau_index'] = textstat.textstat.coleman_liau_index(text)

    # Calculate dale chall readability score
    tweet_info['dale_chall_readability_score'] = textstat.textstat.dale_chall_readability_score(text)

    # Calculate pronoun count in the text
    tweet_info['pronoun_count'] = tag_num.get('PRP', 0)

    # Calculate flesch reading ease
    tweet_info['flesch_reading_ease'] = textstat.textstat.flesch_reading_ease(text)

    # Calculate proper noun count in the text
    tweet_info['proper_noun_count'] = tag_num.get('NNP', 0) + tag_num.get('NNPS', 0)

    # Calculate the rate of capital character in the text
    capital_char_count = sum(1 for c in preproc_text if c.isupper())
    tweet_info['capital_words_percent'] = capital_char_count / len(preproc_text)

    # Calculate the rate of power word in the text
    power_word_count = sum(1 for c in word_tokenize(text) if c in power_word)
    tweet_info['power_words_percent'] = power_word_count / len(str(preproc_text).split())

    # Calculate automated readability index
    tweet_info['automated_readability_index'] = textstat.textstat.automated_readability_index(text)

    # Calculate number of the hashtags in the text
    tweet_info['hashtag_count'] = len(tweet_json['entities']['hashtags'])

    # Calculate smog index
    tweet_info['smog_index'] = textstat.textstat.smog_index(text)

    # Calculate user's tweet count
    user_json = get_user_by_id(tweet_json['user']['id_str'])
    tweet_info['user_tweet_count'] = user_json['statuses_count']

    # Calculate number of the mentions in the text
    tweet_info['mention_count'] = len(tweet_json['entities']['user_mentions'])

    # Calculate retweet count
    tweet_info['retweet_count'] = tweet_json['retweet_count']

    # Calculate flesch kincaid grade
    tweet_info['flesch_kincaid_grade'] = textstat.textstat.flesch_kincaid_grade(text)

    # Calculate flesch kincaid grade
    tweet_info['gunning_fog'] = textstat.textstat.gunning_fog(text)

    mention_list = tweet_json['entities']['user_mentions']
    mention_reliable_user = 0
    for mention in mention_list:
        if mention['screen_name'] in reliable_users:
            mention_reliable_user = 1
            break

    aux_features['mention_reliable_user'] = mention_reliable_user
    aux_features['like_count'] = tweet_json['favorite_count']
    aux_features['followers_count'] = tweet_json['user']['followers_count']
    aux_features['following_count'] = tweet_json['user']['friends_count']
    aux_features['sentiment'] = TextBlob(str(text)).polarity


    return tweet_info, text_standard, aux_features


def predict_tweet_text(text):
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

    _, _, prediction = final_verdict(verdict_dict)
    return prediction


def extract_tweets_feature(tweet_url):
    tweet_id = str(tweet_url).split('status/')[1]
    tweet_json = get_tweets_by_id(tweet_id)
    tweet_info, text_standard, aux_features = calculate_tweet_features(tweet_json)
    return tweet_info, tweet_json['lang'], tweet_json['full_text'], text_standard, aux_features


def netword_predict(tweet_info, text):
    ann = FakenewsConfig.network_base_model
    standardScaler = FakenewsConfig.network_base_standardScaler

    sample = list(tweet_info.values())
    sample = standardScaler.transform([sample])

    network_prediction = ann.predict(sample)[0][0]
    text_prediction = predict_tweet_text(text)

    if network_prediction >= 0.5:
        label = 'real'
    else:
        label = 'false'
    return network_prediction, text_prediction, label

def save_db(sentence_text, sentence_verdict, user_comment):
    with open('./data/comments.csv', 'a') as f:
        f.write(','.join([sentence_text, sentence_verdict, user_comment]) + '\n')


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_vector(document):
    document_to_vector = get_doc_vector(document)
    vector_json = json.dumps({"embedding": document_to_vector}, cls=NumpyArrayEncoder)
    return vector_json


def get_doc_vector(document):
    vector_document = [embedding[x] for x in word_tokenize(document) if x in embedding.vocab]
    document_to_vector = np.sum(vector_document, axis=0)
    return document_to_vector


def cosine_similarity(A, B):
    np.seterr(divide='ignore', invalid='ignore')
    dot = np.dot(A, B)
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    cos = dot / (normA * normB)
    return cos

def compute_similarity(target, claims):
    similarity_list = []
    for claim in claims:
        claim_vector = claim.vector
        vector_json = json.loads(claim_vector)
        vector_embedding = np.array(vector_json['embedding'])
        score = cosine_similarity(target, vector_embedding)
        if type(score) != type(np.array([])):
            similarity_list.append((score, claim.text, claim.verdict))
    similarity_list.sort(reverse=True, key=operator.itemgetter(0))
    return similarity_list


def get_top_5_similar(document):
    target_doc = get_doc_vector(document)
    claims = Claim.objects.all()
    all_similarity = compute_similarity(target_doc, claims)
    return all_similarity[:5]


def top5_similarities(text):

    lang = translator.detect(text)[0]
    if lang != 'en':
        text = translator.translate(text, lang_tgt='en')

    X = FakenewsConfig.similarity
    X2 = np.append(X, [text])

    tfidf = TfidfVectorizer().fit_transform(X2)
    pairwise_similarity = tfidf * tfidf.T
    arr = pairwise_similarity.toarray()
    np.fill_diagonal(arr, np.nan)

    input_idx = list(X2).index(text)
    max_similarity = heapq.nlargest(5, range(len(arr[input_idx])), arr[input_idx].take)
    top5_verdict = [Claim.objects.get(id=i+1).verdict for i in max_similarity]
    top5_text = [X2[i] for i in max_similarity]

    return list(zip(top5_text, top5_verdict))
