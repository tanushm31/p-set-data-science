## Importing Relevant Libraries:



```python
import os
import pandas as pd
import json
from tqdm import tqdm
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
from gensim.utils import simple_preprocess
import gensim, spacy
from gensim.models.ldamulticore import LdaMulticore
import re
import numpy as np

from gensim.models import Phrases
from gensim.models.phrases import Phraser
```

## Importing Data:


```python
directory = "D:\Study\Supermind\ds-data"
dfs= []
jsons = []
for root, subdirectories, files in os.walk(directory):
    for file in files:
        filename = os.path.join(root, file)
        f = open(filename)
        data = json.load(f)
        f.close
        jsons.append(data)
```


```python
chats = {
            "Serial":[],
            "Id":[],
            "Text":[]
        }
```


```python
for j in range(0,len(jsons)):
    for i in jsons[j]:
        chats["Serial"].append(j)
        chats["Id"].append(i['id'])
        chats["Text"].append(i['text']['text'])
        
```


```python
df = pd.DataFrame(chats)
```


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Serial</th>
      <th>Id</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34134</th>
      <td>86</td>
      <td>377889</td>
      <td>That's unrelated. The question is what the inc...</td>
    </tr>
    <tr>
      <th>34135</th>
      <td>86</td>
      <td>377888</td>
      <td>If you have to pow on the user side then you'r...</td>
    </tr>
    <tr>
      <th>34136</th>
      <td>86</td>
      <td>377887</td>
      <td>there doesn't appear to be any. though if you ...</td>
    </tr>
    <tr>
      <th>34137</th>
      <td>86</td>
      <td>377886</td>
      <td>What is the incentive to validate txs in Nano?</td>
    </tr>
    <tr>
      <th>34138</th>
      <td>86</td>
      <td>377885</td>
      <td>I remember first hearing about it when it went...</td>
    </tr>
  </tbody>
</table>
</div>



## Using NLTK for Question Detection:


```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/bert-mini-finetune-question-detection")

model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/bert-mini-finetune-question-detection")

```


```python
import nltk
```


```python
nltk.download('nps_chat')
```

    [nltk_data] Downloading package nps_chat to C:\Users\TANUSH
    [nltk_data]     MAHAJAN\AppData\Roaming\nltk_data...
    [nltk_data]   Package nps_chat is already up-to-date!
    




    True




```python
posts = nltk.corpus.nps_chat.xml_posts()[:10000]

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]

# 10% of the total data
size = int(len(featuresets) * 0.1)

# first 10% for test_set to check the accuracy, and rest 90% after the first 10% for training
train_set, test_set = featuresets[size:], featuresets[:size]

# get the classifer from the training set
classifier = nltk.NaiveBayesClassifier.train(train_set)
# to check the accuracy - 0.67
# print(nltk.classify.accuracy(classifier, test_set))

question_types = ["whQuestion","ynQuestion"]
def is_ques_using_nltk(ques):
    question_type = classifier.classify(dialogue_act_features(ques)) 
    return question_type in question_types
```

## Using Sentence Structure To Detect Questions:


```python
question_pattern = ["do i", "do you", "what", "who", "is it", "why","would you", "how","is there",
                    "are there", "is it so", "is this true" ,"to know", "is that true", "are we", "am i", 
                   "question is", "tell me more", "can i", "can we", "tell me", "can you explain",
                   "question","answer", "questions", "answers", "ask"]

helping_verbs = ["is","am","can", "are", "do", "does"]
# check with custom pipeline if still this is a question mark it as a question
def is_question(question):
    question = question.lower().strip()
    if not is_ques_using_nltk(question):
        is_ques = False
        # check if any of pattern exist in sentence
        for pattern in question_pattern:
            is_ques  = pattern in question
            if is_ques:
                break

        # there could be multiple sentences so divide the sentence
        sentence_arr = question.split(".")
        for sentence in sentence_arr:
            if len(sentence.strip()):
                # if question ends with ? or start with any helping verb
                # word_tokenize will strip by default
                first_word = nltk.word_tokenize(sentence)[0]
                if sentence.endswith("?") or first_word in helping_verbs:
                    is_ques = True
                    break
        return is_ques    
    else:
        return True
```


```python
df['is_question'] = ""
```


```python
df['is_question_using_nltk'] = ""
```


```python
len(df)
```




    34139




```python
for i in tqdm(range(len(df)-1)):
    if(df['Text'][i]):
        df.at[i,'is_question'] = str(is_question(df['Text'][i]))
        df.at[i,"is_question_using_nltk"] = str(is_ques_using_nltk(df['Text'][i]))   
    else:
        df.at[i,'is_question'] = "None"
        df.at[i,"is_question_using_nltk"] = "None"   
```

    100%|███████████████████████████████████████████████████████████████████████████| 34138/34138 [00:59<00:00, 572.09it/s]
    


```python
df.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Serial</th>
      <th>Id</th>
      <th>Text</th>
      <th>is_question</th>
      <th>is_question_using_nltk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>377801</td>
      <td>wat is it lmao</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>377800</td>
      <td>No risk here platform will be free</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>377799</td>
      <td>@lordvladin im more confused after I read your...</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>377798</td>
      <td>has anyone sold or bought a house using crypto...</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>377796</td>
      <td>I sent you DM thank you</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>377795</td>
      <td>Can DM me. I'm out rn but will get back by lat...</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>377794</td>
      <td>Dequest is an sbt2 layer on web3 gamification ...</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>377793</td>
      <td>I think dequest focusing on web3 games, as i s...</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>377792</td>
      <td>but feel free to dm me if using rmrk</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>377791</td>
      <td>look into dequest too</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>377790</td>
      <td>Also if anyone can share what is the defects o...</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>377789</td>
      <td>Also i was trying to use remark for NFTs to cr...</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>377788</td>
      <td>Platform is free to use</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>377787</td>
      <td>I can also share the platform here with you if...</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>377786</td>
      <td>I don’t have connections i am working alone</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>377785</td>
      <td>So the system is ready just working on the con...</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>377784</td>
      <td>I'm.messing around but share what you are doing</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>377783</td>
      <td>Dude don't mind me</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>377782</td>
      <td>Hahaha</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>377781</td>
      <td>I am just excited to talk about it</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>377780</td>
      <td>Sorry</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>377779</td>
      <td>Other wise working on platform to reward peopl...</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>377778</td>
      <td>Sir.  Single para. Pls.</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>377777</td>
      <td>Why working on lame, 2d 90th games</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>377776</td>
      <td>Okay single text - what it does.</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>377775</td>
      <td>And i got an idea while working</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>377773</td>
      <td>No it more simple than this</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>377772</td>
      <td>If yes then please speak</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>377771</td>
      <td>Is this “IT” a shitcoin, ponzu or other food f...</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>377770</td>
      <td>Make it quick. 7 lines max.</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# determining the name of the file
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path
file_name = 'Dataset.xlsx'
  
# saving the excel
df.to_excel(uniquify(file_name))
```

## Extracting Keywords From Questions For Tags


```python
PATH = './Dataset_Questions'
df_questions = pd.read_excel (f'{PATH}.xlsx')
```


```python
import yake
```


```python
df_questions["yake"] = ""
kw_extractor = yake.KeywordExtractor()

for i in tqdm(range(len(df_questions))):
#     1111111111

    keywords = kw_extractor.extract_keywords(df_questions["Text"][i])
    ls = []
    for kw in keywords:
        str1 = kw[0].split(" ")
        if(kw[1]>0.05) and len(str1) ==1:
          ls.append(kw[0])
    df_questions.at[i,'yake']  = listToString2(ls)
        
```

## Saving Result to Excel File


```python
# determining the name of the file
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path
file_name = 'Dataset_Questions_With_Keywords.xlsx'
  
# saving the excel
df_questions.to_excel(uniquify(file_name))
```

## Other Aproaches:

We Can use GSDMM for topic modelling and forming groups of Documents Which have similar theme.


```python
def sent_to_words(sentences):
    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# create N-grams
def make_n_grams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    bigrams_text = [bigram_mod[doc] for doc in texts]
    trigrams_text =  [trigram_mod[bigram_mod[doc]] for doc in bigrams_text]
    return trigrams_text
```


```python
tokens_reviews = list(sent_to_words(df["Text"]))
len(tokens_reviews)
```


```python
tokens_reviews = make_n_grams(tokens_reviews)
len(tokens_reviews)
```


```python
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in gensim.parsing.preprocessing.STOPWORDS] for doc in texts]
```


```python
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
```


```python
# do lemmatization keeping only noun, vb, adv ----------> ADD ADJECTIVES MAYBE???????
# because adj is not informative for reviews topic modeling
text_lemmatized = lemmatization(tokens_reviews, allowed_postags=['NOUN', 'VERB'])

# remove stop words after lemmatization
text_lemmatized = remove_stopwords(text_lemmatized)
```


```python
np.random.seed(0)
```


```python
from gsdmm import MovieGroupProcess
```


```python
model_k = 40
model_alpha = 0.2
model_beta = 0.2
model_iters = 50
mgp = MovieGroupProcess(K=model_k, alpha=model_alpha, beta=model_beta, n_iters=model_iters)

vocab = set(x for text in text_lemmatized for x in text)
n_terms = len(vocab)
model = mgp.fit(text_lemmatized, n_terms)
```


```python
def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print("\nCluster %s : %s"%(cluster,sort_dicts))
```


```python
doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :')

for i in range(len(doc_count)):

    print(i,"->",doc_count[i],end = ', ')

# This is for all topics
top_index = doc_count.argsort()[::-1]
print('\nMost important clusters (by number of docs inside):', top_index)

temp=[]
for i in range(len(top_index)):
#     print(top_index[i], end="#")
    if(doc_count[top_index[i]]==0):
        
        break
    temp.append(top_index[i])
#         top_index = np.delete(top_index, i)
top_index=np.array(temp)
print('\nMost important clusters (by number of docs inside) without Zeroes:', top_index)

# show the top 5 words in term frequency for each cluster 
print("show the top 10 words in term frequency for each cluster")
top_words(mgp.cluster_word_distribution, top_index, 10)
```
