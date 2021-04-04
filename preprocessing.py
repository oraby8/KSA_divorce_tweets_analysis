import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import re
import joblib
loaded_model = joblib.load('twitter_sentimntal.sav')


data=pd.read_csv('tweets_data.csv')


Category={'Violence':['العنف','عنف','ضرب','ايذاء','تحرش'],'financial':['وسفر','قصر','مطاعم','ترفيه'],'responsibility':['واجبهما','مهام','الواجبات','مسءولية'],"second marriage":['تعدد الزواج','الزواج ثانيه'],'not match':['عدم الحوار','عدم المشاركه','الغموض','عدم تفاهمهم'],"women work":['عمل الزوجة','عمل المرءة','الدوام']
,"external reason":['لتاثر بالاصدقاء','تدخلات الاهل']}

Alis_cat={'man':['الزوج','الرجل','الفتي','الولد'],
         'woman':['الزوجه','المرأه','الفتاه','البنت']}

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def preprocessing(tweet):
    ''' Delete not arabic words and punc and digits and arabic stopwords'''
    stopwords_ar=stopwords.words('arabic')
    tweet=''.join([x for t in tweet for x in t if x not in string.punctuation])
    tweet=''.join([x for t in tweet for x in t if not x.isdigit()])
    tweet=''.join([x for t in tweet for x in t if x not in string.ascii_letters])
    tweet=' '.join([x for x in tweet.split() if x not in stopwords_ar])
    return normalize_arabic(tweet)

def wordscounter(sent,Category):
    counter=[]
    for i in Category:
        for ii in Category[i]: 
            if ii in sent:
                counter.append(i)
    return counter

data['Tweets']=data['Tweets'].apply(lambda x:preprocessing(x))
data['Tweets']=data['Tweets'].apply(lambda x:normalize_arabic(x))
cat=[]
for i in data['Tweets'].values:
    out=wordscounter(i,Category)
    if out !=[]:
        cat.append(out[0])
    else:
        cat.append(None)
ana=[]
for i in data['Tweets'].values:
    out=wordscounter(i,Alis_cat)
    if out !=[]:
        ana.append(out[0])
    else:
        ana.append(None)

data['Alies']=ana
data['Category']=cat
data['sentmintal']=data['Tweets'].apply(lambda x:loaded_model.predict([x])[0])
data.to_csv('enhancemnt_data.csv')
