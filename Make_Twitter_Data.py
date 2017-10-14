
'''
Date Created: 10/10/2017
Summary: Script to ingest metadata and tweets to create two matrices:
twitterId x demographics
twitterId x words

Current To-dos:
find the csv file with the demographic data
Re-write to only use the test tweets (it is currently designed to do all tweets)
Run and see what happens.
'''


import csv
import nltk
import re
import numpy as np
import cPickle
import scipy

import FeatureSelection
import Vectorize
import Classifiers
import DrawSamples
import FeatureExtractors
import TopicModels
reload(Vectorize)
reload(FeatureSelection)
reload(Classifiers)
reload(DrawSamples)
reload(FeatureExtractors)
reload(TopicModels)

def importMetaDataFile(filename):
    '''
    Grab the data that used to be living in /home/lfriedl/twitterUSVoters/data/twitterDB-matching/match-results/locsFeb3/national2M-rule3.csv
    We want to create a matrix that is twitterID x demographic data (age, sex, income, etc)
    
    We may want to subsample this file, to the 1k IDs Luke downloaded locally.
    
    Input: string of the file and location for the user metadata
    Output: dictionary of {twitterID: [demographic data]}
    '''
    filename = ''
    with open(filename,'r') as f:
        dta=csv.reader(f)
        for i,line in enumerate(dta):
            if i>0:
                twitterId=line[0]
                meta[twitterId]=line[1]

    return meta

def getTweets(twitterid):
    '''
    Function to get the twitter data for an individual twitter ID.
    This function is written to work with Kenny's github example here: https://github.com/kennyjoseph/twitter_dm
    
    Input: string of twitterID
    Output: list of the raw string of all tweets for twitterID
    '''
    from twitter_dm.TwitterUser import TwitterUser

    tweets=[]
    u = TwitterUser()
    u.populate_tweets_from_file(twitterid+'.json')  #Need to figure out of we can use numeric ID (123456789.json) or name (kenny_joseph.json)
    
    for t in u.tweets:
        tweets.append(t.tokens) #not sure if tokens is exactly what we want, we want the raw words, not necessarily tokens. We'll check this.
    # 
    # texts={}
    # source_filename='Datasets/Twitter/members.zip'
    # parser = etree.XMLParser(encoding='utf8',recover=True)
    # with zipfile.ZipFile(source_filename) as zf:
    #     for i,member in enumerate(zf.infolist()):
    #         name=member.filename.split('/')[1].split('.')[0]    #filename is Raw3/name.csv
    #         if idx ==name:
    #             #print idx, name
    #             raw=zf.open(member)
    #             data=csv.reader(raw)
    #             for j,line in enumerate(data):
    #                 if j>0:
    #                     texts[idx+'_'+str(j)]=line[0]
    # if texts=={}:
    #     print 'no tweets for ', idx
        
    return tweets

def Clean_Tweets(text,stopwords='nltk',onlyHashtags=False):
    '''this function tokenizes `tweets` using simple rules:
    tokens are defined as maximal strings of letters, digits
    and apostrophies.
    The optional argument `stopwords` is a list of words to
    exclude from the tokenzation.  This code eliminates hypertext and markup features to
    focus on the substance of tweets'''
    if stopwords=='nltk':
        stopwords=nltk.corpus.stopwords.words('english')
    else:
        stopwords=[]
    #print stopwords
    stopwords+=['bit','fb','ow','twitpic','ly','com','rt','http','tinyurl','nyti','www']
    retweet=False
    if 'RT @' in text:
        retweet=True
    # make lowercase
    text = text.lower()
    #ELEMINATE HYPERLINKS
    ntext=[]
    text=text.split(' ')
    for t in text:
        if t.startswith('http'):
            #ntext.append('hlink')
            continue
        else:
            ntext.append(t)
    text=' '.join(ntext)
    # grab just the words we're interested in
    text = re.findall(r"[\d\w'#@]+",text)
    # remove stopwords
    if onlyHashtags==True:
        htags=[]
        for w in text:
            if w.startswith('#'):
                htags.append(w)
        return htags
    res = []
    for w in text:
        if w=='hlink':
            res.append('HLINK')
            continue
        if w.startswith('@') and w!='@':
            res.append('MNTON'+'_'+w[1:])
            continue
        if w.startswith('#'):
            res.append('HSHTG'+'_'+w[1:])
            continue
        if w not in stopwords:
            res.append(w)
    if retweet:
        res.append('RTWT')
    res=' '.join(res)    
    return res


##########
#Series of Vectorizing Functions and calls
##########

#most frequent K words
def MakeBagOfWords(texts,num_features=5000):
    '''
    Takes the cleaned tweets for each twitterID and creates a tfidf score for the top num_features features
    
    Input:
        texts = dictionary of {twitterID: cleaned_tweets}
        num_features = numeric, number of features to use
        
    Output:
        vec = tfidf-weighted matrix of twitterID x features
        vectorizer = sklearn object to translate any text into vec
        ids = list of twitterIDs for this sample. 
    '''
    #import 'C:\DonorsChoose\GenderText\Vectorize.py'
    #reload(Vectorize)
    print 'vectorizing text for raw scoring'
    #texts=RemoveStops(texts)
    vec,labels,vectorizer=Vectorize.tfidfVectorize(texts,dict([(k,'') for k in texts.keys()]),max_features=num_features)
    ids=texts.keys()
    #dat=scipy.sparse.csr_matrix(Scores)
    
    print "writing Raw scores for  Twitter"
    with open('Twitter/Data/Twitter_Raw_Scores.pkl','wb') as f:
        cPickle.dump(vec,f)
    with open('Twitter/Data/Twitter_Raw_Names.pkl','wb') as f:
        cPickle.dump(vectorizer.get_feature_names(),f)
    with open('Twitter/Data/Twitter_Raw_Ids.pkl','wb') as f:
        cPickle.dump(ids,f)
    return vec, vectorizer

#note this would pull in all of the twitter data as currently written
meta=importMetaDataFile(filename)

texts={}
for twitterid,data in enumerate(meta):
    tweets=getTweets(twitterid)
    texts[twitterid]=Clean_Tweets(tweets)

vec, vectorizer = MakeBagOfWords(texts,num_features=10000)
    