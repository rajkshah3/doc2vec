#!/usr/bin/env python3

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import textacy
import re
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 32)
    pool = Pool(cpu_count() -2)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def main():
    df = pd.read_csv('./tweets.csv',dtype={'tweet':str})
    df = df.dropna()
    df = df.head(4000)
    df = parallelize_dataframe(df,clean_text_tweet)
    l_tweets_ = df['tweet'].str.split(' ').tolist()

    #make index to a dictionary which returns a column and original index
    l_tweets = [TaggedDocument(doc, ['this_{}'.format(i)]) for i, doc in enumerate(l_tweets_)]

    document_vector_size = 20
    word_prediction_window = 6
    iterations = 100 # Need about this many for infer vector to work
    model_d2v = Doc2Vec(l_tweets, 
                        vector_size=document_vector_size,
                         window=word_prediction_window, 
                         min_count=100, 
                         workers=4,
                         dm=1, #Preserves word order in text
                         seed=321123,
                         epochs=iterations,
                         dbow_words=1, #Train words using skipgram simultaneously

                         )
    vecs = model_d2v.docvecs
    print(vecs)
    inferred_vector = model_d2v.infer_vector(l_tweets[0].words)
    top = vecs.most_similar([inferred_vector],topn=3)
    top2 = vecs.most_similar([vecs[0]],topn=3)
    print(top)
    print(top2)
    import pdb; pdb.set_trace()  # breakpoint e008fbf1 //
    model_d2v.save("doc2vec.model")

    
def remove_trailing_hashtags(s) :
    words = s.split(' ')
    if((words[-1]).startswith('#')) :
        words = words[:-1]
        return remove_trailing_hashtags(' '.join(words))
    else :
        return s

def remove_starting_ats(s,ats=[]) :
    words = s.split(' ')
    if((words[0]).startswith('@')) :
        words = words[1:]
        return remove_starting_ats(' '.join(words)) 
    else :
       return s

def remove_trailing_ats(s,ats=[]) :
    words = s.split(' ')
    if((words[-1]).startswith('@')) :
        ats.append(words[-1][1:])
        words = words[:-1]
        return remove_trailing_ats(' '.join(words),ats)
    else :
        return s 

def clean_text_tweet(df):
    df['tweet'] = df['tweet'].apply(clean_text)
    return df

def clean_text(this_row):
    this_row = str(this_row)
    this_row = this_row.replace(r'http\S+', '')
    this_row = remove_starting_ats(this_row)
    this_row = remove_trailing_hashtags(this_row)
    this_row = remove_trailing_ats(this_row)
    this_row = this_row.replace("\n",' ')
    this_row = this_row.replace(u"!",'')
    this_row = this_row.replace('-', ' ')
    this_row = this_row.replace('"','')
    this_row = this_row.replace("'",'')
    this_row = this_row.replace("`",'')
    this_row = this_row.replace("#",'')
    this_row = this_row.replace("@",'')
    this_row = this_row.replace("&",'')
    
    this_row = textacy.preprocess.preprocess_text(this_row,
                                                  fix_unicode=True, no_urls=True, no_emails=True,
                                                  lowercase=True, no_contractions=True,
                                                  no_numbers=False, no_currency_symbols=True, no_punct=True)


    return this_row

if __name__ == '__main__':
    main()