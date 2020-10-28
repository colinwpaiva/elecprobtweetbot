#!/usr/bin/env python
# coding: utf-8

# In[356]:


import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import tweepy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rcParams
rcParams['font.family'] = 'Helvetica'
rcParams.update({'font.size': 15})
import tweepy


# In[357]:


def get_probs():
    df_538 = pd.read_csv('https://projects.fivethirtyeight.com/2020-general-data/presidential_national_toplines_2020.csv',
                        parse_dates=['timestamp'], infer_datetime_format=True)
    dict_538 = {'Biden':df_538.loc[df_538['timestamp'] ==df_538['timestamp'].max()]['ecwin_chal'].values[0].round(2),
               'Trump':df_538.loc[df_538['timestamp'] ==df_538['timestamp'].max()]['ecwin_inc'].values[0].round(2)}
    dict_538['Biden'] = "{:.0%}".format(dict_538['Biden'])
    dict_538['Trump'] = "{:.0%}".format(dict_538['Trump'])

    r = requests.get('https://projects.economist.com/us-2020-forecast/president')
    econ = pd.read_html(str(BeautifulSoup(r.text, 'html.parser').find('table')))[0].T.reset_index()
    econ.columns = econ.iloc[0]
    econ.drop(0, inplace=True)
    econ_dict = {'Biden':econ['Joe BidenDemocrat'][1].split('or ')[1],
                 'Trump':econ['Donald TrumpRepublican'][1].split('or ')[1]}
    r=requests.get('https://www.270towin.com/2020-simulation/battleground-270')
    df_270 = pd.read_html(str(BeautifulSoup(r.text, 'html.parser').find('table')))[0].dropna(axis=1).T.reset_index()
    df_270.columns = df_270.iloc[0]

    df_270.drop(0, inplace=True)
    dict_270 = {'Biden':df_270['Biden'].values[0],
                 'Trump':df_270['Trump'].values[0]}
    df_270['Biden']  = (df_270['Biden'].str.rstrip('%').astype('float') / 100.0).round(2)
    df_270['Trump']  = (df_270['Trump'].str.rstrip('%').astype('float') / 100.0).round(2)

    dict_270= {'Biden':df_270['Biden'].values[0], 
               'Trump':df_270['Trump'].values[0]}
    dict_270['Biden'] = "{:.0%}".format(dict_270['Biden'])
    dict_270['Trump'] = "{:.0%}".format(dict_270['Trump'])
    
    df = pd.DataFrame([dict_538,econ_dict,
              dict_270], index=['FiveThirtyEight', 'Economist', 
                                '270 to Win']).sort_values(by='Biden', ascending=False)
    df['Biden'] = (df['Biden'].str.rstrip('%').astype('float') / 100.0)
    df['Trump'] = (df['Trump'].str.rstrip('%').astype('float') / 100.0)
    return df


# In[358]:


def get_tweet(df, prev_df):
    results = (100*(df-prev_df)).round().to_dict()
    biden = {x:y for x,y in results['Biden'].items() if y!=0}
    keys = []
    vals = []
    for key, value in biden.items():
        keys.append(key)
        vals.append('{0:+g}'.format(value))
    new_biden_dict = str(dict(zip(keys, vals))).strip("{}").replace("'", "")
    
    trump = {x:y for x,y in results['Trump'].items() if y!=0}
    keys = []
    vals = []
    for key, value in trump.items():
        keys.append(key)
        vals.append('{0:+g}'.format(value))
    
    new_trump_dict = str(dict(zip(keys, vals))).strip("{}").replace("'", "")
    
    if new_biden_dict == '':
        new_biden_dict = 'Probabilities unchanged since last update'
        
    if new_trump_dict == '':
        new_trump_dict = 'Probabilities unchanged since last update'
    status = 'New updates:\nBiden -- '+str(new_biden_dict)+' \nTrump -- '+str(new_trump_dict)
    return status


# In[359]:


def get_chart():
    fig, ax = plt.subplots(1,1, figsize=(8,4.5))
    df.plot(kind='bar', ax=ax, color=['tab:blue', 'tab:red'])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0,1)
    ax.legend(loc = (0.9,0.5), frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    x_offset = -0.1
    y_offset = 0.02
    for p in ax.patches:
        b = p.get_bbox()
        val = "{:.0%}".format(b.y1 + b.y0)        
        ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
    ax.annotate('@ElecProbBot', xy=(2.2, 0.33), fontsize=12)
    plt.xticks(rotation=0)
    plt.savefig('election_graph.png', dpi=500, bbox=True)


# In[360]:


def run_tweet(status):
    ckey = ckey
    csec  = csec

    atok= atok
    asec= asec

    auth = tweepy.OAuthHandler(ckey, csec)
    auth.set_access_token(atok, asec)

    # Create API object
    api = tweepy.API(auth)

    # Create a tweet
    api.update_with_media(filename='election_graph.png',status=status)
    df.to_pickle('election_probs.pkl')


# In[369]:


def run_update():
    get_chart()
    run_tweet(status = get_tweet(df, prev_df))
    
prev_df = pd.read_pickle('election_probs.pkl')
df = get_probs()
if (df == prev_df).any(axis=1).sum() != 3:
    run_update()
else: print('no updates right now')


# In[ ]:




