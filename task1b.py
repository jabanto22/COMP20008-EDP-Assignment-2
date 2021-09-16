import pandas as pd
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from itertools import combinations

def dataBlocking():
    
    # uncomment if error occurs
    #nltk.download('stopwords')
    #nltk.download('punkt')
    #nltk.download('wordnet')
    
    # read the two small data sets from amazon and google
    amazon = pd.read_csv('amazon.csv',encoding = 'ISO-8859-1')
    google = pd.read_csv('google.csv',encoding = 'ISO-8859-1')
    
    stopWords = set(stopwords.words('english'))
    wordnetLemma = WordNetLemmatizer()

    index = {}
    index['g_nomatch'] = 'g_nomatch'
    index['a_nomatch'] = 'a_nomatch'
    matched_index = {}
    a_block = {}
    a_block_cnt = 0
    g_block = {}
    g_block_cnt = 0
    
    for j in range(len(amazon)):
        amazon_title_filtered = []
        amazon_lemmaWord = []
        a_no_dup = []
        a_key_combi = []
        
        # remove symbols and digits from the text
        a_title = re.sub(r'[^\w\s]|[\d]',' ',amazon['title'][j])
        
        # tokenize and transform to lower case
        amazon_title_wordList = nltk.word_tokenize(a_title.lower())
        
        # remove stopwords
        amazon_title_filteredList = [w for w in amazon_title_wordList if not w in stopWords]
        
        # lemmatize
        for l_word in amazon_title_filteredList:
            amazon_lemmaWord.append(wordnetLemma.lemmatize(l_word,pos='n'))
        
        # remove duplicate words
        for w in amazon_lemmaWord:
            if w not in a_no_dup:
                a_no_dup.append(w)
        
        # join words before splitting into n-grams or n-words
        amazon_title_processed = ' '.join(c for c in a_no_dup)
        
        # create block keys
        a_block_key = sorted(amazon_title_processed.split())
        for a_block_k in a_block_key:
            if len(a_block_k) > 1:
                a_k = ''.join(k for k in a_block_k)
                if a_k not in a_key_combi:
                    a_key_combi.append(a_k)
        a_key_list = list(combinations(a_key_combi,2))
        
        # assign keys to records
        for a_keys in a_key_list:
            a_key = ''.join(k for k in a_keys)
            if a_key not in index:
                index[a_key] = a_key
                
            a_block[a_block_cnt] = index[a_key],amazon['idAmazon'][j]
            a_block_cnt = a_block_cnt + 1
        
    a_blocks = pd.DataFrame(a_block).transpose()
    col_name = ['block_key', 'product_id']
    a_blocks.columns = col_name
    
    for i in range(len(google)):
        google_name_filtered = []
        google_lemmaWord = []
        g_no_dup = []
        g_key_combi = []
        
        # remove symbols and digits from the text
        g_name = re.sub(r'[^\w\s]|[\d]',' ',google['name'][i])
        
        # tokenize and transform to lower case
        google_name_wordList = nltk.word_tokenize(g_name.lower())
        
        # remove stopwords
        google_name_filteredList = [w for w in google_name_wordList if not w in stopWords]
        
        # lemmatize
        for l_word in google_name_filteredList:
            google_lemmaWord.append(wordnetLemma.lemmatize(l_word,pos='n'))
        
        # remove duplicate words
        for w in google_lemmaWord:
            if w not in g_no_dup:
                g_no_dup.append(w)
        
        # join words before splitting into n-grams or n-words
        google_name_processed = ' '.join(c for c in g_no_dup)
        
        # create block keys
        g_block_key = sorted(google_name_processed.split())
        for g_block_k in g_block_key:
            if len(g_block_k) > 1:
                g_k = ''.join(k for k in g_block_k)
                if g_k not in g_key_combi:
                    g_key_combi.append(g_k)
        g_key_list = list(combinations(g_key_combi,2))
        
        # assign keys to records
        for g_keys in g_key_list:
            g_key = ''.join(k for k in g_keys)
            if g_key in index:
                g_block[g_block_cnt] = index[g_key],google['id'][i]
                g_block_cnt = g_block_cnt + 1
                matched_index[index[g_key]] = index[g_key]
            else:
                # keys not existing in index[] do not have a matched key with amazon
                g_block[g_block_cnt] = index['g_nomatch'],google['id'][i]
                g_block_cnt = g_block_cnt + 1            

    g_blocks = pd.DataFrame(g_block).transpose()
    col_name = ['block_key', 'product_id']
    g_blocks.columns = col_name
    # remove duplicate entries
    g_blocks = g_blocks.drop_duplicates(ignore_index=True)
    
    # check for keys not in matched_index and transfer to block a_nomatch
    for i in range(len(a_blocks)):
        if a_blocks['block_key'][i] not in list(matched_index):
            a_blocks['block_key'][i] = index['a_nomatch']
    # remove duplicate entries
    a_blocks = a_blocks.drop_duplicates(ignore_index=True)
    
    # combine sets a_blocks and g_blocks, remove blocks with no match and store to a temporary record
    frames = [a_blocks,g_blocks]
    concat_records = (pd.concat(frames)).sort_values(by='block_key')
    records = concat_records.reset_index().iloc[:,1:]
    records = records.set_index('block_key')
    records = records.drop('a_nomatch',axis=0)
    records = records.drop('g_nomatch',axis=0)
    records = records.reset_index()
    records.columns = ['block_key','product_id']
    
    # delete records tagged 'no match' if record exists in other blocks
    for i in range(len(a_blocks)):
        if a_blocks['block_key'][i] == 'a_nomatch':
            if a_blocks['product_id'][i] in list(records['product_id']):
                a_blocks = a_blocks.drop(i)
                
    for j in range(len(g_blocks)):
        if g_blocks['block_key'][j] == 'g_nomatch':
            if g_blocks['product_id'][j] in list(records['product_id']):
                g_blocks = g_blocks.drop(j)
     
    # write to csv file
    open('amazon_blocks.csv','w').write(a_blocks.to_csv(index=False))
    open('google_blocks.csv','w').write(g_blocks.to_csv(index=False))
    
                
# Test function 
def test():
    
    dataBlocking()
    
if __name__ == "__main__":
    test()
