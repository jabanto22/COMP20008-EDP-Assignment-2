import pandas as pd
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

import textdistance

def dataLinkage():
    
    # uncomment if error occurs
    #nltk.download('stopwords')
    #nltk.download('punkt')
    #nltk.download('wordnet')
    
    # read the two small data sets from amazon and google
    amazon_small = pd.read_csv('amazon_small.csv',encoding = 'ISO-8859-1')
    google_small = pd.read_csv('google_small.csv',encoding = 'ISO-8859-1')
    
    stopWords = set(stopwords.words('english'))
    wordnetLemma = WordNetLemmatizer()

    amazon_google_data_linkage = {}
    
    for j in range(len(amazon_small)):
        amazon_title_lemmaWord = []
        amazon_title_filtered = []
        a_title_no_dup = []
        max_similarity_score = 0
        g_id = ''
        
        # preprocess texts and remove stopwords from amazon_small data
        a_title = re.sub(r'[^\w\s]',' ',amazon_small['title'][j])
        amazon_title_wordList = nltk.word_tokenize(a_title.lower())
        amazon_title_filteredList = [w for w in amazon_title_wordList if not w in stopWords]
        for l_word in amazon_title_filteredList:
            amazon_title_lemmaWord.append(wordnetLemma.lemmatize(l_word,pos='v'))
        for w in sorted(amazon_title_lemmaWord):
            if w not in a_title_no_dup:
                a_title_no_dup.append(w)
        amazon_title_filtered = ' '.join(c for c in a_title_no_dup)
        
        for i in range(len(google_small)):
            google_name_lemmaWord = []
            google_name_filtered = []
            g_name_no_dup = []
            
            # preprocess texts and remove stopwords from google_small data
            g_name = re.sub(r'[^\w\s]',' ',google_small['name'][i])
            google_name_wordList = nltk.word_tokenize(g_name.lower())
            google_name_filteredList = [w for w in google_name_wordList if not w in stopWords]
            for l_word in google_name_filteredList:
                google_name_lemmaWord.append(wordnetLemma.lemmatize(l_word,pos='v'))
            for w in sorted(google_name_lemmaWord):
                if w not in g_name_no_dup:
                    g_name_no_dup.append(w)
            google_name_filtered = ' '.join(c for c in g_name_no_dup)
            
            # compute similarity score using JaroWinkler distance
            name_similarity_score = textdistance.JaroWinkler(qval=3).normalized_similarity(google_name_filtered, amazon_title_filtered)
            if (name_similarity_score > 0.52):
                if name_similarity_score > max_similarity_score:
                    g_id = google_small['idGoogleBase'][i]
                    max_similarity_score = name_similarity_score
                        
        if g_id:
            amazon_google_data_linkage[amazon_small['idAmazon'][j]] = g_id
            
    data_linkage = pd.Series(amazon_google_data_linkage).reset_index()
    col_name = ['idAmazon', 'idGoogleBase']
    data_linkage.columns = col_name
    open('task1a.csv','w').write(data_linkage.to_csv(index=False))
        
# Test function 
def test():
    
    dataLinkage()
    
if __name__ == "__main__":
    test()
