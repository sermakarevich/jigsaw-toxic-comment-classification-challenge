from imports import *


class TextCleaner(BaseEstimator):    
    def __init__(self, contractions):
        self.wl = WordNetLemmatizer().lemmatize
        self.wn = wordnet.morphy
        self.wt = nltk.word_tokenize
        self.c_s = contractions
        self.ss = "'\":-.,=`*/|—~\\•"
        self.tp = re.compile('\w{1,}')
        self.tp2 = re.compile('([!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’])')

    def remove_digits(self, x):
        rd = x.maketrans(' ', ' ', digits)
        x = x.translate(rd)
        return x
    
    def lemmatizer(self, x):
        return [self.wl(self.wl('%s'%i, pos='v'), pos='a') for i in x]
    
    
    def morphy(self, x):
        m = self.wn(x)
        if m is None:
            return x
        else:
            return m
    
    def tokenize(self, s): 
        return self.tp.findall(s)
    
    def tokenize2(self, s): 
        return self.tp2.sub(r' \1 ', s).split()
        
    def morphy_list(self, x):
        return [self.morphy(i) for i in x]
        
    def contr(self, x):
        for k, v in self.c_s.items():
            x = x.replace(k, v)
        return x
    
    def special_symbols(self, x):
        for ss in self.ss:
            if len(x) > 1:
                x = x.replace(ss, '')
        return x
    
    def remove_stopwords(self, x):
        return [i for i in x if i not in stopwords.words('english')]
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = map(lambda r: r.replace('_', ' '), x)
        x = map(lambda r: r.replace('`', '\''), x)
        x = map(lambda r: self.remove_digits(r), x)
        x = map(self.contr, x)
#         x = map(lambda r: r.lower(), x)
#         x = map(self.contr, x)
        x = map(self.special_symbols, x)
#         x = map(self.wt, x)
        x = map(self.tokenize2, x)
#         x = map(self.remove_stopwords, x)
#         x = map(self.lemmatizer, x)
#         x = list(map(self.morphy_list, x))
#         x = map(lambda i: ' '.join(i), x)
        x = list(x)
        return x