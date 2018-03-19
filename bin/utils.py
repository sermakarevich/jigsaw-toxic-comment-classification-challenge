from imports import * 


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): 
    return re_tok.sub(r' \1 ', s).split()


def get_coefs(word, *arr): 
    return word.lower(), np.asarray(arr, dtype='float32')

def get_emb_dict(word, *arr): 
    return word.lower(), 1


def substitute(word, neg, pos):
    for n in neg:
        if n.lower() in word.lower():
            return n
    for p in pos:
        if p.lower() in word.lower():
            return p
    return None


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, multiprocessing.cpu_count())
    pool = Pool(multiprocessing.cpu_count())
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def applyParallel(df, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(val) for val in df)
    return retLst


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
