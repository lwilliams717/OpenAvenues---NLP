from metaflow import FlowSpec, step

class MinimumFlow(FlowSpec):
    
    @step
    def start(self):
        import pandas as pd
        self.df = pd.read_csv("open_ave_data.csv")
        self.df.fillna("nan",inplace=True)
        self.find = self.df["findings"].values.tolist()
        self.clin = self.df["clinicaldata"].values.tolist()
        self.exam = self.df["ExamName"].values.tolist()
        self.impr = self.df["impression"].values.tolist()
        self.corpus = self.find + self.clin + self.exam + self.impr
        self.next(self.eda_len)
            
    @step
    def eda_len(self):
        import matplotlib.pyplot as plt
        import numpy as np
        # impressions
        np_imprdarr = np.array(self.impr)
        get_string_length = np.vectorize(len)
        string_lengths = get_string_length(np_imprdarr)
        impr_lengths = np.mean(string_lengths)
        # findings
        np_findarr = np.array(self.find)
        string_lengths = get_string_length(np_findarr)
        find_lengths = np.mean(string_lengths)
        # clinical data
        np_clinarr = np.array(self.clin)
        string_lengths = get_string_length(np_clinarr)
        clin_lengths = np.mean(string_lengths)
        # exams
        np_examarr = np.array(self.exam)
        string_lengths = get_string_length(np_examarr)
        exam_lengths = np.mean(string_lengths)
        
        char_len = [impr_lengths, find_lengths, clin_lengths, exam_lengths]
        labels = ["impressions", "findings", "clinical data", "exam"]

        x = range(len(char_len))
        plt.bar(x, char_len)
        plt.xticks(x,labels)
        plt.savefig('text_length.png')
        self.next(self.eda_stop)
    
    @step
    def eda_stop(self):
        import nltk
        #nltk.download('punkt')
        import matplotlib.pyplot as plt
        from nltk import FreqDist
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        stop=set(stopwords.words('english'))
        words = list(map(lambda x : word_tokenize(x),self.impr))
        words = [word for i in words for word in i]
        stop_words = [i for i in words if i in stop]
        stop_freq = FreqDist(stop_words)
        top_20 = stop_freq.most_common(20)
        
        labels = [item[0] for item in top_20]
        values = [item[1] for item in top_20]
        
        plt.bar(labels, values, width=0.6)
        plt.savefig('impressions_stop_words.png')
        self.next(self.preproc)
    
    @step    
    def preproc(self):
        import re
        import nltk
        #nltk.download('stopwords')
        from nltk.corpus import stopwords
        stopwords_list = stopwords.words('english')
        clean_corpus = []
        for w in self.corpus:
            w = w.lower()
            w = re.sub(r'[^\w\s]','',w)
            words = w.split() 
            clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
            clean_corpus.append(clean_words)
        self.next(self.end)
    
    @step
    def end(self):
        print("done")

if __name__ == "__main__":
    MinimumFlow()