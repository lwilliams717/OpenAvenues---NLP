from metaflow import FlowSpec, step

class NLPFlow(FlowSpec):
    
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
        from functions import eda_len, eda_stop
        
        # character length
        char_len = eda_len(self.find, self.clin, self.exam, self.impr)
        labels = ["impressions", "findings", "clinical data", "exam"]
        x = range(len(char_len))
        plt.bar(x, char_len)
        plt.xticks(x,labels)
        plt.savefig('text_length.png')
        
        # stop words for impressions 
        labels, values = eda_stop(self.impr)
        plt.bar(labels, values, width=0.6)
        plt.savefig('impressions_stop_words.png')
        self.next(self.preproc)
    
    @step    
    def preproc(self):
        from functions import target_vals, clean_data
        
        self.clean_corpus = list(map(clean_data,self.corpus))
        self.corpus = [' '.join(words) for words in self.clean_corpus]
        self.target_vals = target_vals(self.find, self.clin, self.exam, self.impr)
        self.colors = ['red', 'green', 'blue','yellow']
        self.next(self.tfidf, self.word2vec)    
    
    # parallel step
    @step
    def tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer()
        self.tfidf_documents = vectorizer.fit_transform(self.corpus)
        self.next(self.tfidf_plot)
    
    @step
    def word2vec(self):
        import multiprocessing
        from gensim.models import Word2Vec
        import numpy as np
        from functions import word2vec_avg
        
        self.corpus = self.clean_corpus
        cores = multiprocessing.cpu_count()
        self.w2v_model = Word2Vec(min_count=5,window=5,vector_size=300,workers=cores-1,max_vocab_size=100000)
        self.w2v_model.build_vocab(self.corpus)
        self.w2v_model.train(self.corpus,total_examples=self.w2v_model.corpus_count,epochs=50)
        self.document_vectors = np.array(word2vec_avg(self.find, self.clin, self.exam, self.impr, self.corpus, self.w2v_model))
        self.next(self.word2vec_plot)    
    
    # parallel plot
    @step
    def word2vec_plot(self):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt  
        from functions import word2vec_plot      
        
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(self.document_vectors)
        word2vec_plot(plt, vectors_2d, self.colors)
        self.document_vectors
        self.target_vals
        self.next(self.join)
        
    @step
    def tfidf_plot(self):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from functions import tfidf_plot
        labels = ['Findings', 'Clinical Data', 'Exam Name', 'Impressions']
        tfidf_plot(self.tfidf_documents, plt, self.colors, labels)
        self.tfidf_documents
        self.next(self.join)
    
    # compare the models in join step
    @step
    def join(self, inputs):
        import numpy as np
        from functions import logistic_train
        target_vals = inputs.word2vec_plot.target_vals
        # calculate w2v accuracy
        print("W2V Accuracy:", logistic_train(inputs.word2vec_plot.document_vectors, target_vals))
        
        # calculate tfidf accuracy
        tfidf_arr = inputs.tfidf_plot.tfidf_documents.toarray()
        print("TFIDF Accuracy:", logistic_train(tfidf_arr, target_vals))
        self.next(self.end)
    
    @step
    def end(self):
        print("done")

if __name__ == "__main__":
    NLPFlow()