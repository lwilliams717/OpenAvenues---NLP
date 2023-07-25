from metaflow import FlowSpec, step

class MinimumFlow(FlowSpec):
    
    def clean_data(self, w):
        import re
        from nltk.corpus import stopwords
        stopwords_list = stopwords.words('english')
        clean_corpus = []
        w = w.lower()
        w=re.sub(r'[^\w\s]','',w)
        words = w.split() 
        clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
        return clean_words
    
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
    def target_vals(self):
        # repeating the label value for the length of the documents in the column
        f_y = [0] * len(self.find)
        c_y = [1] * len(self.clin)
        e_y = [2] * len(self.exam)
        i_y = [3] * len(self.impr)

        # combine all the labels
        self.category_labels = f_y + c_y + e_y + i_y
        self.next(self.join)
    
    @step    
    def preproc(self):
        import re
        import nltk
        #nltk.download('stopwords')
        from nltk.corpus import stopwords
        self.clean_corpus = list(map(self.clean_data,self.corpus))
        self.corpus = [' '.join(words) for words in self.clean_corpus]
        self.next(self.target_vals, self.tfidf, self.word2vec)
    
    @step
    def tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        self.tfidf_documents = vectorizer.fit_transform(self.corpus)
        self.next(self.join)
    
    @step
    def word2vec(self):
        import multiprocessing
        import gensim
        from gensim.models import Word2Vec
        import numpy as np
        self.corpus = self.clean_corpus
        cores = multiprocessing.cpu_count()
        self.w2v_model = Word2Vec(min_count=5,window=5,vector_size=300,workers=cores-1,max_vocab_size=100000)
        self.w2v_model.build_vocab(self.corpus)
        self.w2v_model.train(self.corpus,total_examples=self.w2v_model.corpus_count,epochs=50)
        
        f = len(self.find)
        c = len(self.clin)
        e = len(self.exam)
        i = len(self.impr)

        self.document_vectors = []
        for doc in self.corpus:
            vectors = [self.w2v_model.wv[word] for word in doc if word in self.w2v_model.wv]
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(300)
            self.document_vectors.append(doc_vector)
        self.document_vectors = np.array(self.document_vectors)
        self.next(self.join)
    
    @step
    def join(self, inputs):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # word2vec logistic  
        X = np.array(inputs.word2vec.document_vectors)
        y = np.array(inputs.target_vals.category_labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
        log_reg = LogisticRegression()

        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)
        
        # Calculate accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print("W2V Accuracy:", accuracy)
        
        #tdidf logistic
        X = inputs.tfidf.tfidf_documents.toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)

        # Calculate accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print("TFIDF Accuracy:", accuracy)
        
        self.w2v_model = inputs.word2vec.w2v_model
        self.tfidf_documents = inputs.tfidf.tfidf_documents
        self.document_vectors = inputs.word2vec.document_vectors
        self.next(self.word2vec_plot)    
    
    @step
    def word2vec_plot(self):
        import sklearn
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt        
        word_vectors = self.w2v_model.wv.vectors
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(self.document_vectors)

        color = -1
        colors = ['red', 'green', 'blue','yellow']
        plt.figure(figsize=(10, 8))
        for i, word in enumerate(vectors_2d):
            if i % 954 == 0:
                color+=1
            x, y = vectors_2d[i, :]
            plt.scatter(x, y, c=colors[color])
        plt.savefig("word2vec.png")
        self.next(self.tfidf_plot)
        
    @step
    def tfidf_plot(self):
        import sklearn
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        colors = ['red', 'green', 'blue','yellow']
        labels = ['Findings', 'Clinical Data', 'Exam Name', 'Impressions']

        pca = PCA(n_components=2)
        tfidf_matrix_2d = pca.fit_transform(self.tfidf_documents.toarray())
        # print(tfidf_matrix_2d.shape)
        color = -1
        label = -1
        for i, document in enumerate(tfidf_matrix_2d):
            if i % 954 == 0:
                color+=1
                label+=1
            x_coords = document[0]
            y_coords = document[1]
            plt.scatter(x_coords, y_coords, color=colors[color], label=labels[label])

        # Set plot title and axis labels.
        plt.title("TF-IDF Matrix Scatter Plot")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        # Display the scatter plot.
        plt.savefig("tfidf.png")
        self.next(self.end)
    
    @step
    def end(self):
        print("done")

if __name__ == "__main__":
    MinimumFlow()