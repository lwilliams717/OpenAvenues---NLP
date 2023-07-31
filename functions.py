import numpy as np

# function to clean data for processing
def clean_data(w):
        import re
        from nltk.corpus import stopwords
        stopwords_list = stopwords.words('english')
        clean_corpus = []
        w = w.lower()
        w=re.sub(r'[^\w\s]','',w)
        words = w.split() 
        clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
        return clean_words

# function to calculate the mean length of each category
def eda_len(find, clin, exam, impr):
    # impressions
    impr_lengths = np_length(impr)
    # findings
    find_lengths = np_length(find)
    # clinical data
    clin_lengths = np_length(clin)        
    # exams
    exam_lengths = np_length(exam)
    
    char_len = [impr_lengths, find_lengths, clin_lengths, exam_lengths]
    return char_len

# takes a body of text and finds the top 20 stop words
def eda_stop(lis):
    import nltk
    nltk.download('punkt')
    from nltk import FreqDist
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
        
    stop=set(stopwords.words('english'))
    words = list(map(lambda x : word_tokenize(x),lis))
    words = [word for i in words for word in i]
    stop_words = [i for i in words if i in stop]
    stop_freq = FreqDist(stop_words)
    top_20 = stop_freq.most_common(20)
    
    labels = [item[0] for item in top_20]
    values = [item[1] for item in top_20]
    return labels, values

# helper function to get the mean length of a list
def np_length(lis):
    np_lis = np.array(lis)
    get_len = np.vectorize(len)
    str_len = get_len(lis)
    len_mean = np.mean(str_len)
    return len_mean

# getting the target values for the model prediction
def target_vals(find, clin, exam, impr):
    # repeating the label value for the length of the documents in the column
    f_y = [0] * len(find)
    c_y = [1] * len(clin)
    e_y = [2] * len(exam)
    i_y = [3] * len(impr)

    # combine all the labels
    category_labels = f_y + c_y + e_y + i_y
    return category_labels

# averages all the word2vec vectors for each document
def word2vec_avg(find, clin, exam, impr, corpus, model):
    f = len(find)
    c = len(clin)
    e = len(exam)
    i = len(impr)

    document_vectors = []
    for doc in corpus:
        vectors = [model.wv[word] for word in doc if word in model.wv]
        if vectors:
            doc_vector = np.mean(vectors, axis=0)
        else:
            doc_vector = np.zeros(300)
        document_vectors.append(doc_vector)
        
    return document_vectors

def tfidf_plot(tfidf, plt, colors, labels):
    print("tfidf plot")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    tfidf_matrix_2d = pca.fit_transform(tfidf.toarray())
    
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
    
def word2vec_plot(plt, vectors_2d, colors):
    print("word2vec plot")
    color = -1
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(vectors_2d):
        if i % 954 == 0:
            color+=1
        x, y = vectors_2d[i, :]
        plt.scatter(x, y, c=colors[color])
    plt.savefig("word2vec.png")
    
def logistic_train(x,target):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X = np.array(x)
    y = np.array(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    log_reg = LogisticRegression()

    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    
    # Calculate accuracy of the model
    return accuracy_score(y_test, y_pred)