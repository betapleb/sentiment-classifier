import nltk
from nltk.tokenize import word_tokenize   # tokenizer
from nltk.stem import WordNetLemmatizer   # lemmatizer
import numpy as np
import random                             # shuffle data
import pickle                             # save
from collections import Counter           # count stuff

lemmatizer = WordNetLemmatizer()
hm_lines = 100000                         # how many lines



def create_lexicon(pos, neg):
    lexicon = []

    with open(pos, 'r', encoding='cp437') as f:  # added cp437 bc got UnicodeDecodeError
        contents = f.readlines()
        for line in contents[:hm_lines]:
            all_words = word_tokenize(line.lower())
            lexicon += list(all_words)

    with open(neg, 'r', encoding='cp437') as f:  
        contents = f.readlines()
        for line in contents[:hm_lines]:
            all_words = word_tokenize(line.lower())
            lexicon += list(all_words)

    # after populating lexicon with every word in dataset,
    # lemmatize the common and uncommon words so they wont skew results.
    
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    word_counts = Counter(lexicon)  # word_counts = {'the':52320, 'and':24004}
    finalLexicon = []
    for w in word_counts:
        # print(word_counts[w])   # prints count of each word in lexicon. 
        if 1000 > word_counts[w] > 50: # only append words appearing between 50-1000 times. don't want super repetitive words.
            finalLexicon.append(w)

    print('Num elements in every input vector:', len(finalLexicon))
    
    return finalLexicon     # list of words ['the', 'a']


# each time lemma in lexicon is found, the index of that lemma in the lexicon is turned "on"
# in previously numpy zeros array that is the same length as the lexicon.
def sample_handling(sample, lexicon, classification):
    featureSet = []
    
    with open(sample, 'r', encoding='cp437') as f:
        contents = f.readlines()
        for lines in contents[:hm_lines]:
            current_words = word_tokenize(lines.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))       # numpy.zeros returns array of zeros of a given length 
            for word in current_words:              # interate through lemmatized words 
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1      # add 1 to index value in features array that's the same index of word in lexicon. 
            features = list(features)
            featureSet.append([features, classification])


    '''
    what featureSet looks like
    [ [feartures, label] ]
    [ [0 1 0 1 1 0], [1 0 ] ] 
    [ [] ],  
    ]
    '''
    return featureSet


# create lexicon based on raw sample data
# everything comes together
def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])
    random.shuffle(features)                        # shuffle data
    features = np.array(features)                   # convert to numpy array


    testing_size = int(test_size*len(features))     # 10% of features

    # build training & testing sets
    train_x = list(features[:,0][:-testing_size])   # [:,0] getindex 0 - every feature up to last 10%
    train_y = list(features[:,1][:-testing_size])   
    
    test_x = list(features[:,0][-testing_size:])    # -testing_size till the very end. 
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y   



if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

    # pickle this data:
    with open('sentiment_set.pickle','wb') as f:         # wb = write binary
        pickle.dump([train_x,train_y,test_x,test_y],f)   # dump train & tst values to a list into a file.

    # read data from pickle
    # with open('sentiment_set.pickle', 'rb') as f:
        # data = pickle.load(f)

    







            
