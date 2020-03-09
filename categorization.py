import os
import re
import regex
import nltk
import string
import pandas as pd
import numpy as np
import warnings
import sklearn
import warnings
import re
import docx
import pandas as pd


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.text import Tokenizer

from numpy import array
from numpy import asarray
from numpy import zeros
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.layers import Dense, Embedding, LSTM, GRU
from keras import regularizers
from keras import optimizers
from zipfile import ZipFile


cwd=os.getcwd()
print(cwd)
#from categorization import *
from data_layer import DataLayer
#from summary_extract import Extractinglayer



zipfile_path = ZipFile(r'C:\Users\bollud\Desktop\ML\training.zip', 'r')

def UnZip(zipfile):
    zf=zipfile_path.extractall('C:/Users/bollud/Desktop')
    zipfile_path.close()
    return zf

zip=UnZip(zipfile_path)

#==============================================================================
# for picking folder names [target labels]
#==============================================================================

#import os
#path=(r'C:\Users\bollud\Desktop\ML Platform\training')  
#folders = []
## r=root, d=directories, f = files
#for r, d, f in os.walk(path):
#    for folder in d:
#        folders.append(folder)
##folders.append(os.path.join(r, folder))   #for joining root path add this line
#for f in folders:
#    print(f)
##==============================================================================
#def categorizeFolder(folder_name):
#    categorize_regex = re.compile(r'action|information|no\s*action')
#    
#    if re.search(categorize_regex,folder_name):   
#==============================================================================  
parent_dir = r'C:\Users\bollu\Desktop\training'

test=(r'C:\Users\bollu\Desktop\0.txt')

def Table_Creation(parent_dir,test):
    #parent_dir = r'C:\Users\bollud\Desktop\training'
    categorization_df = pd.DataFrame()
    categorization_df['Extracted_Text'] = ''
    
    for path in os.walk(parent_dir):
        print (path[1])
        category_folders = path[1]
        
        i = 0
        
        for folder in category_folders:
            cateogory_folder = parent_dir+r'\\'+folder
            print (cateogory_folder)
            
            for file in os.listdir(cateogory_folder):
                Path = cateogory_folder+r'\\'+file
                categorization_df.loc[i,'Category'] = folder
                categorization_df.loc[i,'Path'] = Path
                data_obj = DataLayer(Path)
                file_name,some_dict = data_obj.get_text()
                data_obj2 = Extractinglayer(some_dict)
                article_text = data_obj2.Main()
                categorization_df.loc[i,'Extracted_Text'] = article_text
                
                i = i+1
                
                data_obj = DataLayer(test)
                file_name,some_dict = data_obj.get_text()
                data_obj2 = Extractinglayer(some_dict)
                article_text = data_obj2.Main()
                #df_Test2 = pd.DataFrame(data=None, columns=categorization_df.columns, index=len(article_text))
                df_Test2=pd.DataFrame()
                df_Test2['Extracted_Text'] = ''
                df_Test2.loc[0,'Extracted_Text']=article_text
                
               
                
    return categorization_df,df_Test2
        
#df,df_Test=Table_Creation(parent_dir,test)

#df_Table=df_Test[[""]]
#df_Table=pd.DataFrame({'Document':df_Test["Extracted_Text"]})

#df_Table=pd.DataFrame({'Document':df_Test["Extracted_Text"],'Category':[""]})
#==============================================================================
# pre-processing
#==============================================================================
#def process_text(text):
#    #text=text.astype(str)
#    text=text.lower()
#    text=re.sub(r'\b\w{1,3}\b', '', text)
#    #text= re.sub(r"\n", "",text)
#    #text= re.sub(r'\W*\b\w{1,3}\b', " ", text)
#    text =re.sub(r"\s+"," ", text, flags = re.I)
#    text=text.strip()
#    return text
#f=pd.Series(Clean_Data.split)
#
#Clean_Data=process_text(train_text)
#test_cleandata=process_text(df_Test)
#train_text=df["Extracted_Text"].to_string()
##train_text=pd.Series(df["Extracted_Text"])


def process_text(df):
    df.Extracted_Text=df.Extracted_Text.astype(str)
    df.Extracted_Text=df.Extracted_Text.str.lower().astype(str)
    df.Extracted_Text=df.Extracted_Text.apply((lambda x: re.sub(r"\n", "",x)))
    df.Extracted_Text=df.Extracted_Text.apply((lambda x: re.sub(r'\W*\b\w{1,3}\b', " ", x))).astype(str)
    df.Extracted_Text=df.Extracted_Text.str.strip()
    return df.Extracted_Text

#Clean_Data=process_text(df)
#test_cleandata=process_text(df_Test)
    
   
 
#def process_text(text):
#    #text=df["Extracted_Text"]
#    df["Extracted_Text"]=df["Extracted_Text"].astype(str)
#    df["Extracted_Text"]=df["Extracted_Text"].str.lower().astype(str)
#    df["Extracted_Text"] = df["Extracted_Text"].apply((lambda x: re.sub(r"\n", "",x)))
#    df["Extracted_Text"]=df["Extracted_Text"].apply((lambda x: re.sub(r'\W*\b\w{1,3}\b', " ", x))).astype(str)
#    df["Extracted_Text"]=df["Extracted_Text"].str.strip()
#    return df["Extracted_Text"]
#
#Clean_Data=process_text(df)
#test_cleandata=process_text(df_Test)
#==============================================================================
# removing punctuations
#==============================================================================

def remove_punctuation(text):
    PUNCT_TO_REMOVE = string.punctuation 
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
#Punc_Removed=Clean_Data.apply(lambda x: remove_punctuation(x))

#def remove_punctuation(text):
#    PUNCT_TO_REMOVE = string.punctuation 
#    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
#Test_Punc_Removed=test_cleandata.apply(lambda x: remove_punctuation(x))

#df["Extracted_Text"]=df["Extracted_Text"].apply(lambda x: remove_punctuation(x))

#==============================================================================
#STOP WORDS REMOVAL
#==============================================================================
", " .join(stopwords.words('english'))
stopwords=set(stopwords.words('english'))

def remove_stopwords(sent):
    return " " .join([word for word in str(sent).split()if word not in stopwords])

#Stop_Removed=Punc_Removed.apply(lambda x: remove_stopwords(x))

#def remove_stopwords(sent):
#    return " " .join([word for word in str(sent).split()if word not in stopwords])

#Test_Stop_Removed=Test_Punc_Removed.apply(lambda x: remove_stopwords(x))

#df["Extracted_Text"]=df["Extracted_Text"].apply(lambda x: remove_stopwords(x))

#def helo(parent_dir):
#    Clean_Data=process_text(df)
#    Punc_Removed=Clean_Data.apply(lambda x: remove_punctuation(x))
#    Stop_Remove=Punc_Removed.apply(lambda x: remove_stopwords(x))
#    return Stop_Remove
#final_data=helo(parent_dir)
#==============================================================================
def len_Vocab(Stop_Removed):
    word_tokenizer=Tokenizer()
    word_tokenizer.fit_on_texts(Stop_Removed)
    vocablength = len(word_tokenizer.word_index) + 1
    return word_tokenizer,vocablength
#word_tokenizer,vocab_length=len_Vocab(Stop_Removed)
#word_tokenizer,vocab_length=len_Vocab(Test_Stop_Removed)

def encoding(word_tokenizer,Stop_Removed):
    embedded_sentences = word_tokenizer.texts_to_sequences(Stop_Removed)
    return embedded_sentences
#Embedded_Sentences=encoding(word_tokenizer,Stop_Removed)

def TestEncoding(word_tokenizer,Test_Stop_Removed):
    embedded_sentences = word_tokenizer.texts_to_sequences(Test_Stop_Removed)
    return embedded_sentences
#Test_Embedded_Sentences=TestEncoding(word_tokenizer,Test_Stop_Removed)

#==============================================================================
def Long_Sen(Stop_Removed):
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(Stop_Removed, key=word_count)
    long_sentence = len(word_tokenize(longest_sentence))
    return long_sentence

#length_long_sentence=Long_Sen(Stop_Removed)
#==============================================================================    
#pad_sentences = pad_sequences(Embedded_Sentences, length_long_sentence, padding='post')
maxlen=5000
def padding(sent,maxlen):
    pad_sentences = pad_sequences(sent, maxlen=5000,padding='post')
    return pad_sentences,maxlen

#padded_sentences,train_maxlen=padding(Embedded_Sentences,maxlen)

#Test_padded_sentences,Test_maxlen=padding(Test_Embedded_Sentences,maxlen)

#==============================================================================


#embeddings_dictionary = dict()
#emails_embeddingfile = open(r'C:\Users\bollud\Documents\Emails_embeddings.txt', encoding="utf8")
#
#for line in emails_embeddingfile:
#    records = line.split()
#    word = records[0]
#    vector_dimensions = asarray(records[1:], dtype='float32')
#    embeddings_dictionary [word] = vector_dimensions
#
#emails_embeddingfile.close()

# Save dict
#dictionary = {'hello': 1,2,3,4}
#np.save('Emails_embeddings.npy', embeddings_dictionary) 
#np.save('glove_dict.npy', word_embeddings)

# Load
#read_dictionary = np.load('glove_dict.npy',allow_pickle='TRUE').item()
#print(read_dictionary['court']) # displays "world"

#=============================================================================
def embeddings():
    path2= r'C:\Users\bollud\Documents\Emails_embeddings.npy'
    embeddings_dictionary= np.load(path2, allow_pickle='TRUE').item()
    return embeddings_dictionary

#embeddings_dictionary=embeddings()
#=============================================================================
def Embedding_vectors(vocab_length,word_tokenizer,embeddings_dictionary):
    embedding_matrix = zeros((vocab_length, 300))
    for word, index in word_tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            return embedding_matrix

#embedding_matrix=Embedding_vectors(vocab_length,word_tokenizer,embeddings_dictionary)
#==============================================================================
def OHE(df,padded_sentences):
    le = LabelEncoder()
    df["labels"] = le.fit_transform(df["Category"])
    X=padded_sentences
    y=to_categorical(df["labels"])
    unique_labels=df['Category'].nunique()
    return X,y,unique_labels
#X,y,unique_labels=OHE(df,padded_sentences)
#==============================================================================
# import OneHotEncoder
​# Instatniate One-Hot-Encoder
#ohe = OneHotEncoder(categories = "auto",sparse = False)
​# One-Hot-Encode Class column of df
#y = ohe.fit_transform(df[["Category"]])
#==============================================================================

def Training(X,y,vocab_length,maxlen,embedding_matrix,unique_labels):
    X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.2,random_state=42,shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.3,random_state=42,shuffle=True)
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    
    embedding_dim=300
    model = Sequential()
    embedding_layer = Embedding(vocab_length, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    #model.add(LSTM(100, activation='relu'))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(100, input_shape=(300,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(unique_labels, activation='softmax'))
    
    sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, y,  batch_size=32, epochs=5,  validation_data=(X_val, y_val))
    
    return X_test, model

#X_test, model=Training(X,y,vocab_length,maxlen,embedding_matrix,unique_labels)

#==============================================================================

def Results(df,Trained_model,Test_padded_sentences):
    final_categories = list(set(df['Category']))
    pred=Trained_model.predict(Test_padded_sentences)
    predict_class = np.argmax(pred, axis=1)
    predict_labels=[final_categories[logit] for logit in predict_class]
    return predict_labels

#output=Results(df,Trained_model,Test_padded_sentences)
#Final_Table=df_Table.insert(1, "Category", output, True) 
#Final_Table=pd.DataFrame({'Document':df_Table, 'Category':output})


#original_Sent = list(set(df['Extracted_Text']))
#decode_text = np.argmax(X_test, axis=0)
#mapped_sentences3=[original_Sent[logit] for logit in decode_text]


#==============================================================================
parent_dir = r'C:\Users\bollud\Desktop\training'

test=(r'C:\Users\bollud\Desktop\0.txt')

from nltk.corpus import stopwords
", " .join(stopwords.words('english'))
stopwords=set(stopwords.words('english'))

def Training_page(parent_dir,test):
    df,df_Test=Table_Creation(parent_dir,test)
    df_Table=pd.DataFrame({'Document':df_Test["Extracted_Text"],'Category':[""]})
    Clean_Data=process_text(df)
    Punc_Removed=Clean_Data.apply(lambda x: remove_punctuation(x))
    Stop_Removed=Punc_Removed.apply(lambda x: remove_stopwords(x))
    word_tokenizer,vocab_length=len_Vocab(Stop_Removed)
    Embedded_Sentences=encoding(word_tokenizer,Stop_Removed)
    length_long_sentence=Long_Sen(Stop_Removed)
    maxlen=5000
    padded_sentences,train_maxlen=padding(Embedded_Sentences,maxlen)
    embeddings_dictionary=embeddings()
    embedding_matrix=Embedding_vectors(vocab_length,word_tokenizer,embeddings_dictionary)
    X,y,unique_labels=OHE(df,padded_sentences)
    X_test, model=Training(X,y,vocab_length,maxlen,embedding_matrix,unique_labels)
    
    return model


Trained_model=Training_page(parent_dir,test)

def Testing_page(parent_dir,test,Trained_model):
    df,df_Test=Table_Creation(parent_dir,test)
    df_Table=pd.DataFrame({'Document':df_Test["Extracted_Text"],'Category':[""]})
    test_cleandata=process_text(df_Test)
    Test_Punc_Removed=test_cleandata.apply(lambda x: remove_punctuation(x))
    Test_Stop_Removed=Test_Punc_Removed.apply(lambda x: remove_stopwords(x))
    word_tokenizer=Tokenizer()
    Test_Embedded_Sentences=TestEncoding(word_tokenizer,Test_Stop_Removed)
    maxlen=5000
    Test_padded_sentences,Test_maxlen=padding(Test_Embedded_Sentences,maxlen)
    output=Results(df,Trained_model,Test_padded_sentences)
    #oput=pd.Series(output)
    #Final_Table=df_Table.insert(1, "Category", output, True) 
    Final_Table=pd.DataFrame({"DOCUMENT":df_Table['Document'], "CATEGORY":output})
    return Final_Table


Frame=Testing_page(parent_dir,test,Trained_model)
