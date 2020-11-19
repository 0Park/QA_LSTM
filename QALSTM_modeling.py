#%%
#download the KorQuAD
import json
with open('C:/Users/Young Hun Park/Downloads/KorQuAD_v1.0_train.json') as train_file:
    train_data=json.load(train_file)
with open('C:/Users/Young Hun Park/Downloads/KorQuAD_v1.0_dev.json') as dev_file:
    dev_data=json.load(dev_file)
#%%
train_data=train_data['data']
dev_data=dev_data['data']
#%%
# make data into DataFrame
import pandas as pd
def json_to_df(data):
    arrayForDF=[]
    for current_subject in data:
        subject=current_subject['title']
        for current_context in current_subject['paragraphs']:
            context=current_context['context']
            for current_question in current_context['qas']:
                question=current_question['question']
                for answer in current_question['answers']:
                    answer_text=answer['text']
                    answer_start=answer['answer_start']
                    
                    record={
                        "answer_text":answer_text,
                        "answer_start":answer_start,
                        "question":question,
                        "context":context,
                        "subject":subject
                        
                    }
                    arrayForDF.append(record)
    df=pd.DataFrame(arrayForDF)
    return df
#%%
data_df=json_to_df(train_data)
dev_df=json_to_df(dev_data)
#%%
# extract the answertext from the paragraph
from nltk.tokenize import sent_tokenize

def get_answer_context(df):
    length_context=0
    answer= ""
    for sentence in sent_tokenize(df.context):
        length_context+=len(sentence)+1
        if df.answer_start <= length_context:
            if len(sentence) >= len(str(df.answer_text)):
                if answer=="":
                    return sentence
                else:
                    return answer+""+sentence
            else:
                answer+=sentence
data_df['entire_answer_text']=data_df.apply(lambda row: get_answer_context(row),axis=1)
dev_df['entire_answer_text']=dev_df.apply(lambda row: get_answer_context(row),axis=1)
#%%
# 형태소 분석 and 조사,어미,punctuation 삭제
from konlpy.tag import Okt

def cleaningText(text):
    twitter=Okt()
    p_sentence=[]
    for sentence in text:
        malist=twitter.pos(sentence,norm=True,stem=True)
        r=[]
        for word in malist:
            if not word[1] in ["Josa","Eomi","Punctuation"]:
                r.append(word[0])
        p_sentence.append(r)
    
    return p_sentence
#%%
dev_question_text=dev_df['question']
dev_answer_text=dev_df['entire_answer_text']
dev_question_text=cleaningText(dev_question_text)
dev_answer_text=cleaningText(dev_answer_text)

dev_QA_df=pd.DataFrame(columns=['question','answer'])
dev_QA_df['question']=dev_question_text
dev_QA_df['answer']=dev_answer_text
dev_QA_df.to_csv('C:/Users/Young Hun Park/Desktop/python beginner/NLP/dev_QA_df.csv')
#%%
# cleansing data and save into csv
question_text=data_df['question']
answer_text=data_df['entire_answer_text']

question_text=cleaningText(question_text)
answer_text=cleaningText(answer_text)

QA_df=pd.DataFrame(columns=['question','answer'])
QA_df['question']=question_text
QA_df['answer']=answer_text

QA_df.to_csv('C:/Users/Young Hun Park/Desktop/python beginner/NLP/QA_df.csv')
#%%
# start here
import pandas as pd
QA_df=pd.read_csv('C:/Users/Young Hun Park/Desktop/python beginner/NLP/QA_df.csv')


import gensim
model=gensim.models.Word2Vec.load('C:/Users/Young Hun Park/Desktop/python beginner/NLP/KorQuAD.model')


# integer Encoding(question)
question=QA_df['question']
answer=QA_df['answer']
text=pd.concat([question,answer],axis=0)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len=100

tokenizer=Tokenizer()
tokenizer.fit_on_texts(text)

q_sequences=tokenizer.texts_to_sequences(QA_df['question'])
a_sequences=tokenizer.texts_to_sequences(QA_df['answer'])

#dev_q_sequences=tokenizer.texts_to_sequences(dev_QA_df['question'])
#dev_a_sequences=tokenizer.texts_to_sequences(dev_QA_df['question'])

q_sequences=pad_sequences(q_sequences,maxlen=max_len)
a_sequences=pad_sequences(a_sequences,maxlen=max_len)

#dev_q_sequences=pad_sequences(dev_q_sequences,maxlen=max_len)
#dev_a_sequences=pad_sequences(dev_a_sequences,maxlen=max_len)

word_index=tokenizer.word_index
vocab_size=len(word_index)+1


# make Embeeding matrix using Word2Vec
import numpy as np
Embedding_matrix=np.zeros((vocab_size,200))

for word, i in word_index.items():
    word=word.replace("'","")
    try:
        embedding_vector=model[word]
    except KeyError:
        continue
    Embedding_matrix[i]=embedding_vector

an_sequences=np.array(a_sequences)
np.random.shuffle(an_sequences)

#dev_an_sequences=np.array(dev_a_sequences)
#np.random.shuffle(dev_an_sequences)
#%%
import tensorflow as tf
from tensorflow.keras.layers import Dense

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key와 value는 같음
    # query shape == (batch_size, max_len,hidden size)
    # value shape == (batch_size,max_len.hiddnesize)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
#%%
from tensorflow.keras.layers import Embedding,LSTM,Bidirectional,MaxPool1D
from tensorflow.keras.layers import Lambda,Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

embedding_dim=200
max_len=100
hidden_size=50
margin=0.2

def get_cosine_similarity():
    dot=lambda a,b: K.batch_dot(a, b,axes=1)
    return lambda x: dot(x[0],x[1])/K.maximum(K.sqrt(dot(x[0],x[0])*dot(x[1],x[1])),K.epsilon())

question=Input(shape=(max_len,),dtype='float64',name='question_base')
answer=Input(shape=(max_len),dtype='float64',name='answer')
answer_good=Input(shape=(max_len),dtype='float64',name='answer_good_base')
answer_bad=Input(shape=(max_len,),dtype='float64',name='answer_bad_base')

qa_embedding=Embedding(vocab_size,embedding_dim,weights=[Embedding_matrix],input_length=max_len,trainable=True,mask_zero=True)
bi_lstm=Bidirectional(LSTM(units=hidden_size,dropout=0.2,return_sequences=True))

question_embedding=qa_embedding(question)
question_lstm=bi_lstm(question_embedding)
question_pooling=MaxPool1D(max_len)(question_lstm)

answer_embedding=qa_embedding(answer)
answer_lstm=bi_lstm(answer_embedding)
answer_pooling=MaxPool1D(max_len)(answer_lstm)

similarity=get_cosine_similarity()
question_answer_merged=Lambda(similarity,name='lambda_layer')([question_pooling,answer_pooling])
lstm_model=Model(name='q_bilstm',inputs=[question,answer],outputs=question_answer_merged)
good_similarity=lstm_model([question,answer_good])
bad_similarity=lstm_model([question,answer_bad])

# compute the loss
loss=Lambda(lambda x: K.maximum(0.0,margin-x[0]+x[1]))([good_similarity,bad_similarity])
#return training and prediction model

training_model=Model(inputs=[question,answer_good,answer_bad],outputs=loss,name='training_model')
training_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer="rmsprop")
prediction_model=Model(inputs=[question,answer_good],outputs=good_similarity,name='prediction_model')
prediction_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer='rmsprop')
#%%
Y = np.zeros(shape=(q_sequences.shape[0],))
training_model.fit([q_sequences, a_sequences, an_sequences],Y,epochs=1,batch_size=64,validation_data=[dev_q_sequences,dev_a_sequences,dev_an_sequences],verbose=1)
#%%
a= b


