#%% SQuAD data download
import json
with open('C:/Users/Young Hun Park/Downloads/KorQuAD_v1.0_train.json') as train_file:
    train_data=json.load(train_file)

#%%
train_data=train_data['data']
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
#%%
## extract context
contexts=[]
for context in data_df.context.unique():
    contexts.append(context)

context_df=pd.DataFrame(contexts)
#%%
print(context_df.isnull().values.any())


#%%
# extract sentences from paragraph
from nltk.tokenize import sent_tokenize
sentences=[]
for context in contexts:
    temp_X=sent_tokenize(context)
    for sentence in temp_X:
        sentences.append(sentence)


#%%
from gensim.models import word2vec
from konlpy.tag import Okt

twitter = Okt()

#텍스트를 한줄씩 처리합니다.
result = []
for line in sentences:
     #형태소 분석하기, 단어 기본형 사용
    malist = twitter.pos( line, norm=True, stem=True)
    r = []
    for word in malist:
         #Josa”, “Eomi”, “'Punctuation” 는 제외하고 처리
        if not word[1] in ["Josa","Eomi","Punctuation"]:
            r.append(word[0])
     #형태소 사이에 공백 " "  을 넣습니다. 그리고 양쪽 공백을 지웁니다.
    rl = (" ".join(r)).strip()
    result.append(rl)
    #print(rl)

 #형태소들을 별도의 파일로 저장 합니다.
with open("context.nlp",'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))

 #Word2Vec 모델 만들기
wData =word2vec.LineSentence("context.nlp")
wModel =word2vec.Word2Vec(wData, size=200, window=10, hs=1, min_count=2, sg=1)
print(wModel.wv.most_similar('베토벤'))
#%%
wModel.save('KorQuAD.model')

#%%
import gensim
model=gensim.models.Word2Vec.load('C:/Users/Young Hun Park/Desktop/python beginner/NLP/KorQuAD.model')
#%%
print(model.wv.most_similar('박지성'))

'''
#%%
import reㄴ

stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# clean special character

def cleanText(readData):
 
 
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
 
    return text
#%%
from konlpy.tag import Okt
okt=Okt()
tokenized_data=[]

for sentence in sentences:
    sentence=cleanText(sentence)
    temp_X=okt.morphs(sentence,stem=True)
    temp_X=[word for word in temp_X if not word in stopwords]
    tokenized_data.append(temp_X)
#%%
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
print('최대 길이:',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))

plt.hist([len(s) for s in tokenized_data],bins=50)
plt.xlabel('length of sentence')
plt.ylabel('number of samples')

model=Word2Vec(sentences=tokenized_data,size=150,window=5,min_count=10,workers=-1)
print(model.wv.vectors.shape)
print(model.wv.most_similar('베토벤'))
'''
