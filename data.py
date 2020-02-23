# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

td=pd.read_csv('train_values.csv')
testdata=pd.read_csv('test_values.csv')
trlab=pd.read_csv('train_labels.csv')


x=td.iloc[:,1:14]
y=trlab.iloc[:,[1]]


x=x.drop('thalach',axis=1).values
'''
x=np.array(x)
from sklearn.manifold import TSNE
x_embedded=TSNE().fit_transform(x)
x_embedded.shape
'''

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()                
x[:,1]=label.fit_transform(x[:,1])


print(len(x))
cnt=0
print(y)
for i in range(len(y)):
    if y==1:
        cnt=cnt+1
        
for i in np.nditer(y):
    if i==0:
        cnt=cnt+1
print(cnt)
'''from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
xx= onehotencoder.fit_transform(x).toarray()
xxx=pd.DataFrame(x)'''
svm_score=[]
log_scor=[]
for_score=[]
svm_ind=[]
log_ind=[]
for_ind=[]

min_svm_i=0
min_eff=1

max_svm_i=0
max_eff=0
for i in range(100):
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.15, random_state = 89)
    
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    
    scaled_x=scaler.fit_transform(xtrain)
    scaled_test=scaler.transform(xtest)
    
    from sklearn.svm import SVC
    classifier=SVC(kernel='linear',gamma='scale')
    classifier.fit(xtrain,ytrain)
    
    ypred=classifier.predict(xtest)
    
    print("score train : "+ str(classifier.score(xtrain,ytrain))+" test : "+str(classifier.score(xtest,ytest)))
    svm_score.append(classifier.score(xtest,ytest))
    svm_ind.append(i)   
    
    if min_eff > classifier.score(xtest,ytest):
        min_eff=classifier.score(xtest,ytest)
        min_svm_i=i
    if max_eff<classifier.score(xtest,ytest):
        max_eff=classifier.score(xtest,ytest)
        max_svm_i=i
    
        
    
    
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(ytest,ypred)
    
    from sklearn.linear_model import LogisticRegression
    cla_logis=LogisticRegression(random_state=0)
    cla_logis.fit(xtrain,ytrain)
    
    print("score train : "+ str(cla_logis.score(xtrain,ytrain))+" test : "+str(cla_logis.score(xtest,ytest)))
    log_scor.append(cla_logis.score(xtest,ytest))
    log_ind.append(i)    
    
    from sklearn.ensemble import RandomForestClassifier
    for_clf=RandomForestClassifier()
    for_clf.fit(scaled_x,ytrain)
    
    
    print("score train : "+ str(for_clf.score(xtrain,ytrain))+" test : "+str(for_clf.score(xtest,ytest)))
    for_score.append(for_clf.score(xtest,ytest))
    for_ind.append(i)    
    print("---------------------------------------------------------------------")
plt.figure(figsize=(20,10))
plt.plot(svm_ind,svm_score,'r')

plt.plot(svm_ind,log_scor,'b')

plt.plot(svm_ind,for_score,'g')
plt.show()

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('acc_graf.png', dpi=100)

from sklearn.metrics import log_loss
log_loss(ytest,classifier.predict(xtest))


classifier.predict(xtest)


add=0
for i in range(len(log_scor)):
    add=add+log_scor[i]
print(add/len(log_scor))
    

from sklearn.model_selection import train_test_split
svm_xtr, svm_xte, svm_ytr, svm_yte = train_test_split(x, y, test_size = 0.15, random_state = min_svm_i)


from sklearn.model_selection import train_test_split
max_svm_xtr, max_svm_xte, max_svm_ytr, max_svm_yte = train_test_split(x, y, test_size = 0.15, random_state = max_svm_i)
    

from sklearn.metrics import log_loss
log_loss(max_svm_yte,classifier.predict(max_svm_xte))





    
'''from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(14, input_dim=13, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
history = model.fit(scaled_x, ytrain, nb_epoch=50,  verbose=1)
# Print Accuracy
test_loss, test_acc = model.evaluate(xtest, ytest)

print('Test accuracy:', test_acc)
'''
