#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[10]:


dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")


# In[11]:


dataFrame.head()


# In[12]:


import seaborn as sbn


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


sbn.pairplot(dataFrame)


# In[15]:


###veriyi test/train olarak ikiye ayırmak


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


#train_test_split


# In[18]:


dataFrame


# In[19]:


y= dataFrame["Fiyat"].values


# In[20]:


x= dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values


# In[21]:


x_train, x_test, y_train,y_test =train_test_split(x,y,test_size=0.33, random_state=15)


# In[22]:


x_train.shape


# In[23]:


y_train.shape


# In[24]:


y_test.shape


# In[25]:


x_test.shape


# In[26]:


x_test


# In[27]:


#scaling: 0 ile 1 arasi


# In[28]:


from sklearn.preprocessing import MinMaxScaler


# In[29]:


scaler=MinMaxScaler()


# In[30]:


scaler.fit(x_train)


# In[31]:


x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)


# In[32]:



x_train


# In[33]:


import tensorflow as tf


# In[34]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[35]:


model= Sequential()


# In[36]:


model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))


# In[37]:


model.add(Dense(1))


# In[38]:


model.compile(optimizer="rmsprop",loss="mse")


# In[39]:


model.fit(x_train,y_train, epochs=250)


# In[44]:


loss= model.history.history["loss"]


# In[45]:


sbn.lineplot(x=range(len(loss)),y=loss)


# In[48]:


trainLoss= model.evaluate(x_train,y_train,verbose=0)


# In[49]:


testLoss= model.evaluate(x_test,y_test,verbose=0)


# In[50]:


trainLoss


# In[51]:


testLoss


# In[52]:


testTahminleri= model.predict(x_test)


# In[53]:


testTahminleri


# In[54]:


tahminDf= pd.DataFrame(y_test,columns=["Gerçek Y"])


# In[55]:


tahminDf


# In[56]:


testTahminleri= pd.Series(testTahminleri.reshape(330,))


# In[57]:


testTahminleri


# In[58]:


tahminDf= pd.concat([tahminDf,testTahminleri],axis=1)


# In[59]:


tahminDf


# In[60]:


tahminDf.columns= ["Gerçek Y", "Tahmin Y"]


# In[61]:


tahminDf


# In[62]:


sbn.scatterplot(x="Gerçek Y", y="Tahmin Y", data=tahminDf)


# In[63]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[64]:


mean_absolute_error(tahminDf["Gerçek Y"],tahminDf["Tahmin Y"])


# In[65]:


dataFrame.describe()


# In[70]:


yeniBisikletOzellikleri=[[1750,1749]]


# In[71]:


yeniBisikletOzellikleri =scaler.transform(yeniBisikletOzellikleri)


# In[72]:


model.predict(yeniBisikletOzellikleri)


# In[73]:


from tensorflow.keras.models import load_model


# In[74]:


model.save("bisiklet_modeli.h5")


# In[75]:


sonraCagirilanModel=load_model("bisiklet_modeli.h5")


# In[77]:


sonraCagirilanModel.predict(yeniBisikletOzellikleri)


# In[ ]:




