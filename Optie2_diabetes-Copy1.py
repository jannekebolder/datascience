#!/usr/bin/env python
# coding: utf-8

# In[330]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import streamlit as st


# In[331]:


# Titel en subtitel van de app

st.title('Diabetes detectie')
st.header('Kijk of u diabetes heeft met behulp van deze app')

# Teskst behorende bij de app

st.write('Diabetes is een chronische stofwisselingsziekte. Bij diabetes zit er te veel suiker in het bloed. Het lichaam kan de bloedsuiker niet op peil houden. Met behulp van deze machine learning web app wordt het mogelijk om aan de hand van ingevoerde parameters een diagnose te maken over de mogelijkheid dat iemand diabetes heeft.De app maakt gebruik van historische data om de kans op diabetes te calculeren. Dit kan mensen helpen om betere en snellere diagnoses te maken of mensen helpen die geen tijd of geld hebben om een doctor te bezoeken.')

#Voeg afbeelding toe > werkt alleen als de persoon die het upload naar streamlit zelf de afbeelding opslaat en filepath noteerd naar de afbeelding
#Image = Image.open("C:\Users\joshua.bierenbrood\Documents\Data Science\Intro to datascience\Werkcollege week 3\diabetes.jpg")
#st.image(image, caption = 'ML', use_column_width = True)


# In[332]:


df = pd.read_csv('diabetes.csv')


# In[333]:


st.title("Diabetes Database")


# In[334]:


df.head()


# In[335]:


df.info()


# In[336]:


df.eq(0).sum()
#veel 0 waarden, vervangen door NaN en dan opvullen met gemiddelden


# In[337]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI',
    'DiabetesPedigreeFunction','Age']]= df[['Glucose','BloodPressure',
    'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)


# In[338]:


df.fillna(df.mean(), inplace=True)


# In[339]:


df.eq(0).sum() #checken


# In[340]:


df.describe()


# In[341]:


st.subheader('Informatie over de data:')
st.dataframe(df)


# In[342]:


st.title("Correlatie tussen de verschillende variabelen")
st.subheader('In de onderstaande heatmap is te zien hoe de variabelen van elkaar afhangen.')
fig, ax = plt.subplots()
sns.heatmap(df.corr(), ax=ax, annot=True)
st.write(fig)


# In[343]:


st.title('Korte vraag tussendoor')
option = st.selectbox(
     'Hoe bent u bij deze app gekomen?',
     ('Google', 'Mail', 'Link'))

st.write('U bent hier gekomen via', option)


# In[344]:


#Machine learning
x_data= df.drop('Outcome',axis=1)
y_data= df['Outcome']


# In[345]:


x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.3, random_state=42)


# In[346]:


def get_input():
    pregnancies = st.sidebar.slider('Zwangerschappen', 0, 17,0)
    glucose = st.sidebar.slider('Glucose', 44, 199, 44)
    blood_pressure = st.sidebar.slider('Bloeddruk',24, 122,24)
    skin_thickness = st.sidebar.slider('Huiddikte', 7, 99, 7)
    insulin = st.sidebar.slider('Insuline', 14.0, 846.0, 14.0)
    BMI = st.sidebar.slider('BMI', 18.2, 67.1, 18.2)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.078)
    age = st.sidebar.slider('Leeftijd', 21, 81, 21)
    
    user_data = {'Zwangerschappen': pregnancies,
                 'Glucose': glucose,
                 'Bloeddruk': blood_pressure,
                 'Huiddikte': skin_thickness,
                 'Insuline': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'Leeftijd': age
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features

gekozen_input = get_input()

st.subheader('Doe zelf de test of u diabetes heeft!')
st.write('Selecteer de voor u geldende parameters in de sliderboxen aan de linkerkant om de app te laten bepalen of u diabetes heeft.')
                        
st.subheader('Uw gekozen waarden:')
st.write(gekozen_input)


# In[347]:


RandomForest=RandomForestClassifier()
RandomForest.fit(x_train, y_train)
Predict= RandomForest.predict(x_test)
print(Predict)


# In[348]:


accuracyRFC = accuracy_score(y_test, Predict)
print("Accuracy with Random Forrest Classification:", accuracyRFC)


# In[349]:


st.subheader('Accuratiescore bij het model')
st.write(str(accuracy_score(y_test, Predict) * 100) + '%')


# In[350]:


diabetes_ja_nee = RandomForest.predict(gekozen_input)


# In[351]:


st.subheader('Wel of geen diabetes?')
st.write("Wanneer de uitslag 1 is, heeft u wel diabetes. Wanneer de uitslag 0 is, heeft u geen diabetes.")
st.write(diabetes_ja_nee)


# In[352]:


st.subheader('Heeft u volgens de test diabetes?')
ja = st.checkbox('Ja')
nee = st.checkbox('Nee')

if ja:
     st.write('Wat vervelend voor u..')
if nee:
    st.write('Wat fijn voor u!')

