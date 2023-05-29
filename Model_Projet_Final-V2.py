#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install -U scikit-learn
#!pip install gensim


# In[194]:


import numpy as np # linear algebra
import docx2txt
import pickle
import re, os
import string
import pandas as pd
import seaborn as sns
import nltk
import warnings
import docx2txt
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from wordcloud import WordCloud
from wordcloud import WordCloud ,STOPWORDS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
import gensim.downloader as api
from gensim.utils import simple_preprocess
from sklearn.svm import SVC
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")


# In[195]:


#Data
df = pd.read_csv('Resume.csv')
# create list of all categories
categories = np.sort(df['Category'].unique())
# lenght
df['length_str'] = df['Resume_str'].str.len()


# In[196]:


#Feature Engineering
#Cleaning
def clean_text(text):
       
    text = text.lower() # lowercase text
    text = text.replace('\d+', '') # remove digits
    text = re.compile('[/(){}\[\]\|@,;]').sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = re.compile('[^0-9a-z #+_]').sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in set(stopwords.words('english'))) # remove stopwors from text
    
    
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]',r' ', text) #remplacer tous les caractères non ASCII
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    
    # remove non-english characters, punctuation and numbers
    text = re.sub('[^a-zA-Z]', ' ', text) 
    # tokenize word
    text = nltk.tokenize.word_tokenize(text) 
    # remove stop words
    text = [w for w in text if not w in nltk.corpus.stopwords.words('english')]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(text) for text in text]

    return ' '.join(text)

def filter_text(text): 
    text = ' '.join(word for word in text.split() if word not in set(stopwords.words('english')).union(['state','summary', 'city','company', 'name', 'skill'])) # remove stopwors from text
    return text

#Encode the labels into numeric

def set_code(row):
    if row["Category"] == "ACCOUNTANT": return 1
    elif row["Category"] == "ADVOCATE": return 2
    elif row["Category"] == "AGRICULTURE": return 3
    elif row["Category"] == "APPAREL": return 4
    elif row["Category"] == "ARTS": return 5
    elif row["Category"] == "AUTOMOBILE": return 6
    elif row["Category"] == "AVIATION": return 7
    elif row["Category"] == "BANKING": return 8
    elif row["Category"] == "BPO": return 9
    elif row["Category"] == "BUSINESS-DEVELOPMENT": return 10
    elif row["Category"] == "CHEF": return 11
    elif row["Category"] == "CONSTRUCTION": return 12
    elif row["Category"] == "CONSULTANT": return 13
    elif row["Category"] == "DESIGNER": return 14
    elif row["Category"] == "DIGITAL-MEDIA": return 15
    elif row["Category"] == "ENGINEERING": return 16
    elif row["Category"] == "FINANCE": return 17
    elif row["Category"] == "HEALTHCARE": return 18
    elif row["Category"] == "HR": return 19
    elif row["Category"] == "INFORMATION-TECHNOLOGY": return 20
    elif row["Category"] == "PUBLIC-RELATIONS": return 21
    elif row["Category"] == "SALES": return 22
    elif row["Category"] == "TEACHER": return 23
    elif row["Category"] == "FITNESS": return 24
    else: return 25
    
df = df.assign(target=df.apply(set_code, axis=1))  


# In[197]:


#cleaning
df['Resume_Clean'] = df['Resume_str'].apply(clean_text)
df['length_Clean'] = df['Resume_Clean'].str.len()
df['Resume_filtered'] = df['Resume_Clean'].apply(lambda w: filter_text(w))
df['length_filtered'] = df['Resume_filtered'].str.len()


# In[198]:


#Balance
def upsample_classes(data, target):
    
    lst = list(data[target].unique())
    
    classes = []
    for c in lst:
        classes.append(data[data[target]==c])
    
    length = 0
    class_lab = None
    for c in classes:
        if len(c)>length:
            length=len(c)
            class_lab = c
    class_lab = class_lab[target].unique()[0]
    
    regroup = pd.concat(classes)
    maj_class = regroup[regroup[target]==class_lab]

    lst.remove(class_lab)
    
    new_classes=[]
    for i in lst:
        new_classes.append(resample(data[data[target]==i],replace=True, n_samples=len(maj_class)))

    minority_classes = pd.concat(new_classes)
    upsample = pd.concat([regroup[regroup[target]==class_lab],minority_classes])

    return upsample

df_balanced = (upsample_classes(df,'target'))
#split
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Resume_filtered'], df_balanced['target'], test_size = 0.2, random_state=8)


# Bag Of Words
# vectorize text data
vectorizer = CountVectorizer()
conuntvectorizer_train = vectorizer.fit_transform(X_train).astype(float)
conuntvectorizer_test = vectorizer.transform(X_test).astype(float)


# In[13]:


# liste des modèles que vous souhaitez entraîner
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

models = [
    
    {
        'name': 'GradientBoostingClassifier',
        'model': OneVsRestClassifier(GradientBoostingClassifier()),
        'params': {
            'estimator__learning_rate': [0.05, 0.1, 0.5],
            'estimator__n_estimators': [10, 30,100],
            'estimator__max_depth': [1, 3, 5]
        }
    }
]

# Parcourez la liste des modèles et entraînez-les avec différents hyperparamètres
for model in models:
    print(f"Training {model['name']}...")
    clf = GridSearchCV(model['model'], model['params'], cv=5)
    clf.fit(conuntvectorizer_train,y_train)
    print(f"Best parameters for {model['name']}: {clf.best_params_}")
    
# Calcul de l'accuracy score sur l'ensemble d'entraînement
accuracy = clf.score(conuntvectorizer_train,y_train)

print("GradientBoostingClassifier Accuracy Score on Training Set -> ", accuracy * 100)

# Calcul de l'accuracy score sur l'ensemble de test
accuracy_test = clf.score(conuntvectorizer_test, y_test)
print(f"Accuracy Score on Test Set for {model['name']}: {accuracy_test * 100}")


# In[199]:


# Diviser les données en ensembles d'entraînement, de validation et de test
X_train_1, X_val, y_train_1, y_val = train_test_split(conuntvectorizer_train, y_train, test_size=0.2, random_state=42)

# Initialiser le modèle
gb = OneVsRestClassifier(GradientBoostingClassifier())

# Spécifier les paramètres à tester avec GridSearchCV
params = {
    'estimator__learning_rate': [0.05],
    'estimator__max_depth': [3],
    'estimator__n_estimators': [100]
}

# Initialiser GridSearchCV
clf = GridSearchCV(gb, params)


# Entraîner le modèle avec les meilleurs paramètres sur l'ensemble d'entraînement
clf.fit(X_train_1, y_train_1)

# Évaluer les performances sur l'ensemble de validation
accuracy_val = clf.score(X_val, y_val)
print("Accuracy Score on Validation Set:", accuracy_val)

# Une fois que vous êtes satisfait des performances sur l'ensemble de validation,
# vous pouvez utiliser l'ensemble d'entraînement complet pour entraîner le modèle final

# Entraîner le modèle final avec l'ensemble d'entraînement complet
clf.fit(conuntvectorizer_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = clf.predict(conuntvectorizer_test)

# Calculer l'accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Score on Test Set -> ", accuracy * 100)


# In[220]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[225]:


new_data


# In[224]:


# Prédiction sur de nouvelles données
new_data = docx2txt.process('python-sample-resume.docx')
#new_data = df.iloc[288].Resume_str

# Preprocess the text
# Preprocess the text
new_data = clean_text(new_data)
new_data = filter_text(new_data)

# Create a new CountVectorizer with the same vocabulary as the original vectorizer
new_vectorizer = CountVectorizer(vocabulary=vectorizer.vocabulary_)

# Vectorize the preprocessed text using the new vectorizer
features = new_vectorizer.transform([new_data])

# Ensure the number of features matches the expected number
if features.shape[1] != 33645:
    # Pad the features with zeros to match the expected number of features
    padding = scipy.sparse.csr_matrix((features.shape[0], 33645 - features.shape[1]))
    features = scipy.sparse.hstack([features, padding])

# Make the prediction
sentiment = clf.predict(features)[0]

print("Predicted Sentiment:", sentiment)

# Décodage des prédictions
# Map the predicted sentiment to the real value
reverse_mapping = { 1: "ACCOUNTANT", 2: "ADVOCATE", 3: "AGRICULTURE", 4: "APPAREL",
    5: "ARTS", 6: "AUTOMOBILE", 7: "AVIATION", 8: "BANKING",
    9: "BPO", 10: "BUSINESS-DEVELOPMENT", 11: "CHEF", 12: "CONSTRUCTION",
    13: "CONSULTANT", 14: "DESIGNER", 15: "DIGITAL-MEDIA", 16: "ENGINEERING",
    17: "FINANCE", 18: "HEALTHCARE", 19: "HR", 20: "INFORMATION-TECHNOLOGY",
    21: "PUBLIC-RELATIONS", 22: "SALES", 23: "TEACHER", 24: "FITNESS", 25: "UNKNOWN"}

# Make the prediction
sentiment = clf.predict(features)[0]
# Map the predicted sentiment to the category name
predicted_category = reverse_mapping.get(sentiment, "UNKNOWN")
print("Predicted Category:", predicted_category)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Définition du pipeline
pipeline = Pipeline([
    ('clean', clean_text),
    ('filter', filter_text),
    ('vectorizer', CountVectorizer()),
    ('classifier', clf)
])

# Prédiction sur de nouvelles données
#new_data = ["Loren Brekke \n\n9873"]
new_data = [docx2txt.process('python-sample-resume.docx')]
# Convert the list to a string
new_data = ' '.join(new_data)
# Prédiction sur de nouvelles données
encoded_predictions = pipeline.predict([new_data])


# Décodage des prédictions
decoded_predictions = [set_code.inverse_transform([prediction])[0] for prediction in encoded_predictions]

print("Predictions:", decoded_predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Resume_Scanner_for_Job_Description


# In[171]:


# Télécharger le modèle Word2Vec pré-entraîné
model_name = "word2vec-google-news-300"
model = api.load(model_name)
warnings.filterwarnings("ignore")


# In[226]:


# Fonction pour calculer la similarité cosinus entre une description de poste et une liste de CV
def compute_similarity(job_description, resume_list, model):
    similarity_scores = []
    
    for resume in resume_list:
        # Tokeniser la description de poste et le CV en mots
        job_words = job_description.lower().split()
        resume_words = resume.lower().split()
        
        # Vérifier que les deux listes ne sont pas vides
        if len(job_words) == 0 or len(resume_words) == 0:
            similarity_scores.append(0.0)  # Assigner une similarité de 0 en cas de liste vide
        else:
            # Calculer la similarité cosinus moyenne entre les mots des deux textes
            similarity_score = model.n_similarity(job_words, resume_words)
            similarity_scores.append(similarity_score)
    
    return similarity_scores

# Liste de CV
resume_list = df['Resume_filtered'].tolist() 

# Exemple d'utilisation
job_description = docx2txt.process('python-job-description.docx')

# Preprocess the text
job_description = clean_text(job_description)
job_description = filter_text(job_description)

# Calculate the cosine similarity between the job description and all the resumes
similarities = compute_similarity(job_description, resume_list, model)

# Add a column with the similarity values to the DataFrame
df['Similarity'] = similarities

# Sort the resumes based on their similarity (from most similar to least similar)
df_sorted = df.sort_values(by='Similarity', ascending=False)

# Select the top 10 most similar resumes
top_10_cv = df_sorted.loc[:, ['ID', 'Resume_str', 'Category', 'Similarity']].head(10)

# Display the top 10 CVs
print(top_10_cv)
print("       --------------------------------------------------------------------------------------")
print("       --------------------------------------------------------------------------------------")
print("       --------------------------------------------------------------------------------------")

# Afficher les 10 meilleurs CV avec leur ID correspondant
for index, row in top_10_cv.iterrows():
    cv_id = row['ID']
    cv_similarity = row['Similarity']
    cv_text = correspondance[cv_id]
    print(f"CV ID: {cv_id}")
    print(f"Similarity: {cv_similarity}")
    print(f"CV Text: {cv_text}")
    print("       --------------------------------------------------------------------------------------")
    print("       --------------------------------------------------------------------------------------")
    print("       --------------------------------------------------------------------------------------")
    print("       --------------------------------------------------------------------------------------")
    print("       --------------------------------------------------------------------------------------")
    print("       --------------------------------------------------------------------------------------")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




