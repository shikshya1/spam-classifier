import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


#Dataset link -  https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
sms = pd.read_csv('data/SMSSpamCollection', sep='\t', names=["label", "message"])

#Data cleaning and preprocessing

wordnet=WordNetLemmatizer()
corpus = []
for i in range(len(sms)):
    review = re.sub('[^a-zA-Z]', ' ', sms['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Tf-idf


from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()


X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(sms['label'])
y=y.iloc[:,1].values



# 80-20 Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


confusion_m=confusion_matrix(y_test, y_pred)
acc=accuracy_score( y_test, y_pred)

print(acc)

print (classification_report(y_test, y_pred) )















