import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
import pandas as pd
import wordcloud
data = pd.read_excel('participants.xlsx')

from sklearn.feature_extraction.text import CountVectorizer
objectives = data['Objective / Goal (s)'].fillna('')
cv = CountVectorizer(stop_words='english', ngram_range=(1, 3), binary=True).fit(objectives)
counts = pd.DataFrame(cv.transform(objectives).toarray() / len(objectives), columns=['-'.join(feature.split(' ')) for feature in cv.get_feature_names()])
counts = counts.sum().sort_values(ascending=True)
counts = counts[counts > 0.1]
_ = counts.plot.barh()
