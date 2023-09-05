import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
import pandas as pd
import wordcloud
data = pd.read_excel('participants.xlsx')

from sklearn.feature_extraction.text import CountVectorizer
wc = wordcloud.WordCloud(width=1200, height=400, mode='RGBA', colormap='seismic', background_color='lightyellow').fit_words(counts.to_dict())
wc.to_image()
