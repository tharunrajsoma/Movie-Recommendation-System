import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Select maximum number columns to display
pd.set_option('display.max_columns', 120)

# Top 250 movies database from IMDB
df = pd.read_csv('data/IMDB_Top250Engmovies2_OMDB_Detailed.csv')

# Select those features on which your recommendation should base on
df = df[['Title','Genre','Director','Actors','Plot']]

# I am considering only first 3 actors names by removing commas between actors' names
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])

# populating genres into a list of words
df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))

df['Director'] = df['Director'].map(lambda x: x.split(' '))

# merging together first and last name for each actor and director,
# With this it becomes one word and there will be no mix up between people having common first name
for index, row in df.iterrows():
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = ''.join(row['Director']).lower()


# initializing the new column
df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']
    
    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column for the corresponding movie
    row['Key_words'] = list(key_words_dict_scores.keys())

# dropping the Plot column
df.drop(columns = ['Plot'], inplace = True)
df.set_index('Title', inplace = True)


df['words_bag'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'Director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['words_bag'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'words_bag'], inplace = True)

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['words_bag'])


# creating a Series for the movie titles so that they are associated to an ordered numerical list 
# which we will use in the function to match the indexes
indices = pd.Series(df.index)

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function that takes in movie title as input and can
# return top 10 movies similar to this based on content 
# using NLP and Cosine Similarity techniques
def recommendations(title, cosine_sim = cosine_sim):
    
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies

 # Printing the list of top 10 recommended movies
def print_recommendations(recommended_movies):
    serial_no = 1
    print('\nList of recommended movies:')
    for movie in recommended_movies:
        print(serial_no,'.',movie)
        serial_no+=1  

recommended_movies = []
recommended_movies = recommendations('Interstellar')

# Printing recommended movies generated using our algorithm
if recommended_movies == []:
    print('No recommended movies found')
else:
    print_recommendations(recommended_movies)