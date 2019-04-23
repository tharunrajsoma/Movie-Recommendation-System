# Movie-Recommendation-System
Content based movie recommendation system with NLP

## Before you start


## Recommendation System 

It is a system that seeks to predict or filter preferences according to the user’s choices. Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general.

There are two types of recommendation systems
### Collaborative filtering
- Collaborative filtering approaches build a model from user’s past behavior (i.e. items purchased or searched by the user) as well as similar decisions made by other users.
- This model is then used to predict items (or ratings for items) that user may have an interest in.

### Content-based filtering
- Content-based filtering approaches uses a series of discrete characteristics of an item in order to recommend additional items with similar properties.
- Content-based filtering methods are totally based on a description of the item and a profile of the user’s preferences. It recommends items based on user’s past preferences.

## Cosine Similarity
- Cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them. 
- The dot product is important when defining the similarity, as it is directly connected to it. The definition of similarity between two vectors u and v is, in fact, the ratio between their dot product and the product of their magnitudes.

<p align = 'center'>
  <img src = '/images/cosine_similarity.png'>
</p>


## Code Description

### Gather the Data

The dataset used for this recommendation systems contains top 250 top rated movies from IMDb. 

However only movie director, actors, genre and plot were considered for modeling because content based recommendations mostly depend on these features.

```
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('data/IMDB_Top250Engmovies2_OMDB_Detailed.csv')

df = df[['Title','Genre','Director','Actors','Plot']]
```

### Data Cleaning

- There is a awesome NLTK package that allows to extract keywords from a text, and it even assigns scores to each word.
- From the 'rake_nltk' package only RAKE module/function was imported to extract key words from the Plot column. Here instead of using the entire sentences describing the plot just considered the most relevant words in the description.
- To achieve this I applied this function to each row under the Plot column and assigned the list of key words to a new column, named 'Key_words'.

```
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
```

### Modeling

- To detect similarities between vectorization is needed. So for this I used 'CountVectorizer' which is performs well in our case because it gives importance to words that are more present in the entire corpus (Plot column in our case).
- Once the matrix containing the count for each word is ready we can apply the 'Cosing Similarity' function.

```
# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['words_bag'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
```

Now we can write the actual function to return top 10 recommended movies to a movie given as input.
- For doing this first get the index of the movie that matches the title.
- Then get the similarity scores of this input movie with all the movies present in our database.
- Now get the top 10 movies among having higher similarity scores among the list. Store just the iindexes of those movies.
- Finally suitable conversion from index to movie names list is also done. 

```
# creating a Series for the movie titles so that they are associated to an ordered numerical list 
# which we will use in the function to match the indexes
indices = pd.Series(df.index)

# Function that takes in movie title as input and can return top 10 movies similar to this based on content 
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
```

### Testing

Congratulations content based movie recommendation system using NLP is successfully done.

To test the functionality you can simply replace the title 'Interstellar' and give your own movie which is present in the database as an input.

### Evaluation

- Inorder to evaluate the model we need a proper metrics. So used the IMDb site itself as the evaluation metric to our problem.

- For a given movie got the top 10 recommended movies from IMDb site itself using the 'More Like this' field present in the site. 

- Inorder to scrape the 'More Like this' field data, I used Beautiful Soup (HTML parser) library.
```
# Finding 10 similar movies from imdbby crawling imdb site
# To check for rec_item which means recommended item
def crawled(m_name, m_id):
    B=[]
    
    # Crawling to the movie imdb website
    # Beautiful scoup for web scrapping and finding our 'rec_item'
    soup=BeautifulSoup(urllib.request.urlopen("http://www.imdb.com/title/"+m_id+"/"), "lxml")
    mydivs = soup.find_all("div", {"class": "rec_item"})
    
    i=1
    #appending 10 similar movies to A
    for d in mydivs:
        B.append(d["data-tconst"])
        
        # Need exactly 10 recommended movies from IMDb
        if(i==10):
            break
        i=i+1

    return B
 ```

- I have used the Precision and Recall methods to evaluate this model by taking recommended movies through my model and through IMDb as the input sets.

<p align = 'center'>
  <img src = '/images/precision_recall.png'>
</p>
