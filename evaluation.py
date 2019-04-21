from movie_recommendations import recommendations
from bs4 import BeautifulSoup
import urllib.request
from imdb import IMDb
import pandas as pd

ia = IMDb()

df = pd.read_csv('data/IMDB_Top250Engmovies2_OMDB_Detailed.csv')
df = df[['Title','imdbID']]
df.set_index('Title', inplace = True)
indices = pd.Series(df.index)

# Convert list of movie_names to imdbIDs
def list_of_imdbIDs(B):
    # List for indexes
    list_idx = []
    for title in B:
        idx = indices[indices == title].index[0]
        list_idx.append(idx)

    # Get the list of imdbIDs from above indexes list
    list_imdbIDs = []
    df.set_index('imdbID', inplace = True)
    for idx in list_idx:
        list_imdbIDs.append(list(df.index)[idx])

    return list_imdbIDs

# Check if the movie_id of the recommended movie crawled from IMDb
# is actually present in our local database
def check(mid):
    idx = []
    indices = pd.Series(df.index)
    # Condition to match mid with our database indices list
    idx = indices[indices == mid].index[0]
    if idx.size > 0:
        return True
    return False

#Caluclating precision for two sets A and B
def precision(A,B):
    # Count is relevant movies out of retrieved movies
    count = 0
    # Total is complete retrieved movies which is exactly equal to 10
    # As we are recommending exactly 10 movies as of now
    total = 10

    #count is the number of movies which are in B and A
    for i in A:
        if i in B:
            count = count + 1
    #print(count)
    return count/float(total)

#Caluclating recall for two sets A and B
def recall(A,B):
    # Check if the movie_id of the recommended movie crawled from IMDb
    # is actually present in our local database. So C is that movies list
    C = []

    # total is the total count of relevant movies
    total = 0
    # Count is relevant movies out of retrieved movies
    count = 0

    # total is the number of movies which are in B and database
    for i in B:
        if check(i):
            C = C + [i]
            total+=1

    # If none of the movies in B are in database
    # return recall as 0
    if total == 0:
        return 0

    #count is the number of movies which are in A,B and database
    for i in A:
        if i in C:
            count = count + 1

    return count/float(total)

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

def evaluate_model(movie_name, movie_id):
    A = []
    B = []
    A = crawled(movie_name, movie_id)
    B = recommendations(movie_name)
    B = list_of_imdbIDs(B)
    print('Precision: ', precision(A,B))
    print('Recall: ', recall(A,B))
    
evaluate_model('Interstellar', 'tt0816692')