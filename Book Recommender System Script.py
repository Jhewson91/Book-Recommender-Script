
# In[1]:


import random
import difflib
import sklearn
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import Reader
from surprise import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import Book data set


book_data = pd.read_csv('C:/Users/jhews/OneDrive/Documents/Data Analytics/Portfolio/Project 5 - Recommender System/Books Dataset/BX-Books.csv',
                        error_bad_lines=False, encoding='latin-1', sep=';')


# In[6]:


book_data.head(10)


# In[7]:


print(book_data.shape)
book_data.info()


# In[8]:


# Import Book User dataset


# In[9]:


user_data = pd.read_csv('C:/Users/jhews/OneDrive/Documents/Data Analytics/Portfolio/Project 5 - Recommender System/Books Dataset/BX-Users.csv',
                        error_bad_lines=False, encoding='latin-1', sep=';')


# In[10]:


user_data.head(10)


# In[11]:


print(user_data.shape)
user_data.info()


# In[12]:


# Import Rating data set


# In[13]:


rating_data = pd.read_csv('C:/Users/jhews/OneDrive/Documents/Data Analytics/Portfolio/Project 5 - Recommender System/Books Dataset/BX-Book-Ratings.csv',
                          error_bad_lines=False, encoding='latin-1', sep=';')


# In[14]:


rating_data.head(10)


# In[15]:


print(rating_data.shape)
rating_data.info()


# In[16]:


# merge rating_data with book_data on common column ISBN


# In[17]:


book_rating_merge = pd.merge(rating_data, book_data, on='ISBN')


# In[18]:


print(book_rating_merge.info())


# In[19]:


# drop irrelivant columns


# In[20]:


irrelavant_columns = ['Book-Author', 'Year-Of-Publication',
                      'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']


# In[21]:


book_rating_merge = book_rating_merge.drop(irrelavant_columns, axis=1)


# In[22]:


print(book_rating_merge.info())
book_rating_merge.head(10)


# In[23]:


# Preliminary visualization of Columns


# In[24]:


# "Book-Rating"
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(book_rating_merge['Book-Rating'])
plt.xlabel("Book-Rating")
plt.ylabel("Frequency")
plt.title("Book-Rating")
plt.show()


# In[25]:


# create a surprise dataset object

# In[28]:


surprise_reader = Reader(rating_scale=(1, 5))


# In[29]:


surprise_data = Dataset.load_from_df(
    book_rating_merge[['User-ID', 'ISBN', 'Book-Rating']], surprise_reader)


# In[30]:


# In[31]:


# build a singular value decomposition (SVD) model and cross validate it


# In[32]:


svd = SVD(verbose=True, n_epochs=10)
cross_validate(svd, surprise_data, measures=[
               'RMSE', 'MAE'], cv=3, verbose=True)


# In[33]:


# Tune Hyperparamters


# In[34]:


# In[35]:


param_grid = {
    'n_factors': [20, 50, 100],
    'n_epochs': [5, 10, 20]
}


# In[36]:


grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=10)
grid_search.fit(surprise_data)


# In[37]:


print(grid_search.best_score['rmse'])
print(grid_search.best_params['rmse'])


# In[38]:


# best hyperparameters
best_factor = grid_search.best_params['rmse']['n_factors']
best_epoch = grid_search.best_params['rmse']['n_epochs']
# Apply best paramters to SVD model
svd = SVD(verbose=True, n_factors=best_factor, n_epochs=best_epoch)


# In[39]:


# Train the above model on the entire dataset


# In[40]:


trainset = surprise_data.build_full_trainset()
svd.fit(trainset)


# In[41]:


# Create Predictions


# In[42]:


svd.predict(uid=1000, iid=1234)


# In[43]:


# In[47]:


def get_book_id(book_title, metadata):

    existing_titles = list(metadata['Book-Title'].values)
    closest_titles = difflib.get_close_matches(book_title, existing_titles)
    book_id = metadata[metadata['Book-Title']
                       == closest_titles[0]]['ISBN'].values[0]
    return book_id


def get_book_info(book_id, metadata):

    book_info = metadata[metadata['ISBN'] == book_id][['ISBN',
                                                       'Book-Author', 'Book-Title']]
    return book_info.to_dict(orient='records')


def predict_review(user_id, book_title, model, metadata):

    book_id = get_book_id(book_title, metadata)
    review_prediction = model.predict(uid=user_id, iid=book_id)
    return review_prediction.est


def generate_recommendation(user_id, model, metadata, thresh=4):

    book_titles = list(metadata['Book-Title'].values)
    random.shuffle(book_titles)

    for book_title in book_titles:
        rating = predict_review(user_id, book_title, model, metadata)
        if rating >= thresh:
            book_id = get_book_id(book_title, metadata)
            return get_book_info(book_id, metadata)


# In[46]:


# generate recommendation using parameters: "user_id",(the user) "svd" (model), and "metadata" (book_data)


# In[48]:


generate_recommendation(100, svd, book_data)


# In[ ]:
