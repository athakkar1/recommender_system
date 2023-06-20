import numpy as np
import pandas as pd
from  numpy.linalg import matrix_rank as rank
import math
def compute_slice(U, S, VT, r):
    Xapprox = U[:,:r]@S[:r,:r]@VT[:r,:]
    return Xapprox

def top_user_similarity(data, user_id):
    similarity = []
    for i in range(len(data)):
        ph = data[user_id - 1]
        xt = np.transpose(ph)
        curUser = data[i]
        sim = (np.dot(xt, curUser))/(np.linalg.norm(ph)*np.linalg.norm(curUser))
        if(np.isnan(sim)):
            sim = 0
        similarity.append(sim)
    negsim = [-x for x in similarity]
    similarity = np.argsort(negsim)
    user = data[similarity[1]]
    neguser = [-x for x in user]
    topusermovies = np.argsort(neguser)
    return topusermovies

def top_movie_similarity(data, movie_id, top_n=5):
    similarity = []
    for i in range(len(data[0])):
        ph = data[:, movie_id - 1]
        xt = np.transpose(ph)
        curMovie = data[:, i]
        sim = (np.dot(xt, curMovie))/(np.linalg.norm(ph)*np.linalg.norm(curMovie))
        if(np.isnan(sim)):
            sim = 0
        similarity.append(sim)
    negsim = [-x for x in similarity]
    similarity = np.argsort(negsim)
    return similarity

def print_similar_movies(movie_titles, top_indices):
    movienames = []
    for i in range(6):
        rownum = int(movie_titles[movie_titles['MovieID']==top_indices[i] + 1].index[0])
        movietitle = str(movie_titles.at[rownum, 'Title'])
        movienames.append((movietitle))
    
    print('Most Similar movies: ', movienames)

column_list_ratings = ["UserID", "MovieID", "Ratings","Timestamp"]
ratings_data  = pd.read_csv('ratings.dat',sep='::',names = column_list_ratings, engine='python')
column_list_movies = ["MovieID","Title","Genres"]
movies_data = pd.read_csv('movies.dat',sep = '::',names = column_list_movies, engine='python', encoding = 'latin-1')
column_list_users = ["UserID","Gender","Age","Occupation","Zixp-code"]
user_data = pd.read_csv("users.dat",sep = "::",names = column_list_users, engine='python')
data = pd.merge(pd.merge(ratings_data, user_data), movies_data)
ratings = np.zeros((len(user_data), movies_data.max()[0]))
print(ratings.shape)
batmanusers = [[],[]]
for i in range(len(ratings_data)):
    userid = (int)(ratings_data.at[ratings_data.index[i], 'UserID'])
    movieid = (int)(ratings_data.at[ratings_data.index[i], 'MovieID'])
    ratingnum = (ratings_data.at[ratings_data.index[i], 'Ratings']).astype('uint8')
    ratings[userid-1][movieid-1] = ratingnum
    if movieid == 1377 and len(batmanusers[0]) < 3:
        batmanusers[0].append(userid)
        batmanusers[1].append(ratingnum)
print("Shape of Ratings: ", ratings.shape)
print("Batman Returns Users and Reviews: ", batmanusers)
ratings[ratings == 0] = np.nan
means = np.nanmean(ratings, axis=0)
meanslist = np.ndarray.tolist(means)
ratings = ratings - meanslist
ratings[np.isnan(ratings)] = 0
std = np.std(ratings, axis=0)
stdlist = np.ndarray.tolist(std)
ratings = ratings/stdlist
ratings[np.isnan(ratings)] = 0
U, S, VT = np.linalg.svd(ratings)
S = np.diag(S)
print("U Shape: ", U.shape)
print("S Shape: ", S.shape)
print("VT Shape: ", VT.shape)
ranks = [100, 1000, 2000, 3000]
rank100 = compute_slice(U=U, S=S, VT=VT, r=ranks[0])
rank1000 = compute_slice(U=U, S=S, VT=VT, r=ranks[1])
rank2000 = compute_slice(U=U, S=S, VT=VT, r=ranks[2])
rank3000 = compute_slice(U=U, S=S, VT=VT, r=ranks[3])
print("Original Ratings: ", batmanusers[1])
rank100ratings = []
rank1000ratings = []
rank2000ratings = []
rank3000ratings = []
for i in range(3):
    rank100ratings.append(rank100[batmanusers[0][i]-1][1376] * stdlist[1376] + meanslist[1376])
    rank1000ratings.append(rank1000[batmanusers[0][i]-1][1376] * stdlist[1376] + meanslist[1376])
    rank2000ratings.append(rank2000[batmanusers[0][i]-1][1376] * stdlist[1376] + meanslist[1376])
    rank3000ratings.append(rank3000[batmanusers[0][i]-1][1376] * stdlist[1376] + meanslist[1376])
#print(ratings[9][1376] * stdlist[1376] + meanslist[1376])
print("Rank100 Ratings: ", rank100ratings)
print("Rank1000 Ratings: ", rank1000ratings)
print("Rank2000 Ratings: ", rank2000ratings)
print("Rank3000 Ratings: ", rank3000ratings)
print(top_movie_similarity(rank1000, 1377))
print_similar_movies(movies_data, top_movie_similarity(rank1000, 1377))
print(top_user_similarity(rank1000, 10))
print_similar_movies(movies_data, top_user_similarity(rank1000, 10))