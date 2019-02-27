import pandas as pd
import numpy as np

'''     Mean absolute error calculate   '''
def meanAbsoluteError(list):
    count=0
    for i in range(0,len(list)):
        count+=abs(list[i][0]-list[i][1])
    return count/len(list)

'''     Average of user   '''
def average_user(dict,id):
    liste = dict[id].values()
    liste = [value if value != 0 else np.nan for value in liste]
    average = np.nanmean(np.array(liste))
    return average

'''     Average of book   '''
def average_book(dict,book):
    total = 0
    count = 0
    for user in dict:
        user_rate = dict[user][book]
        if user_rate != 0:
            total += user_rate
            count += 1

    return total / count

'''     Correlation similarity function   '''
def correlation(list1, list2):
    return np.corrcoef(list1, list2)

'''     Unweighted KNN function   '''
def unweighted_knn(similarity_array,k,id):
    if(similarity_array.__len__()>=k):
        count=0
        for i in range(0,k):
            count+=similarity_array[i][2]
        return count/k
    else:
        return average_book(train_data_dict,id)

def weighted_knn(similarity_array,k,id):
    total=0
    count=0
    if (similarity_array.__len__() >= k):

        for i in range(len(similarity_array)):
            sim = similarity_array[i][1]
            rate = similarity_array[i][2]
            total += sim*rate
            count += sim

        if(sim!=0):
            return count / sim

    return average_book(train_data_dict, id)


'''     Sorting list for chosen item   '''
def takeSecond(elem):
    return elem[1]

'''     Cosine similariity function  '''
def cosine_similarity(a,b):
    return np.dot(a, b) / (np.linalg.norm(a)* np.linalg.norm(b))

'''     This function found anyone user and other users similarity    '''
def predict(id2 , book , rate):
    similarity_list= [(id, cosine_similarity(list(books_ratings_matrix[id2]), list(books_ratings_matrix[id])), train_data_dict[id][book]) for id in books_ratings_matrix if id != id2 and train_data_dict[id][book] != 0]
    similarity_list.sort(key=takeSecond)
    similarity_list.reverse()
    return similarity_list

'''                     ###        DATA READ AND SHAPE COLUMN       ###                                               '''

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
train = pd.read_csv('BXBookRatingsTrain1.csv', sep=';', error_bad_lines=False, encoding="latin-1")
train.columns = ['userID', 'ISBN', 'bookRating']
test_data = pd.read_csv('BXBookRatingsTest1.csv', sep=';', error_bad_lines=False, encoding="latin-1")
test_data.columns = ['userID', 'ISBN', 'bookRating']



'''                     ###        COMBİNE ALL DATASET       ###                                                    '''

combine_book_rating = pd.merge(train, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
train_combine_book_rating = combine_book_rating.drop(columns, axis=1)
combined = train_combine_book_rating.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')
us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
ratings_explicit = us_canada_user_rating[us_canada_user_rating.bookRating != 0]

'''                     ###        SETTING TEST DATA       ###                                                    '''

test=test_data
test=test[test['bookRating']>0]
test=pd.merge(test,users,on='userID')
test=pd.merge(test,books,on='ISBN')
test=test[test['Location'].str.contains("usa|canada")]
test=test.dropna()
'''                     ###        PIVOTING TRAINING DATASET       ###                                                    '''
ratings_pivot = ratings_explicit.pivot(index='ISBN', columns='userID',values='bookRating')
books_ratings_matrix=ratings_pivot.fillna(0)
train_data_dict=books_ratings_matrix.to_dict()

'''###  Cross validation partition of data is train data (%80) and test data (%20)

#partition=np.random.rand(len(ratings_explicit))<0.8
#train_data=ratings_explicit[partition]
#test_data=ratings_explicit[~partition]                                         ###                                 '''


k=3
mae_list=[]
'''                     ###        READING TEST DATA AND CALL FUNCTION       ###                                                    '''
for index,row in test.iterrows():
    id=row['userID']
    books=row['ISBN']
    rate=row[2]
    if id not in train_data_dict.keys():
        if books not in list(train_data_dict.values())[0]:
            continue
        else:
            pre=average_book(train_data_dict,row[1])
            liste = [rate, pre]
            mae_list.append(liste)

    else:                                                                       ###    CONTROL STATEMENT TEST DATA      ###
        if books not in list(train_data_dict.values())[0]:
            pre=average_user(train_data_dict,row[0])
            liste = [rate, pre]
            mae_list.append(liste)


        else:
            liste=predict(row[0],row[1],row[2])
            #pre=unweighted_knn(liste,k,row[1])
            pre=weighted_knn(liste,k,row[1])
            liste=[rate, pre]
            mae_list.append(liste)

'''###        CALL MAE FUNCTION AND RETURN VALUE       ###'''

print('MAE----->',meanAbsoluteError(mae_list))



