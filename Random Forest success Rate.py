
# coding: utf-8

# In[44]:


# Required Python Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import style
from sklearn.cluster import KMeans


# In[45]:


df = pd.read_csv('mysoreRandomForestCsv.csv')
df.shape


# In[46]:


headers = ["Name of Restaurant", "Area Code", "Average Cost (for 2)","No of reviews","Food rating","Service rating","Look and Feel rating","Alcohol availability(Y1/N0)", "Veg0/Non-veg1",
                "Italian", "North Indian", "Chinese", "Continental",
               "European", "Mediteranean", "Andhra", "Mughlai", "South Indian",
               "Asian","cafe","Zomato Rating(out of 5)"]
df.columns = headers


# In[47]:


df["Name of Restaurant"][0]


# In[48]:


df.isnull().any()


# In[49]:


type(df.iat[1,0])


# In[50]:


for i in range(0,len(df)):
    if df.iat[i,1] == "Bannimantap":
        df.iat[i,1] = 0
    elif df.iat[i,1] == "Bogadi":
        df.iat[i,1] = 1
    elif df.iat[i,1] == "Chamarajapura":
        df.iat[i,1] = 2
    elif df.iat[i,1] == "Chamrajpura":
        df.iat[i,1] = 2
    elif df.iat[i,1] == "Chamundipuram":
        df.iat[i,1] = 3
    elif df.iat[i,1] == "Doora":
        df.iat[i,1] = 4
    elif df.iat[i,1] == "Gangothri Layout":
        df.iat[i,1] = 5
    elif df.iat[i,1] == "Gayathripuram":
        df.iat[i,1] = 6
    elif df.iat[i,1] == "Gokulam":
        df.iat[i,1] = 7
    elif df.iat[i,1] == "Hebbal":
        df.iat[i,1] = 8
    elif df.iat[i,1] == "Ittige Gudu":
        df.iat[i,1] = 9
    elif df.iat[i,1] == "Jayalakhsmipuram":
        df.iat[i,1] = 10
    elif df.iat[i,1] == "JC Nagar":
        df.iat[i,1] = 11
    elif df.iat[i,1] == "Kuvempunagar":
        df.iat[i,1] = 12
    elif df.iat[i,1] == "Mandi Mohalla":
        df.iat[i,1] = 13
    elif df.iat[i,1] == "Nelson Mandela Circle":
        df.iat[i,1] = 14
    elif df.iat[i,1] == "Saraswathipuram":
        df.iat[i,1] = 15
    elif df.iat[i,1] == "Srirangapatna":
        df.iat[i,1] = 16
    elif df.iat[i,1] == "V V Mohalla":
        df.iat[i,1] = 17
    elif df.iat[i,1] == "Vidayaranya Puram":
        df.iat[i,1] = 18
    elif df.iat[i,1] == "Vijay Nagar":
        df.iat[i,1] = 19
    elif df.iat[i,1] == "Yadavgiri":
        df.iat[i,1] = 20
    elif df.iat[i,1] == "Nazarbad":
        df.iat[i,1] = 22


# In[51]:


for i in range(0,len(df)):
    print df["Area Code"][i]


# In[52]:


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """
 
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y


# In[53]:


train_x, test_x, train_y, test_y = split_dataset(df, 0.7, headers[1:-1], headers[-1])


# In[54]:


print "Train_x Shape :: ", train_x.shape
print "Train_y Shape :: ", train_y.shape
print "Test_x Shape :: ", test_x.shape
print "Test_y Shape :: ", test_y.shape


# In[55]:


from sklearn.ensemble import RandomForestRegressor
def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    depth = 10
    rf = RandomForestRegressor(max_depth=depth, random_state=2)
    # Train the model on training data
    rf.fit(features, target);
    # clf = RandomForestClassifier()
    #clf.fit(features, target)
    return rf


# In[56]:


trained_model = random_forest_classifier(train_x, train_y)


# In[57]:


print "Trained model :: ", trained_model


# In[58]:


predictions = trained_model.predict(test_x)


# In[59]:


for i in xrange(0, len(test_x)):
        print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], round(predictions[i],1))


# In[60]:


errors = abs(predictions - test_y)


# In[61]:





# In[62]:


round(np.mean(errors), 2)


# In[63]:


mape = 100 * (errors / test_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print 'Accuracy:', round(accuracy, 2), '%.'

