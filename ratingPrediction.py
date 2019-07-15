# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
df = pd.read_csv('mysoreRandomForestCsv.csv')
headers = ["Name of Restaurant", "Area Code", "Average Cost (for 2)","No of reviews","Food rating","Service rating","Look and Feel rating","Alcohol availability(Y1/N0)", "Veg0/Non-veg1",
                "Italian", "North Indian", "Chinese", "Continental",
               "European", "Mediteranean", "Andhra", "Mughlai", "South Indian",
               "Asian","cafe","Zomato Rating(out of 5)"]
df.columns = headers
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
    
train_x, test_x, train_y, test_y = split_dataset(df, 0.7, headers[1:-1], headers[-1])
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
trained_model = random_forest_classifier(train_x, train_y)
predictions = trained_model.predict(test_x)
# test_1 =  pd.DataFrame([[0, 600, 50, 4.5, 4,4.2,1,0,1,1,0,0,0,1,1,1,1,0,0]], 
#                       columns=("Area Code", "Average Cost (for 2)","No of reviews","Food rating","Service rating","Look and Feel rating","Alcohol availability(Y1/N0)", "Veg0/Non-veg1",
#                 "Italian", "North Indian", "Chinese", "Continental",
#                "European", "Mediteranean", "Andhra", "Mughlai", "South Indian",
#                "Asian","cafe"))  
#  predict = trained_model.predict(test_1)  
for i in range(0, len(test_x)):
    print(round(predictions[i],1))