import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle


# read csv file
heart_df = pd.read_csv(r'C:\Users\ad\Desktop\MBURU\heart_statlog_cleveland_hungary_final.csv')

# split into features and labels
feat = heart_df.drop("target", axis=1)
X = feat.values
y = heart_df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 8)



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# lbraries
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

pickle.dump(rf, open('heart.pkl', 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))
