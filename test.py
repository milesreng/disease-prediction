import numpy as np
import pandas as pd

from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

train_data = pd.read_csv("data/Training.csv").dropna(axis = 1)

disease_counts = train_data["prognosis"].value_counts()
temp_df = pd.DataFrame({
  "Disease": disease_counts.index,
  "Counts": disease_counts.values
})

# Exploratory data visualization
plt.figure(figsize=(18,8))
sns.barplot(x = "Disease", y = "Counts", data = temp_df)
plt.xticks(rotation = 90)
plt.show()