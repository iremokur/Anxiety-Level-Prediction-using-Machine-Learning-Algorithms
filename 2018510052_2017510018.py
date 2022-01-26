# -*- coding: utf-8 -*-
# Load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_predict, cross_val_score
from sklearn import metrics
import pickle

from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV


# Load dataset from a specific path (Note: if there is no # header, use the parameter header = None)
data = pd.read_csv("data.csv", sep='\t')

# Load data
dataframe = pd.DataFrame(data)

# drop columns
delete = ["Q1A", "Q1I", "Q1E", "Q2I", "Q2E", "Q3A", "Q3I", "Q3E", "Q4I", "Q4E", "Q5A", "Q5I", "Q5E", "Q6A", "Q6I", "Q6E", "Q7I", "Q7E", "Q8A", "Q8I", "Q8E", "Q9I", "Q9E", "Q10A", "Q10I", "Q10E", "Q11A", "Q11I", "Q11E", "Q12A", "Q12I", "Q12E", "Q13A", "Q13I", "Q13E", "Q14A", "Q14I", "Q14E", "Q15I", "Q15E", "Q16A", "Q16I", "Q16E", "Q17A", "Q17I", "Q17E", "Q18A", "Q18I", "Q18E", "Q19I", "Q19E", "Q20I", "Q20E", "Q21A", "Q21I", "Q21E", "Q22A", "Q22I", "Q22E", "Q23I", "Q23E", "Q24A", "Q24I", "Q24E", "Q25I", "Q25E", "Q26A", "Q26I", "Q26E", "Q27A", "Q27I", "Q27E", "Q28I", "Q28E", "Q29A", "Q29I", "Q29E", "Q30I", "Q30E", "Q31A", "Q31I", "Q31E", "Q32A", "Q32I", "Q32E", "Q33A", "Q33I", "Q33E", "Q34A", "Q34I", "Q34E", "Q35A", "Q35I", "Q35E", "Q36I", "Q36E", "Q37A", "Q37I", "Q37E", "Q38A", "Q38I", "Q38E", "Q39A", "Q39I", "Q39E", "Q40I", "Q40E", "Q41I", "Q41E", "Q42A", "Q42I", "Q42E", "source", "introelapse", "testelapse", "surveyelapse", "TIPI1", "TIPI2", "TIPI3", "TIPI4", "TIPI5", "TIPI6", "TIPI7", "TIPI8", "TIPI9", "TIPI10", "VCL1", "VCL2", "VCL3", "VCL4", "VCL5", "VCL6", "VCL7", "VCL8", "VCL9", "VCL10", "VCL11", "VCL12", "VCL13", "VCL14", "VCL15", "VCL16", "engnat", "screensize", "uniquenetworklocation", "hand", "voted", "familysize"]
for i in delete:
    dataframe.drop([i], axis=1, inplace=True)


# Remove observations with missing values
dataframe['major'].replace(" ", np.nan)
droppedData = dataframe.dropna()
droppedData.drop(droppedData[droppedData.major.str.contains('-', case=False)].index, inplace = True)
droppedData.drop(droppedData[droppedData['country']=="NONE"].index, inplace = True)
data2 = droppedData[~droppedData.major.str.contains('No', case=False)]

#correct spelling mistakes by using contains.
#to access the dataframe's row and columns used loc.
data2.loc[data2['major'].str.contains('engineer', case=False), 'major'] = 'Engineering'
data2.loc[data2['major'].str.contains('acc', case=False), 'major'] = 'Accounting'
data2.loc[data2['major'].str.contains('art', case=False), 'major'] = 'Art'
data2.loc[data2['major'].str.contains('education', case=False), 'major'] = 'Education'
data2.loc[data2['major'].str.contains('admin', case=False), 'major'] = 'Administration'
data2.loc[data2['major'].str.contains('architecture', case=False), 'major'] = 'Architecture'
data2.loc[data2['major'].str.contains('science', case=False), 'major'] = 'Science'
data2.loc[data2['major'].str.contains('human resource', case=False), 'major'] = 'Human Resources'
data2.loc[data2['major'].str.contains('hr', case=False), 'major'] = 'Human Resources'

# to ensure there is no spelling mistakes.
data2 = data2[data2.groupby('major').major.transform('count')>10].copy() 

questions = ['Q2A', 'Q4A', 'Q7A', 'Q9A', 'Q15A', 'Q19A', 'Q20A', 'Q23A', 'Q25A', 'Q28A', 'Q30A', 'Q36A', 'Q40A', 'Q41A']
demographic = ['education', 'urban', 'gender', 'age', 'religion', 'orientation', 'race', 'married']
categorical = ['major', "country"]

# Imputation for 0 value.
for col in demographic:
    imr = SimpleImputer(missing_values=0, strategy='most_frequent')
    imr = imr.fit(data2[[col]])
    data2[col] = imr.transform(data2[[col]]).ravel()

# Describe
# Data quality
describeCont = data2[questions+demographic].describe().T
describeCategoric = data2[categorical].astype(str).describe()


def scores(df):
    col = list(df)
    df["Scores"] = df[col].sum(axis=1)
    return df
def level(df, string,DASS_bins):
    conditions = [
    ((df['Scores'] >= DASS_bins[string][0][0])  & (df['Scores'] < DASS_bins[string][0][1])),
    ((df['Scores'] >= DASS_bins[string][1][0])  & (df['Scores'] < DASS_bins[string][1][1])),
    ((df['Scores'] >= DASS_bins[string][2][0])  & (df['Scores'] < DASS_bins[string][2][1])),
    ((df['Scores'] >= DASS_bins[string][3][0])  & (df['Scores'] < DASS_bins[string][3][1])),
    (((df['Scores'] >= DASS_bins[string][3][1])))
    ]
    levels = ['Normal','Mild', 'Moderate', 'Severe', 'Extremely Severe']
    df['Anxiety Level'] = np.select(conditions, levels)
    return df
    
    
DASS_keys = {'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]}
DASS_bins = {'Anxiety': [(0, 8), (8, 10), (10, 15), (15, 20)]}
anxiety_q = data2.filter(regex='Q\d{1,2}A')
train_anx = scores(anxiety_q)
train_anx = level(train_anx, 'Anxiety',DASS_bins)
demografic= data2.drop(questions,axis=1)
demografic.to_csv("Demog.csv")
train_anx.to_csv("Anxiety.csv")
# =============================================================================
# PLOTS
# =============================================================================
# =============================================================================
# #Overview of variable distribution
# data2.hist(figsize = (15,15))
# 
# 
# # Correlation and heatmap
# plt.figure(figsize=(16,16),dpi=80)
# corr = data2.corr()
# mask = np.triu(np.ones_like(corr, dtype=bool))
# sns.heatmap(corr, mask=mask, robust=True, center=0, square=True, linewidths=.5)
# plt.title('Correlation', fontsize=15)
# plt.show()
# 
# # Box-plot
# plt.figure(figsize=(20,10))
# plt.boxplot(data2[questions], labels=questions)
# plt.title('Question Answers Box-Plot', fontsize=15)
# # show plot
# plt.show()
# =============================================================================


# =============================================================================
# # Distribution Plot of the education and religion
# warnings.filterwarnings('ignore')
# plt.figure(figsize=(16,5))
# plt.subplot(1,2,1)
# plt.title(label="Education Distribution",
#            loc="left",
#            fontstyle='italic')
# sns.distplot(data2['education'])
# plt.subplot(1,2,2)
# plt.title(label="Religion Distribution",
#            loc="left",
#            fontstyle='italic')
# sns.distplot(data2['religion'])
# plt.show()
# 
# 
# # =============================================================================
# # QUESTIONS BAR GRAPHS
# # =============================================================================
# for col in questions:
#     plots = data2[col].astype(float).value_counts().plot(kind='bar',title='\n\n' + col)
#     plots.set_xlabel("Choice", fontsize=12)
#     plots.set_ylabel("Number of People", fontsize=12)
#     textstr = ("1 = Did not apply to me at all \n" +
#         "2 = Applied to me to some \ndegree,or some of the time\n"+
#         "3 = Applied to me to a \nconsiderable degree, or a \ngood part of the time\n"+
#         "4 = Applied to me very much,\nor most of the time")
#     # these are matplotlib.patch.Patch properties
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     
#     # place a text box in upper left in axes coords
#     plots.text(0.65, 1, textstr, transform=plots.transAxes, fontsize=10,
#             verticalalignment='center', bbox=props)
#     plt.show()
# 
# # =============================================================================
# # PIE CHART FOR GENDER
# # =============================================================================
# 
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.axis('equal')
# gender = data2.gender.unique()
# ax.pie(data2["gender"].astype(float).value_counts(), labels = ["female", "male", "other"],
#         autopct='%1.2f%%')
# 
# # displaying the title
# plt.title(label="Gender",
#           loc="left",
#           fontstyle='italic')
# 
# # =============================================================================
# # PIE CHART FOR EDUCATION
# # =============================================================================
# 
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.axis('equal')
# ax.pie(data2["education"].value_counts(), labels = [ "University degree",  "High school","Graduate degree","Less than high school"],
#         autopct='%1.2f%%')
#   
# #displaying the title
# plt.title(label="Education",
#            loc="left",
#            fontstyle='italic')
# 
# =============================================================================


#FINDING OUTLIERS
# =============================================================================
# def outlier():
#     # question that has outliers (understands from the boxplot) 
#     outlierQuestion=data2[['Q4A','Q7A','Q15A','Q19A','Q23A','Q41A']]
#     # finding the 1st quartile
#     q1 = np.quantile(outlierQuestion, 0.25)
#      
#     # finding the 3rd quartile
#     q3 = np.quantile(outlierQuestion, 0.75)
#      
#     # finding the interquartile range
#     iqr = q3-q1
#   
#     # finding upper and lower whiskers
#     upperBound = q3+(1.5*iqr)
#     lowerBound = q1-(1.5*iqr)
#     print(iqr, upperBound, lowerBound)
#     outliers = outlierQuestion[(outlierQuestion <= lowerBound) | (outlierQuestion >= upperBound)]
#     print(outliers)
# =============================================================================
# =============================================================================
#    
#     
# outlier()
# =============================================================================



