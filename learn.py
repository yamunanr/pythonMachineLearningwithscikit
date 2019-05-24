import pandas as pd
from sklearn import preprocessing

print(pd.__version__)

exam_data = pd.read_csv('E:/Python Machinelearning scikit/exams.csv', quotechar='"')
print(exam_data)


#Standardize the subject scores because each of the subjects have scores for different total marks

math_average = exam_data['math score'].mean()
reading_average = exam_data['reading score'].mean()
writing_average = average = exam_data['writing score'].mean()

print('Math Average: ', math_average)
print('Reading Average: ', reading_average)
print('Writing Average: ', writing_average)


#Standardization divide each score and subtract by standard deviation
#After scaling, mean is zero or almost zero and the standard deviation of scores is one

exam_data[['math score']] = preprocessing.scale(exam_data[['math score']])
exam_data[['reading score']] = preprocessing.scale(exam_data[['reading score']])
exam_data[['writing score']] = preprocessing.scale(exam_data[['writing score']])
print(exam_data)
math_average = exam_data['math score'].mean()
reading_average = exam_data['reading score'].mean()
writing_average = average = exam_data['writing score'].mean()
print('Math Average: ', math_average)
print('Reading Average: ', reading_average)
print('Writing Average: ', writing_average)

#Encoding categorical variable -> Label encoder
# representing categorical data (gender) 1 represents male, 0 represents female
le = preprocessing.LabelEncoder()
exam_data['gender'] = le.fit_transform(exam_data['gender'].astype(str))
exam_data.head()
print(exam_data)

le.classes_
#pandas library offers easy way to convert data into One-hot representation
One_hot = pd.get_dummies(exam_data['race/ethnicity'])
print(One_hot)

# get_dummies can be used upon multiple columns at same time

One_hot3 = pd.get_dummies(exam_data, columns= ['parental level of education', 'lunch', 'test preparation course'])
print(One_hot3)