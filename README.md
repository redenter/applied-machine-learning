# The-Titanic-Disaster
The goal is to determine the survival of passengers who  were travelling on Titanic. 
The dataset which is from Kaggle and  provides the following information about the passengers.   
Name: Name of the passenger  Pclass: a proxy for socioAeconomic status from 1 to 3, 1 indicating the highest  status. 
Fare: Cost of the ticket 
Sex: Sex of the passenger (male or female)  Embarked: Where the passenger embarked, S, C or Q.  SibSp: Number of siblings or spouse present in the ship.  Parch: Whether the passenger had any children or parent with him/her.   Age: age of the passenger 
Cabin: cabin location of the passenger.

Logistic regression has been used for classification and the algorithm has an accuracy of 78.469%.

File : execute.py

Data manipulation was done on the input data to obtain the result.

1.  Fare :  This column had very few missing features. So, an average value  across the entire dataset was taken.  
2.  Cabin: This column had too many missing values. So, this data was not  used. 
3.  Age:  This is a crucial piece of data, as members of different age groups  might have reacted differently to the disaster and hence might have  positioned themselves accordingly to improve their (or their family  members’) chances of survival. Different techniques were attempted in  filling out the missing age data. For instance, using the gender wise  mean of training data was attempted. However, this did not improve the  results. We obtained better results when the title of the passenger  (extracted from the name) was used.  
a.  Mean value of all passengers with title Mr. was used for those  with similar title. Similar approach was taken for passengers with  title Mrs., Master., Dr. and Rev.  
b.  For passengers with title Miss., class was considered. This was  done primarily because passengers with the above title could  belong to any age group. By calculating class wise average, we can  get better accuracy. For instance, members of class 3 with title  Miss. could mostly be teenagers, whereas members of class 1 with  the title could be middle aged women.  
  As mentioned before, different permutations of the dataset were selected as  features. However, it was noticed that including more features didn’t  necessarily translate to an improvement in the prediction, and in some cases  the accuracy decreased. For example, when using gender, class, SibSp, Parch,  Embarked and age were used as features the accuracy did not improve, and  when gender, class, SibSp, Parch, Fare and Embarked were used, the result  went down to 76.077%.  


Different permutations attempted and their corresponding accuracy has been  mentioned below. 
 
Gender, class, SibSp, Parch : 77.03%  
Gender, class, SibSp, Parch, Fare : 77.03% 
Gender, class, Fare : 76.55%  
Gender, class, SibSp, Parch, Fare, Embarked: 76.077%  
Gender, class, SibSp, Parch, Fare, Age : 77.512%  
Gender, class, SibSp, Parch, Age : 77.03%  
Gender, class, SibSp, Parch, Embarked, Age : 76.55%  
Gender, class, Age : 78.469%  
