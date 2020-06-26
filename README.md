# Machine-Learning-Projects
 Repository of various machine learning projects.
 
 ##1)Language Classifier

###How to use the Code: -
Training and testing data is stored in Language Classfier folder with the name Train.dat and test.dat.

It has two entry points (different modules or command-line args):

a) To train the data:
train <examples> <hypothesisOut> <learning-type>
train – by providing train keyword it will start the training procedure
examples – It will contain the data which will be used to Train the model, the data exactly matches the format asked in the question (no gap between “label”| and text) i.e 15 words with the proper label for example : en|The batting side scores runs by striking the ball bowled at the wicket with the nl|Het is een teamsport waarin om beurt het ene team eerst gooit en het andere.
hypothesisOut – It contains the file through which prediction will be performed, it will contain the object
according to the learning type.(give the filename u want to store the prediction )
Learning Type – It specifies the learning algorithim “dt” for decision Tree “ada” for adaboost
 
b) To predict the data
predict <hypothesis> <file>
predict – by providing predict keyword it will start the prediction process
hypothesis – It contains the file through which prediction will be performed, it will contain the object
according to the learning type (filename where you have stored the prediction )
File - It will contain the data which will be used to Test the model, the data exactly matches the format asked in the question i.e 15 words with the no label for example : Zijn eerste boek The Peasants Revolt verscheen in 2009. Zijn tweede boek over het Huis he went to Sweden, fleeing the Soviet occupation of Estonia, and continued studying Romance
 
Data Collection
There are 2000+ which is scrapped using web scrapper and some copied manually, the data is collected from wiki pages.

