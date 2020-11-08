import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets


D = numpy.loadtxt('sampleQuadData2020.txt')
DT = numpy.loadtxt('sampleQuadDataTransformed2020.txt')
boob = datasets.load_breast_cancer()
wine = datasets.load_wine()

y = D[:,-1]
X = D[:,:-1]

yb = boob.target
Xb = boob.data

yt = DT[:,-1]
Xt = DT[:,:-1]

#  Assume that the cost of incorrectly labeling an instance as benign when it is actually malignant is four times the cost
# of incorrectly labeling a benign instance as malignant. Train a standard (costinsensitive) LogisticRegression model on the dataset. Save the class predictions
# made by that model; you’ll need them. Extract the predicted class probabilities
# for all of the data instances from the trained model, and use them to make optimal
# cost-sensitive predictions by following your solution to the preceding part. Compare the class predictions of your cost-sensitive version to those of the standard
# cost-insensitive model.
## QUESTION: 1
model = LogisticRegression(max_iter=300000)
model.fit(Xb,yb)

prediction = model.predict(Xb)

probability = model.predict_proba(Xb)

result = []
difference = []

for i in range(len(probability)):
    if probability[i, -1] >= 0.2:
        result +=[1]
    else:
        result +=[0]

counter = 0

for i in range(len(prediction)):
    if prediction[i] != result[i]:
        counter += 1
        difference.append(i)

print("Difference between two predictions:", counter,"\n")

#END # QUESTION: 1

#
# Fit an SGDClassifier object from the sklearn.linear_model module to the
# labeled data in the file sampleQuadData.txt, using default values for all parameters. Report the score obtained on this data set. Repeat the process, using the
# pre-processed data in the file sampleQuadDataTransformed.txt, which contains
# the same set of data examples as in sampleQuadData.txt, but transformed as
# described in the statement, above. This time, specify fit_intercept=False in
# the constructor method; the inclusion of the 1 constant as the first component of
# the transformed data, z1, makes the bias (intercept) term unnecessary. How do
# the results compare? Discuss, considering the respective classification error rates.
# CSCI 3345, Alvarez (Fall 2020)
# (c) Classification error rate, while useful, implicitly assumes equal sizes (and importances) of the different classes. Use the sklearn.metrics.ConfusionMatrix
# class to compute confusion matrices for the two models from the preceding part.
# Describe any differences between the two matrices.
# (d) Take a closer look at the SGDClassifier instance for the transformed data, by
# examining its coefficients in the _coef data field. What is the approximate equation of the decision boundary in the space (z1, z2, z3, z4, z5, z6)? What is the corresponding equation in the original space (x1, x2)? Explain in detail.
# (e) Ignore the mixed x1x2 term in the boundary equation from the preceding part in
# order to identify the geometric shape of the boundary (look up information about
# analytic geometry). State the specific shape name. Does this boundary shape
# seem like a good model of the available data in this task?


## QUESTION: 2
classifier = SGDClassifier()
classifierT = SGDClassifier(fit_intercept=False)


step1 = classifier.fit(X,y)
score = step1.score(X,y)

step1t = classifierT.fit(Xt,yt)

#print("coeficient values", classifierT.coef_,"\n")

scoret = step1t.score(Xt,yt)

print("score for untransformed data:", score,"\n")
print("score for transformed data:", scoret, "\n")

ypred = classifier.predict(X)
t = confusion_matrix(y, ypred)

ypredT = classifierT.predict(Xt)
T = confusion_matrix(yt, ypredT)

print("untransformed data confusion matrix \n", t, "\n")
print("transformed data confusion matrix \n", T, "\n")
#END # QUESTION: 2


# Train a sequence of SVC models on the breast-cancer data set, one for each of
# the values 10p
# , p = −1, 0, · · · , 9 of the regularization constant, C. Estimate the
# out-of-sample performance of each model using 4-fold cross-validation. Report
# the mean accuracy for each of the models; plot the accuracy as a function of C.
# (b) Analyze your results from the preceding part. Does the value of the regularization
# constant have a measurable effect on performance? If so, how? Does performance
# tend to be better when more regularization is used, or less? Discuss in quantitative
# terms. Read the documentation carefully.

## QUESTION: 3
svm =[]
for n in range(-1, 10, 1):

    value = 10**n
    clf = SVC(C=value)
    clf.fit(Xb,yb)
    clf.score(Xb,yb)

    temp = numpy.mean(cross_val_score(clf, Xb, yb, cv=4))
    svm.append(temp)

plt.title("Cross Validation:")
plt.plot(svm)
plt.show()
#END # QUESTION: 3
#
# (a) Assuming that leave-one-out-cross-validation is used to estimate the classification
# performance of each possible combination of p and C as described above, how
# many individual classifier models will need to be trained during the entire model
# selection process? Explain in detail, arriving at a concrete number in the end.
# CSCI 3345, Alvarez (Fall 2020)
# (b) Use sklearn.model_selection.GridSearchCV to carry out the model selection
# task described in the above statement. Select the parameter values so that leaveone-out cross-validation is used. Set the iid parameter to False. Submit your
# Python code, and report the best hyperparameter values as determined by the
# grid search procedure.

## QUESTION: 4
def num4(c,d):
    loo = LeaveOneOut()

    Xw = wine.data
    yw = wine.target

    parameters = {'C':[c], 'degree':[d]}

    scv = SVC(kernel = 'poly')
    Gclf = GridSearchCV(scv, parameters)
    Gclf.fit(Xw,yw)
    Gclf.score(Xw,yw)
    result = numpy.mean(cross_val_score(Gclf, Xw,yw, cv = loo))

    print(result)

#will prnt all the scores depending on the different values
num4(.5,1)
num4(.5,2)
num4(.5,4)
num4(1,1)
num4(1,2)
num4(1,4)
num4(2,1)
num4(2,2)
num4(2,4)

#END # QUESTION: 4
