# Ensemble Voting Model

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Define your models here

models = [('lr', LogisticRegression()), ('svc', SVC()), ('dt', DecisionTreeClassifier())]
ensemble_model = VotingClassifier(estimators=models, voting='hard')

# Train your ensemble model here
