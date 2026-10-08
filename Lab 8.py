import pandas as pd
import seaborn as sns
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
#Load and preprocess the titanic dataset
df = sns.load_dataset('titanic')
df = df[['survived','pclass','sex','age']]
df.dropna(inplace = True)
df['age'] = pd.cut(df['age'], bins=[0,12,30,50,100], labels=['child', 'young', 'adult', 'senior'])
#Convert categorical columns to numeric codes
df['sex'] = df['sex'].astype('category').cat.codes
df['age'] = df['age'].astype('category').cat.codes
model = BayesianNetwork([
    ('pclass','survived'),
    ('sex', 'survived'),
    ('age','survived')
])
model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')
inference = VariableElimination(model)
#Query 1: First class, female , young
result1 = inference.query(
    variables=['survived'],
    evidence={'pclass':1, 'sex':0, 'age':1}
)
print("Query 1 - 1st class, female, young:")
print(result1)
#Query2: Third class, male, adult
result2 = inference.query(
    variables=['survived'],
    evidence = {'pclass':3, 'sex': 1, 'age':2}
)
print("\nQuery 2 - 3rd class, male, adult:")
print(result2)
#Query3: Second class, male, senior
result3 = inference.query(
    variables=['survived'],
    evidence={'pclass': 1,'sex': 0,'age':0}
)
print("\nQuery 4 - 1st class, female, child:")
print(result4)
