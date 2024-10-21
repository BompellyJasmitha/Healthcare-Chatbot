import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load training and testing data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Group data for maximum values per prognosis
reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Train the Decision Tree Classifier
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())

# Train the SVC model
model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

# Initialize dictionaries for symptoms, severity, descriptions, and precautions
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

def calc_condition(exp, days):
    total_severity = sum(severityDictionary[item] for item in exp)
    if (total_severity * days) / (len(exp) + 1) > 13:
        print("You should take the consultation from a doctor.")
    else:
        print("It might not be that bad, but you should take precautions.")

def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 1:  # Ensure there are enough columns
                severityDictionary[row[0]] = int(row[1])

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    name = input("\nYour Name? \t\t\t\t-> ")
    print("Hello, ", name)

def check_pattern(dis_list, inp):
    inp = inp.replace(' ', '_')
    pattern = re.compile(inp)
    pred_list = [item for item in dis_list if pattern.search(item)]
    return (1, pred_list) if pred_list else (0, [])

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        if item in symptoms_dict:  # Ensure the symptom exists
            input_vector[symptoms_dict[item]] = 1

    # Convert to DataFrame for prediction
    input_df = pd.DataFrame([input_vector], columns=X.columns)
    return rf_clf.predict(input_df)

def print_disease(node):
    node = node[0]
    val = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        disease_input = input("\nEnter the symptom you are experiencing \t\t-> ")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("Searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            conf_inp = int(input(f"Select the one you meant (0 - {len(cnf_dis) - 1}): ")) if len(cnf_dis) > 1 else 0
            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. From how many days? : "))
            break
        except ValueError:
            print("Enter valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            val = 1 if name == disease_input else 0

            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any symptoms?")

            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                while True:
                    inp = input(f"{syms}? (yes/no): ")
                    if inp in ["yes", "no"]:
                        break
                    else:
                        print("Provide proper answers i.e. (yes/no): ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])
            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precaution_list = precautionDictionary[present_disease[0]]
            print("Take the following measures: ")
            for i, j in enumerate(precaution_list):
                print(i + 1, ")", j)

    recurse(0, 1)

# Load necessary data
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols)
print("----------------------------------------------------------------------------------------")
