# healthcare-chatbot
a chatbot based on sklearn where you can give a symptom and it will ask you questions and will tell you the details and give some advice.
itching,
skin_rash,
nodal_skin_eruptions,
continuous_sneezing,
shivering,
chills,
joint_pain,
stomach_pain,
acidity,
ulcers_on_tongue,
muscle_wasting,
vomiting,
burning_micturition,
spotting_ urination,
fatigue,
weight_gain,
anxiety,
cold_hands_and_feets,
mood_swings,
weight_loss,
restlessness,
lethargy,
patches_in_throat,
irregular_sugar_level,
cough,
high_fever,
sunken_eyes,
breathlessness,
sweating,
dehydration,
indigestion,
headache,
yellowish_skin,
dark_urine,
nausea,
loss_of_appetite,
pain_behind_the_eyes,
back_pain,
constipation,
abdominal_pain,
diarrhoea,
mild_fever,
yellow_urine,
yellowing_of_eyes,
acute_liver_failure,
fluid_overload,
swelling_of_stomach,
swelled_lymph_nodes,
malaise,
blurred_and_distorted_vision,
phlegm,
throat_irritation,
redness_of_eyes,
sinus_pressure,
runny_nose,congestion,
chest_pain,
weakness_in_limbs,
fast_heart_rate,
pain_during_bowel_movements,
pain_in_anal_region,
bloody_stool,
irritation_in_anus,
neck_pain,
dizziness,
cramps,
bruising,
obesity,
swollen_legs,
swollen_blood_vessels,
puffy_face_and_eyes,
enlarged_thyroid,
brittle_nails,
swollen_extremeties,
excessive_hunger,
extra_marital_contacts,
drying_and_tingling_lips,
slurred_speech,
knee_pain,
hip_joint_pain,
muscle_weakness,
stiff_neck,
swelling_joints,
movement_stiffness,
spinning_movements,
loss_of_balance,
unsteadiness,
weakness_of_one_body_side,
loss_of_smell,
bladder_discomfort,
foul_smell_of urine,continuous_feel_of_urine,
passage_of_gases,
internal_itching,
toxic_look_(typhos),
depression,
irritability,
muscle_pain,
altered_sensorium,
red_spots_over_body,
belly_pain,
abnormal_menstruation,
dischromic _patches,
watering_from_eyes,
increased_appetite,
polyuria,
family_history,
mucoid_sputum,
rusty_sputum,
lack_of_concentration,
visual_disturbances,
receiving_blood_transfusion,
receiving_unsterile_injections,
coma,
stomach_bleeding,
distention_of_abdomen,
history_of_alcohol_consumption,
fluid_overload,
blood_in_sputum,
prominent_veins_on_calf,
palpitations,
painful_walking,
pus_filled_pimples,
blackheads,
scurring,
skin_peeling,
silver_like_dusting,
small_dents_in_nails,
inflammatory_nails,
blister,
red_sore_around_nose,
yellow_crust_ooze,
prognosis
HealthCare ChatBot Project
Overview
This project implements a HealthCare ChatBot that predicts potential diseases based on user-reported symptoms using machine learning models. The chatbot also provides severity assessments and precautions based on symptoms.

# Requirements
Before running the project, ensure you have the following packages installed:

Python 3.x
pandas
scikit-learn
pyttsx3
numpy
csv
re

You can install the required libraries using pip:
#       pip install pandas scikit-learn pyttsx3 numpy

## Project Structure
Data/
Training.csv - Contains training data for the model.
Testing.csv - Contains testing data for evaluation.
MasterData/
symptom_Description.csv - Contains descriptions for symptoms.
symptom_severity.csv - Contains severity ratings for symptoms.
symptom_precaution.csv - Contains precautionary measures for symptoms.

##   Step-by-Step Execution

Step 1: Load Data
The script loads training and testing data using pandas. Ensure that Training.csv and Testing.csv files are correctly placed in the Data/ directory.
#       training = pd.read_csv('Data/Training.csv')
#       testing = pd.read_csv('Data/Testing.csv')

Step 2: Prepare Data
Extract features (symptoms) and labels (prognosis) from the training data.
Encode the labels using LabelEncoder from scikit-learn.
#       cols = training.columns[:-1]
#       x = training[cols]
#       y = training['prognosis']

Step 3: Split the Data
The dataset is split into training and testing sets.
#       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

Step 4: Train Models
Train a Decision Tree Classifier and a Support Vector Classifier (SVC) on the training data.
#       clf = DecisionTreeClassifier().fit(x_train, y_train)
#       model = SVC().fit(x_train, y_train)

Step 5: Load Additional Data
Load severity ratings, descriptions, and precautions for symptoms.
#       getSeverityDict()
#       getDescription()
#       getprecautionDict()

Step 6: User Interaction
The chatbot initiates interaction by asking for the user's name and symptoms.
#       getInfo()

Step 7: Symptom Checking
The chatbot allows users to input symptoms and evaluates the potential diseases based on a decision tree.
#       tree_to_code(clf, cols)

Step 8: Severity and Precaution Assessment
Based on symptoms reported, the chatbot calculates the severity and provides precautionary measures.

Step 9: Run the Script
Run the main script using Python:
#        python your_script_name.py

## Notes
Ensure that all CSV files are formatted correctly and contain the necessary columns as referenced in the code.
Modify the paths to the CSV files as necessary based on your local directory structure.

## Disclaimer
This chatbot is intended for informational purposes only and should not be considered a substitute for professional medical advice. Always consult a healthcare professional for medical concerns.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.