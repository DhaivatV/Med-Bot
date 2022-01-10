#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xgb
import pandas as pd


# In[2]:


model = xgb.Booster({'nthread':4})
model.load_model('Med_Bot.model')


# In[11]:


array=([[  5.,   9.,   65., 53., 72., 83., 63., 85., 11., 25., 123.,
       1., 33., 83., 73., 42., 13.]])
lst= []
for item in array:
    lst.append(item)
lst


# In[12]:


test= xgb.DMatrix(array, label=None)


# In[13]:


input_dict = {0: 'itching',
 1: 'skin_rash',
 2: 'nodal_skin_eruptions',
 3: 'continuous_sneezing',
 4: 'shivering',
 5: 'chills',
 6: 'joint_pain',
 7: 'stomach_pain',
 8: 'acidity',
 9: 'ulcers_on_tongue',
 10: 'muscle_wasting',
 11: 'vomiting',
 12: 'burning_micturition',
 13: 'spotting_urination',
 14: 'fatigue',
 15: 'weight_gain',
 16: 'anxiety',
 17: 'cold_hands_and_feets',
 18: 'mood_swings',
 19: 'weight_loss',
 20: 'restlessness',
 21: 'lethargy',
 22: 'patches_in_throat',
 23: 'irregular_sugar_level',
 24: 'cough',
 25: 'high_fever',
 26: 'sunken_eyes',
 27: 'breathlessness',
 28: 'sweating',
 29: 'dehydration',
 30: 'indigestion',
 31: 'headache',
 32: 'yellowish_skin',
 33: 'dark_urine',
 34: 'nausea',
 35: 'loss_of_appetite',
 36: 'pain_behind_the_eyes',
 37: 'back_pain',
 38: 'constipation',
 39: 'abdominal_pain',
 40: 'diarrhoea',
 41: 'mild_fever',
 42: 'yellow_urine',
 43: 'yellowing_of_eyes',
 44: 'acute_liver_failure',
 45: 'fluid_overload',
 46: 'swelling_of_stomach',
 47: 'swelled_lymph_nodes',
 48: 'malaise',
 49: 'blurred_and_distorted_vision',
 50: 'phlegm',
 51: 'throat_irritation',
 52: 'redness_of_eyes',
 53: 'sinus_pressure',
 54: 'runny_nose',
 55: 'congestion',
 56: 'chest_pain',
 57: 'weakness_in_limbs',
 58: 'fast_heart_rate',
 59: 'pain_during_bowel_movements',
 60: 'pain_in_anal_region',
 61: 'bloody_stool',
 62: 'irritation_in_anus',
 63: 'neck_pain',
 64: 'dizziness',
 65: 'cramps',
 66: 'bruising',
 67: 'obesity',
 68: 'swollen_legs',
 69: 'swollen_blood_vessels',
 70: 'puffy_face_and_eyes',
 71: 'enlarged_thyroid',
 72: 'brittle_nails',
 73: 'swollen_extremeties',
 74: 'excessive_hunger',
 75: 'extra_marital_contacts',
 76: 'drying_and_tingling_lips',
 77: 'slurred_speech',
 78: 'knee_pain',
 79: 'hip_joint_pain',
 80: 'muscle_weakness',
 81: 'stiff_neck',
 82: 'swelling_joints',
 83: 'movement_stiffness',
 84: 'spinning_movements',
 85: 'loss_of_balance',
 86: 'unsteadiness',
 87: 'weakness_of_one_body_side',
 88: 'loss_of_smell',
 89: 'bladder_discomfort',
 90: 'foul_smell_ofurine',
 91: 'continuous_feel_of_urine',
 92: 'passage_of_gases',
 93: 'internal_itching',
 94: 'toxic_look_(typhos)',
 95: 'depression',
 96: 'irritability',
 97: 'muscle_pain',
 98: 'altered_sensorium',
 99: 'red_spots_over_body',
 100: 'belly_pain',
 101: 'abnormal_menstruation',
 102: 'dischromic _patches',
 103: 'watering_from_eyes',
 104: 'increased_appetite',
 105: 'polyuria',
 106: 'family_history',
 107: 'mucoid_sputum',
 108: 'rusty_sputum',
 109: 'lack_of_concentration',
 110: 'visual_disturbances',
 111: 'receiving_blood_transfusion',
 112: 'receiving_unsterile_injections',
 113: 'coma',
 114: 'stomach_bleeding',
 115: 'distention_of_abdomen',
 116: 'history_of_alcohol_consumption',
 117: 'fluid_overload',
 118: 'blood_in_sputum',
 119: 'prominent_veins_on_calf',
 120: 'palpitations',
 121: 'painful_walking',
 122: 'pus_filled_pimples',
 123: 'blackheads',
 124: 'scurring',
 125: 'skin_peeling',
 126: 'silver_like_dusting',
 127: 'small_dents_in_nails',
 128: 'inflammatory_nails',
 129: 'blister',
 130: 'red_sore_around_nose',
 131: 'yellow_crust_ooze',
 132: 'prognosis'}


# In[14]:


output_dict= {'Fungal infection': 0,
 'Allergy': 10,
 'GERD': 20,
 'Chronic cholestasis': 30,
 'Drug Reaction': 40,
 'Peptic ulcer diseae': 50,
 'AIDS': 60,
 'Diabetes ': 70,
 'Gastroenteritis': 80,
 'Bronchial Asthma': 90,
 'Hypertension ': 100,
 'Migraine': 110,
 'Cervical spondylosis': 120,
 'Paralysis (brain hemorrhage)': 130,
 'Jaundice': 140,
 'Malaria': 150,
 'Chicken pox': 160,
 'Dengue': 170,
 'Typhoid': 180,
 'hepatitis A': 190,
 'Hepatitis B': 200,
 'Hepatitis C': 210,
 'Hepatitis D': 220,
 'Hepatitis E': 230,
 'Alcoholic hepatitis': 240,
 'Tuberculosis': 250,
 'Common Cold': 260,
 'Pneumonia': 270,
 'Dimorphic hemmorhoids(piles)': 280,
 'Heart attack': 290,
 'Varicose veins': 300,
 'Hypothyroidism': 310,
 'Hyperthyroidism': 320,
 'Hypoglycemia': 330,
 'Osteoarthristis': 340,
 'Arthritis': 350,
 '(vertigo) Paroymsal  Positional Vertigo': 360,
 'Acne': 370,
 'Urinary tract infection': 380,
 'Psoriasis': 390,
 'Impetigo': 400}


# In[15]:


new_dict = dict(zip(output_dict.values(), output_dict.keys()))


# In[16]:


pred= model.predict(test)
pred


# In[21]:


def output_dis(pred):
    for item in pred:
        k= int(item)
        
        
    disease = new_dict.get(k)
    print(disease)


# In[22]:


output_dis(pred)


# In[ ]:




