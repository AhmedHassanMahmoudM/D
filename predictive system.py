# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:54:28 2024

@author: ah407
"""

import numpy as np
import pickle



# Lode The Saving Model
loaded_model = pickle.load(open(r'C:/Users\ah407\OneDrive\Desktop/Diabetes deployment/trained_model.sav','rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')