#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import shap
# from tkinter import * 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)


# In[5]:


# Load data and split into X and y
df = pd.read_excel(r"D:\Articles\GFRP Timber Article (1)\1.xlsx"  ,header = 0 )
y_load = df.loc[:, 'Ultimate load [kN] '].to_numpy().reshape((-1, 1))
y_slip = df.loc[:, 'Free end slip [mm]'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0, 1, 2, 3]].to_numpy()

indices = df.index.values

# Split the data
train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)

# Use the index array to select the rows for train and test sets

loadtrain_X, loadtrain_y,  = X[train_indices], y_load[train_indices]
loadtest_X, loadtest_y = X[test_indices], y_load[test_indices]

sliptrain_X, sliptrain_y,  = X[train_indices], y_slip[train_indices]
sliptest_X, sliptest_y = X[test_indices], y_slip[test_indices]

print("Training set indices:", train_indices)
print("Test set indices:", test_indices)

# Scale X
scaler = StandardScaler()
X_scaled_load = scaler.fit_transform(loadtrain_X)
X_scaled_slip = scaler.fit_transform(sliptrain_X)

# Train models
model_load = XGBRegressor(random_state=42)
model_load.fit(X_scaled_load, loadtrain_y)

model_slip = XGBRegressor(random_state=42)
model_slip.fit(X_scaled_slip, sliptrain_y)


# In[6]:


root = tk.Tk()
root.title("Target Predictor")
root.geometry("1200x600")

title_label = tk.Label(root, text="ML model for prediction of ultimate strength and free end slip glued-in rod timber joints", 
                       font=("Arial", 18))
title_label.pack(pady=20)

target_frame = tk.Frame(root)
target_frame.pack(pady=10)

# Radio buttons to select target
target_frame = tk.Frame(root)
target_frame.pack(pady=10)

target_label = tk.Label(target_frame, text="Select Target:" , font=22)
target_label.pack(side=tk.LEFT, padx=5)

target_var = tk.StringVar()
target_var.set("load")

load_rb = tk.Radiobutton(target_frame, text="Load", variable=target_var, value="load")
load_rb.pack(side=tk.LEFT, padx=5)

slip_rb = tk.Radiobutton(target_frame, text="Free End Slip", variable=target_var, value="slip")
slip_rb.pack(side=tk.LEFT, padx=5)

# Entry widgets for features

features_frame = tk.Frame(root)
features_frame.pack(pady=10)

f1_label = tk.Label(features_frame, text="Condition:")
f1_label.pack(side=tk.LEFT, padx=5)

f1_entry = tk.Entry(features_frame)
f1_entry.pack(side=tk.LEFT, padx=5)

f2_label = tk.Label(features_frame, text="Rod Diameter [mm]:")
f2_label.pack(side=tk.LEFT, padx=5)

f2_entry = tk.Entry(features_frame)
f2_entry.pack(side=tk.LEFT, padx=5)

f3_label = tk.Label(features_frame, text="Bonded Length [mm]:")
f3_label.pack(side=tk.LEFT, padx=5)

f3_entry = tk.Entry(features_frame)
f3_entry.pack(side=tk.LEFT, padx=5)

f4_label = tk.Label(features_frame, text="Number of Exposure Cycles:")
f4_label.pack(side=tk.LEFT, padx=5)

f4_entry = tk.Entry(features_frame)
f4_entry.pack(side=tk.LEFT, padx=5)
def predict():
    # Get feature values from entry widgets
    f1 = float(f1_entry.get())
    f2 = float(f2_entry.get())
    f3 = float(f3_entry.get())
    f4 = float(f4_entry.get())

    # Scale feature values
    features = [[f1, f2, f3, f4]]
    features_scaled = scaler.transform(features)

    # Get target model and predict
    if target_var.get() == "load":
        model = model_load
    else:
        model = model_slip

    y_pred = model.predict(features_scaled)
    result_label.config(text="Predicted value: {}".format(y_pred))



# Prediction button
predict_button = tk.Button(root, text="Predict", font=("Arial", 16), command=predict)
predict_button.pack(pady=10)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

def predict():
    # Get feature values from entry widgets
    f1 = float(f1_entry.get())
    f2 = float(f2_entry.get())
    f3 = float(f3_entry.get())
    f4 = float(f4_entry.get())

    # Scale feature values
    features = [[f1, f2, f3, f4]]
    features_scaled = scaler.transform(features)

    # Get target model and predict
    if target_var.get() == "load":
        model = model_load
    else:
        model = model_slip

    y_pred = model.predict(features_scaled)
    result_label.config(text="Predicted value: {}".format(y_pred))

  
    

root.mainloop()

