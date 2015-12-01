import os
import csv
import sys
import numpy as np
from datetime import date
from time import mktime
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

##
# read files
##
def read_file(fpath):
  with open(fpath) as csvfile:
    reader = csv.reader(csvfile)

    rawlabels = reader.next()
    labels = {}
    for i in range(len(rawlabels)):
      labels[rawlabels[i]] = i

    cont_labels = ['Date', 'Latitude', 'Longitude', 'NumMosquitos']
    species = {} # maps a species string to an integer
    species_ct  = 0  # total number of species


    data = []
    spec_data = []
    classes = []
    for row in reader:
      parsed_row = [row[labels[s]] for s in cont_labels]
      
      # Convert date to number
      date_parts = parsed_row[0].split('-')
      parsed_row[0] = mktime(date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2])).timetuple())

      # Convert species to number
      if parsed_row[1] not in species:
        species[parsed_row[1]] = species_ct
        species_ct += 1
      spec_data.append([species[parsed_row[1]]])

      # Convert lat/long to numbers (instead of strings)
      parsed_row[1] = float(parsed_row[1])
      parsed_row[2] = float(parsed_row[2])
      parsed_row[3] = int(parsed_row[3])

      classes.append(row[labels['WnvPresent']])
      data.append(parsed_row)

  return data, spec_data, classes
  

##
# Performs preprocessing and some classification
##
if __name__ == '__main__':
  # Read the files
  data, spec_data, classes = read_file('../data/train.csv')

  # Create discrete and continuous data matrices
  discrete_X = np.array(spec_data)
  cont_X = np.array(data)

  # Discrete basis representation
  enc = OneHotEncoder()
  enc.fit(discrete_X)
  discrete_X = enc.transform(discrete_X).toarray()

  # Continuous scaling
  scaler = StandardScaler()
  scaler.fit(cont_X)
  cont_X = scaler.transform(cont_X)

  # Merge to one array
  X = np.concatenate((discrete_X, cont_X), axis=1) 

  for k in ['rbf', 'linear', 'poly']:
    print "%s & " % k
    svm = SVC(kernel=k)

    cval_scores = cross_val_score(svm, X, classes, cv=5)

    print "%0.5f (+/- %0.5f) & " % (cval_scores.mean(), cval_scores.std() * 2)
    
