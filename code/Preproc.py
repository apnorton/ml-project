import os
import csv
import sys
import numpy as np
from datetime import date
from time import mktime
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

##
# read files
##

species = {'UNSPECIFIED CULEX':'NaN'} # maps a species string to an integer
species_ct  = 0  # total number of species
def read_file(fpath, isTraining=True):
  global species
  global species_ct

  with open(fpath) as csvfile:
    reader = csv.reader(csvfile)

    # Read in data labels
    rawlabels = reader.next()
    labels = {}
    for i in range(len(rawlabels)):
      labels[rawlabels[i]] = i

    # Continous variable labels
    cont_labels = ['Date', 'Latitude', 'Longitude']

    data = [] # This stores the total continuous dataset
    spec_data = [] # This is the discrete data set (species data)
    classes = [] # The class labels

    for row in reader:
      cont_data = [row[labels[s]] for s in cont_labels] # get continuous data

      # Convert date to number
      date_parts = cont_data[0].split('-')
      cont_data[0] = mktime(date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2])).timetuple())

      # Convert species to number
      if row[labels['Species']] not in species:
        species[row[labels['Species']]] = species_ct
        species_ct += 1
      spec_data.append([species[row[labels['Species']]]])

      # Convert lat/long to numbers (instead of strings)
      cont_data[1] = float(cont_data[1])
      cont_data[2] = float(cont_data[2])

      if isTraining:
        classes.append(int(row[labels['WnvPresent']]))

      data.append(cont_data)

  return data, spec_data, classes

def read_weather_data(fname):
  cont_data = []
  with open(fpath) as csvfile:
    reader = csv.reader(csvfile)

    # Get CSV headers
    rawlabels = reader.next()
    labels = {}
    for i in range(len(rawlabels)):
      labels[rawlabels[i]] = i

    # Labels we want:
    cont_labels = ['Date', 'Tavg', 'Depart', 'DewPoint', 'StnPressure', 'AvgSpeed']

    for row in reader:
      # Only consider station 1
      if row[labels['Station']] != '1':
        continue

      cont_row = [row[labels[s]] for s in cont_labels] # get the columns we want

      # Convert date to number
      date_parts = cont_row[0].split('-')
      cont_row[0] = mktime(date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2])).timetuple())

      # Convert strings to integers/floats
      cont_row[labels['Tavg']] = int(cont_row[labels['Tavg']])
      cont_row[labels['Depart']] = int(cont_row[labels['Depart']])
      cont_row[labels['DewPoint']] = int(cont_row[labels['DewPoint']])
      cont_row[labels['StnPressure']] = float(cont_row[labels['StnPressure']])
      cont_row[labels['AvgSpeed']] = float(cont_row[labels['AvgSpeed']])

      cont_data.append(cont_row)

    return cont_data


def process(discrete, cont):
  # Create discrete and continuous data matrices
  discrete_X = np.array(discrete)
  cont_X = np.array(cont)

  # Impute discrete valueso
  imp = Imputer(strategy='most_frequent')
  discrete_X = imp.fit_transform(discrete_X)

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
  return X

def merge_data(test_data, weather_data):
  total_data = None
  i = 0
  for t_row in test_data:
    while(t_row[0] != weather_data[i][0])
      i += 1


##
# Performs preprocessing and some classification
##
if __name__ == '__main__':
  # Read the files
  data, spec_data, classes = read_file('../data/train.csv')

  X = process(spec_data, data)


  ## Read test file:
  test_cdata, test_ddata, dummy = read_file('../data/test.csv', isTraining=False)

  testX = process(test_ddata, test_cdata)

  svm = SVC(kernel='rbf')
  svm.fit(X, classes)
  y_hat = svm.predict(testX)
  with open('../data/testoutput.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['id', 'WnvPresent'])
    for i in range(len(y_hat)):
      csvwriter.writerow([i+1, y_hat[i]])

  print 'File written'


  for k in ['linear', 'poly', 'rbf']:
    svm = SVC(kernel=k)

    cval_scores = cross_val_score(svm, X, classes, cv=5)

    print "%s: %0.5f (+/- %0.5f) " % (k, cval_scores.mean(), cval_scores.std() * 2)
