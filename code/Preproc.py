import os
import csv
import sys
import numpy as np
from datetime import date
from time import mktime
from operator import add
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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


    pos_ct = {}
    neg_ct = {}
    lastRow = [0, 0, 0]
    mosCount = []
    for row in reader:
      if row[:-2] == lastRow[:-2]:
        # If this row is a duplicate of the prior row
        # only update the count of mosquitos and class label
        if isTraining:
          mosCount[-1] += int(row[labels['NumMosquitos']])
          classes[-1]  = max(classes[-1], int(row[labels['WnvPresent']]))
      else:
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
          mosCount.append(int(row[labels['NumMosquitos']]))
          classes.append(int(row[labels['WnvPresent']]))

        data.append(cont_data)

      lastRow = row

  return data, spec_data, mosCount, classes

def read_weather_data(fpath):
  cont_data = []
  with open(fpath) as csvfile:
    reader = csv.reader(csvfile)

    # Get CSV headers
    rawlabels = reader.next()
    labels = {}
    for i in range(len(rawlabels)):
      labels[rawlabels[i]] = i

    # Labels we want:
    cont_labels = ['Date', 'Tavg', 'Depart', 'DewPoint', 'StnPressure', 'AvgSpeed', 'Tmax', 'Tmin', 'WetBulb', 'PrecipTotal', 'DiffPressure']

    for row in reader:
      # Only consider station 1
      if row[labels['Station']] != '1':
        continue

      cont_row = [row[labels[s]] for s in cont_labels] # get the columns we want

      # Convert date to number
      date_parts = cont_row[0].split('-')
      cont_row[0] = mktime(date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2])).timetuple())

      # Convert strings to integers/floats
      for i in range(1, len(cont_row)):
        if cont_row[i] == '  T':
          cont_row[i] = 0.01 # trace precip
        elif cont_row[i] == 'M':
          cont_row[i] = 'NaN'
        elif cont_row[i] == '#VALUE!':
          cont_row[i] = 'NaN'
        else:
          cont_row[i] = float(cont_row[i])

      cont_data.append(cont_row)

    return cont_data


def process(discrete, cont):
  # Create discrete and continuous data matrices
  discrete_X = np.array(discrete)
  cont_X = np.array(cont)

  # Impute discrete values
  imp = Imputer(strategy='most_frequent')
  discrete_X = imp.fit_transform(discrete_X)

  # Impute continuous values
  imp_c = Imputer(strategy='mean')
  cont_X = imp_c.fit_transform(cont_X)

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
  total_data = [] 
     
# below is code that is separate from what you have above.  Take a look at it, the idea
# here is to take the data from test_data, and compare it to weather_data.  It cycles through all
# the weather_data dates until it finds a match, then it concatenates the two data entries into
# total_data.  Does this for all the points in test_data. This takes advantage of the fact that
# both the weather_data and total_data time ascending order.       
  i = 0
  for j in range(len(test_data)): 
    while(weather_data[i][0] != test_data[j][0]):
      i+= 1
    total_data.append(np.concatenate((test_data[j][:], weather_data[i][1:]), axis=0))

  return np.array(total_data)
	
##
# Performs preprocessing and some classification
##
if __name__ == '__main__':
  # Read the files
  data, spec_data, mos_count, classes = read_file('../data/train.csv')
  weather_data = read_weather_data('../data/weather.csv')

  print len(mos_count)
  print len(data)

  data = merge_data(data, weather_data)

  with open('../data/traindata.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in range(len(data)):
      csvwriter.writerow(list(data[i]) + [mos_count[i], classes[i]])

  print 'Wrote merged file'

  X = process(spec_data, data)


  ## Read test file:
  test_cdata, test_ddata, dummy, dummy2 = read_file('../data/test.csv', isTraining=False)
  test_cdata = merge_data(test_cdata, weather_data)

  testX = process(test_ddata, test_cdata)

  #svm = SVC(kernel='rbf')
  #svm.fit(X, classes)
  #y_hat = svm.predict(testX)

  rfc = RandomForestClassifier(n_jobs=-1, n_estimators=5)
  knn = KNeighborsClassifier(5)
  nb = GaussianNB()
  dt = DecisionTreeClassifier(random_state=0, criterion='entropy')
  svm = SVC(class_weight = 'balanced', C=100, kernel='poly')

  nb.fit(X, classes)
  print "Fit Naive Bayes"
  rfc.fit(X, classes)
  print "Fit Random Forest"
  dt.fit(X, classes)
  print "Fit D tree"
  knn.fit(X, classes)
  print "Fit KNN"
  svm.fit(X, classes)
  print "Fit SVM"

  y_hat1 = nb.predict(testX)
  #y_hat2 = rfc.predict(testX)
  #y_hat3 = dt.predict(testX)
  #y_hat4 = knn.predict(testX)
  y_hat5 = svm.predict(testX)
  with open('../data/testoutput.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['id', 'WnvPresent'])
    for i in range(len(y_hat1)):
      classification = y_hat5[i]#max([y_hat1[i], y_hat2[i], y_hat3[i], y_hat4[i]])
      csvwriter.writerow([i+1, classification])

  print 'File written'

  cval_scores = cross_val_score(rfc, X, classes, cv=3)
  print "%s: %0.5f (+/- %0.5f) " % ('rfc', cval_scores.mean(), cval_scores.std() * 2)
  cval_scores = cross_val_score(dt, X, classes, cv=3)
  print "%s: %0.5f (+/- %0.5f) " % ('dt', cval_scores.mean(), cval_scores.std() * 2)
  cval_scores = cross_val_score(nb, X, classes, cv=3)
  print "%s: %0.5f (+/- %0.5f) " % ('nb', cval_scores.mean(), cval_scores.std() * 2)

  cval_scores = cross_val_score(knn, X, classes, cv=3)
  print "%s: %0.5f (+/- %0.5f) " % ('knn', cval_scores.mean(), cval_scores.std() * 2)

  cval_scores = cross_val_score(svm, X, classes, cv=3)

  print "%s: %0.5f (+/- %0.5f) " % ('SVC C=...', cval_scores.mean(), cval_scores.std() * 2)
  
  for k in ['linear', 'poly', 'rbf']:
    svm = SVC(C=10, kernel=k)

    cval_scores = cross_val_score(svm, X, classes, cv=3)

    print "%s: %0.5f (+/- %0.5f) " % (k, cval_scores.mean(), cval_scores.std() * 2)
