import os
import csv
import sys
import numpy as np
from sys import argv
from time import mktime
from operator import add
from datetime import date
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    labels = ['WNV', 'No WNV']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

        cont_data.append(int(date_parts[1])) # month feature
        cont_data.append(int(date_parts[2])) # day feature

      lastRow = row

  return data, spec_data, mosCount, classes

def read_weather_data(fpath, s='1'):
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
      r1 = row
      r2 = reader.next()
      # Only consider station 1
      if row[labels['Station']] != '1':
        continue

      cont_row = [r1[labels[s]] for s in cont_labels] + [r2[labels[s]] for s in cont_labels[1:]]# get the columns we want

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
    total_data.append(np.concatenate((test_data[j][1:], weather_data[i][1:]), axis=0))

  return np.array(total_data)
	
##
# Performs preprocessing and some classification
##
if __name__ == '__main__':
  
  print "===== Reading Files ====="
  # Read the files
  data, spec_data, mos_count, classes = read_file('../data/train.csv')
  weather_data = read_weather_data('../data/weather.csv')

  print np.array(weather_data).shape

  data = merge_data(data, weather_data)

  if '-m' in argv:
    with open('../data/traindata.csv', 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      for i in range(len(data)):
        csvwriter.writerow(list(data[i]) + [mos_count[i], classes[i]])

    print 'Wrote merged training data file\n'

  # Do processing
  X = process(spec_data, data)


  ## Read test file:
  test_cdata, test_ddata, dummy, dummy2 = read_file('../data/test.csv', isTraining=False)
  test_cdata = merge_data(test_cdata, weather_data)

  testX = process(test_ddata, test_cdata)

  print "====Train Classifiers===="
  rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100, class_weight='balanced')
  knn = KNeighborsClassifier(5)
  nb = GaussianNB()
  dt = DecisionTreeClassifier(random_state=0, criterion='entropy')
  svmC = 85 
  svm = SVC(class_weight = 'balanced', C=svmC, kernel='poly') 

  labels = ['nb', 'rfc', 'dt', 'knn', 'svm']
  machines = [nb, rfc, dt, knn, svm]
  for i in range(len(machines)):
    machines[i].fit(X, classes)
    print 'Fit ' + labels[i]

  if '-o' in argv:
    for i in range(len(machines)):
      y_hat = machines[i].predict(testX)
      with open('../data/testoutput-' + labels[i] + '.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'WnvPresent'])
        for j in range(len(y_hat)):
          csvwriter.writerow([j+1, y_hat[j]])

      print 'Testing response written for machine ' + labels[i]

  
  if '-cv' in argv:
    print "====Cross Validation===="
    for i in range(len(machines)):
      cval_scores = cross_val_score(machines[i], X, classes, cv=3)
      print "%s: %0.5f (+/- %0.5f) " % (labels[i], cval_scores.mean(), cval_scores.std() * 2)

  # Confusion Matrix
  if '-stat' in argv:
    print "====Confusion Matrix===="
    
    for label, m in zip(labels, machines):
      X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.33, random_state=42, stratify=classes)
      m.fit(X_train, y_train)
      y_pred = m.predict(X_test)
      cm = confusion_matrix(y_test, y_pred, labels=[1, 0]).astype(np.float)
      print(cm)
      print "Actual positive samples: %d" % (sum(y_test),)
      plt.figure()
      plot_confusion_matrix(cm, title='Confusion Matrix for ' + label)

      # Precision & Recall:
      print "\n====Statistics for %s====" % (label,)
      prec = cm[0][0] / (cm[0][0] + cm[1][0])
      rec  = cm[0][0] / (cm[0][0] + cm[0][1])
      f1 = 2*prec*rec/(prec + rec)
      print "Precision: %.4f" % (prec,)
      print "Recall:    %.4f" % (rec,)
      print "F1 Score:  %.4f" % (f1,)

  if '-pca' in argv:
    #### Do the PCA plot
    plt.figure()

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    
    no = np.array([X_r[i] for i in range(len(X_r)) if  classes[i] == 0])
    yes = np.array([X_r[i] for i in range(len(X_r)) if  classes[i] == 1])
    
    for c, data, target_name in zip("gr", [no, yes], ["No WNV", "WNV"]):
      plt.scatter(data[:,0], data[:, 1], c=c, s=20, label=target_name)
    plt.legend()
  

  if '-pca' in argv or '-stat' in argv:
    # Print any plots
    plt.show()
