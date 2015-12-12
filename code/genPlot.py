import os
import csv
import sys
from math import log
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from time import mktime

x = []
y = []
with open('../data/train.csv') as csvfile:
  reader = csv.reader(csvfile)

  # Read in data labels
  rawlabels = reader.next()
  labels = {}
  for i in range(len(rawlabels)):
    labels[rawlabels[i]] = i
  
  pos_ct = {}
  neg_ct = {}
  lastRow = [0, 0, 0]
  x = -1
  y = 0
  for row in reader:
    if row[:-2] == lastRow[:-2]:
      x += int(row[labels['NumMosquitos']])
      y = max(y, int(row[labels['WnvPresent']]))
    else:
      if y==1:
        pos_ct[x] = pos_ct.get(x, 0) + 1
      else:
        neg_ct[x] = neg_ct.get(x, 0) + 1

      x = int(row[labels['NumMosquitos']])
      y = int(row[labels['WnvPresent']])

    lastRow = row

  if y==1:
    pos_ct[x] = pos_ct.get(x, 0) + 1
  else:
    neg_ct[x] = neg_ct.get(x, 0) + 1


  x = [0]
  y = [0]
  s = [0]
  for pos_x in pos_ct:
    x.append(log(pos_x))
    y.append(1)
    s.append(100*pos_ct[pos_x] / (pos_ct[pos_x] + neg_ct.get(pos_x, 0)))

  for neg_x in neg_ct:
    if neg_x == -1:
      continue
    x.append(log(neg_x))
    y.append(0)
    s.append(100*neg_ct[neg_x] / (neg_ct[neg_x] + pos_ct.get(neg_x, 0)))


plt.title('Class vs. Number of Mosquitos')
plt.xlabel('$\log($num_mosquitos$)$')
plt.ylabel('Wnv_Present')
plt.scatter(x, y, s = s) 
plt.show()
