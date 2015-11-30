import os
import csv
import sys
import numpy as np
from datetime import date
from time import mktime

##
# This is a placeholder file; eventually, this will perform the preprocessing step
##

if __name__ == '__main__':
  with open('../data/train.csv') as csvfile:
    reader = csv.reader(csvfile)

    rawlabels = reader.next()
    labels = {}
    for i in range(len(rawlabels)):
      labels[rawlabels[i]] = i

    labels_to_keep = ['Date', 'Species', 'Latitude', 'Longitude', 'NumMosquitos']
    species = {} # maps a species string to an integer
    species_ct  = 0  # total number of species


    data = []
    classes = []
    for row in reader:
      parsed_row = [row[labels[s]] for s in labels_to_keep]
      
      # Convert date to number
      date_parts = parsed_row[0].split('-')
      parsed_row[0] = mktime(date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2])).timetuple())

      # Convert species to number
      if parsed_row[1] not in species:
        species[parsed_row[1]] = species_ct
        species_ct += 1
      parsed_row[1] = species[parsed_row[1]]

      # Convert lat/long to numbers (instead of strings)
      parsed_row[2] = float(parsed_row[2])
      parsed_row[3] = float(parsed_row[3])
      parsed_row[4] = int(parsed_row[4])

      classes.append(labels['WnvPresent'])
      data.append(parsed_row)

    X = np.array(data)
    print X

