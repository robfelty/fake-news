#!/usr/bin/env python
# print distributions of train,dev, and test sets
import csv
categories = {'agree':0, 'disagree':0, 'discuss':0, 'unrelated':0}


print("partition\t"+"\t".join(sorted(categories)))
for partition in ['train', 'dev', 'test']:
    header=True
    with open(partition+'.csv', newline='',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if header:
                header=False
                continue
            categories[row[2]]+=1
    distribution=partition+"\t"
    for category in sorted(categories):
        distribution += '%d %0.2f\t' % (categories[category], categories[category]/sum(categories.values())*100)
    print(distribution)
