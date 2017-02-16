#!/usr/bin/env python
# requires python 3
import csv
import math
import random
from collections import defaultdict

bodies = {}
header = True
with open('train_bodies.csv', newline='',encoding='utf-8') as csvfile:
    body_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in body_reader:
        if header:
            header = False
            continue
        bodies[int(row[0])] = (row[1])
print('#bodies: %d' % (len(bodies)))

stances = defaultdict(list)
header = True
with open('train_stances.csv', newline='',encoding='utf-8') as csvfile:
    stance_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in stance_reader:
        if header:
            header = False
            continue
        stances[int(row[1])].append((row[0],row[2]))

print('#stances: %d' % (len(stances)))
assert len(bodies) == len(stances)
assert bodies.keys() == stances.keys()
train_percent = 0.7
dev_percent = 0.2
test_percent = 0.1
total_num = len(bodies)
ids = list(bodies.keys())
# use the same randomization each time
random.seed('foobar')
random.shuffle(ids)

def write_example(writer, body_id, bodies, stance):
    writer.writerow([body_id, stance[0], stance[1], bodies[body_id]])
def write_test(writer, body_id, bodies, stance):
    writer.writerow([stance[0], body_id, stance[1]])

train_num = math.floor(train_percent * total_num)
dev_num = math.floor(dev_percent * total_num)+train_num
train_file = open('train.csv', 'w', newline='', encoding='utf-8')
train_writer = csv.writer(train_file, delimiter=',', dialect='unix', quotechar='"',)
dev_file = open('dev.csv', 'w', newline='', encoding='utf-8')
dev_writer = csv.writer(dev_file, delimiter=',', dialect='unix', quotechar='"',)
test_file = open('test.csv', 'w', newline='', encoding='utf-8')
test_writer = csv.writer(test_file, delimiter=',', dialect='unix', quotechar='"',)
test_writer.writerow(['Headline', 'Body ID', 'Stance'])
sample = 10
for body_id in ids:
    #if body_id > sample: continue # uncomment to just make a really small sample
    for stance in stances[body_id]:
        if body_id < train_num:
            write_example(train_writer, body_id, bodies, stance)
        elif body_id < dev_num:
            write_example(dev_writer, body_id, bodies, stance)
        else:
            write_example(test_writer, body_id, bodies, stance)
