import os
import sys

raw_acc_file='./yolov3_16_4_raw_mAP.txt'
final_acc_file='./logs/yolov3_16_4_mAP.txt'
unique_accs = []
with open(raw_acc_file, 'r') as f:
  for i, line in enumerate(f.readlines()):
    if (i+1) % 4 == 0:
      acc = float(line) #round(float(line), 2)
      print(acc)
      unique_accs.append(acc)

with open(final_acc_file, 'w') as f:
  for acc in unique_accs:
    f.write(str(acc) + '\n')

