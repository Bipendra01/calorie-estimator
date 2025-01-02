import os

dataset_path = 'data/dummyDataSet/images'
food_labels = sorted(os.listdir(dataset_path))

with open('food_labels.txt', 'w') as f:
    for label in food_labels:
        f.write(label + '\n')