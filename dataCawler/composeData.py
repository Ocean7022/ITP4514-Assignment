import json
from tqdm import tqdm

def remove_special_characters(input_string):
    special_chars = ['\u00ad', '\u2018', '\u2019', '\u201c', '\u201d', '\u00AD', '\n', '\t', '\r', '\f']
    for char in special_chars:
        input_string = input_string.replace(char, "")
    return input_string

with open('./newsData/nbcnewsData.json', 'r') as f:
    nbcnewsData = json.load(f)

with open('./newsData/theStandard.json', 'r') as f:
    theStandardData = json.load(f)

with open('./newsData/aljazeerayData.json', 'r') as f:
    aljazeerayData = json.load(f)

print('NBC:', len(nbcnewsData))
print('The Standar:', len(theStandardData))
print('Aljazeeray:', len(aljazeerayData))

newData = nbcnewsData + theStandardData + aljazeerayData
category_counts = {}

for data in tqdm(newData, desc="Cleaning", unit="item"):
    data['title'] = remove_special_characters(data['title'])
    data['content'] = remove_special_characters(data['content'])

    category = data['category']
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

with open('../data/newsDataSet.json', 'w', encoding = 'utf-8') as f:
    json.dump(newData, f, indent = 4)

for category, count in category_counts.items():
    print(f"Category: {category}, Count: {count}")



