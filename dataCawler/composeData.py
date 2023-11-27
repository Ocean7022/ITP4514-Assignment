import json, os
from tqdm import tqdm

def remove_special_characters(input_string):
    special_chars = ['\u00ad', '\u2018', '\u2019', '\u201c', '\u201d', '\u00AD', '\n', '\t', '\r', '\f']
    for char in special_chars:
        input_string = input_string.replace(char, "")
    return input_string

def cutFile(path, part = 3):
    with open(path, 'r') as f:
        data = json.load(f)

    fileName = os.path.basename(path)
    numOfEachPart = int(len(data) / part)

    for i in range(part):
        if i == part - 1:
            partData = data[i*numOfEachPart:]
        else:
            partData = data[i*numOfEachPart:(i + 1)*numOfEachPart]

        with open(f'{path[:-len(fileName)]}{fileName[:-5]}-part{i + 1}.json', 'w') as f:
            json.dump(partData, f, indent=4)

with open('./newsData/nbcnewsData.json', 'r') as f:
    nbcnewsData = json.load(f)

with open('./newsData/theStandard.json', 'r') as f:
    theStandardData = json.load(f)

cutFile('./newsData/aljazeerayData.json')
aljazeerayData = []
for i in range(3):
    with open(f'./newsData/aljazeerayData-part{i + 1}.json', 'r') as f:
        aljazeerayData += json.load(f)

with open('./newsData/cnnnewsData.json', 'r') as f:
    cnnnewsData = json.load(f)

print('NBC:', len(nbcnewsData))
print('The Standar:', len(theStandardData))
print('Aljazeeray:', len(aljazeerayData))
print('CNN:', len(cnnnewsData))

newData = nbcnewsData + theStandardData + aljazeerayData + cnnnewsData
category_counts = {}

for data in tqdm(newData, desc="Cleaning", unit="item", ncols=80):
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
    print(f"{category:<{10}} - Count: {count}")

cutFile('../data/newsDataSet.json', 4)
