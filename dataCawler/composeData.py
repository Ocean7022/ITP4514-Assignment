import json

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
print('Total news:', len(newData))

with open('../data/newsData.json', 'w') as f:
    json.dump(newData, f)