import json

with open('.\dailyBTC1.json', 'r') as startingData:
    jsonData = json.load(startingData)
    print(jsonData)