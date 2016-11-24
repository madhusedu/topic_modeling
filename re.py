import json

review=[]
with open('yelp_academic_dataset_review.json') as json_data:
  data=json.load(json_data)
  for r in data[replace-'node_name']:
  review=review.append(r[replce-'column_name'])
