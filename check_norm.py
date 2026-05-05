import json
f = open('/transfer/merged-v7/norm_stats.json')
s = json.load(f)
print(len(s['mean']), len(s['std']))
