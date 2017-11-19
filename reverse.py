import json

with open("hobbies.json") as f:
  h = json.load(f)

result = {}
for hobby, reddits in h.items():
  for reddit in reddits:
    if result.get(reddit, None) is not None:
      result[reddit].append(hobby)
    else:
      result[reddit] = [hobby]

with open("hobbiesReversed.json", "w+") as f:
  json.dump(result, f)
