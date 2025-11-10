#!/user/bin/env python3
# map used to trace generated logs, 
import json
with open("logs.json", "r") as fp:
    logs = json.load(fp)
malog = logs["malog"]
checkpoint = logs["checkpoint"]
