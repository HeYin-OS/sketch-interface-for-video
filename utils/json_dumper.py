import json


def json_dumper(obj):
    with open('stroke.json', 'w') as f:
        obj_list = [coord.tolist() for coord in obj]
        json.dump(obj_list, f)
