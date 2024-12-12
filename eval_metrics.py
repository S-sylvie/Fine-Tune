import json
import re

def extract_json_from_string(string):
    json_pattern = re.compile(r'\{.*?\}')
    match = json_pattern.search(string)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def compare_json_objects(json1, json2):
    return json1 == json2

def calculate_joint_acc(string1, string2):
    json1 = extract_json_from_string(string1)
    json2 = extract_json_from_string(string2)
    if json1 and json2:
        return compare_json_objects(json1, json2)
    return False

