# -*- coding: utf-8 -*-

import json

def read_json(path):
    json_value = None
    with open(path, 'r', encoding='utf-8') as file:
        json_string = file.read()
        
        json_value = json.loads(json_string)
    
    return json_value
    # contents = open(path, "r").read() 
    # data = [json.loads(str(item)) for item in contents.strip().split('\n')] 
    # return data