import json
class server():
    def __init__(self):
        self.text = {}
        
    def print(self,key,value):
        self.text[key]=value
    
    def send(self):
        a = self.text
        a = json.dumps(a)
        self.text = {}
        return a
    
