# You will write all the queries under different methods
import requests
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Model:

    global db

    db = {
    "name": "http://dbcontainer:5000/api/v1/datastore",
    "endpoint": [
        "read",
        "write_pred",
        "write_input",
        "delete",
        "update",
        "scan",
    ]
    }   
    
    def addOutput(links, output, text):
        payload = {"objtype": "prediction", "objkey": output}
        url = db['name'] + '/' + db['endpoint'][5]
        response = requests.get(url, params = payload)
        count = int(response.json()['Count'])
        str_link = "\n".join([str(i) for i in links])
        if count == 0:
            url = db['name'] + '/' + db['endpoint'][1]
            response = requests.post(url, json = {"objtype":"prediction","label":output,"frequency":"1", "links":str_link})
        else:

            url = db['name'] + '/' + db['endpoint'][4]
            response = requests.put(url, params = {"objtype": "prediction", "objkey": output}, json={"frequency":"2"})
        
        url = db['name'] + '/' + db['endpoint'][2]
        response = requests.post(url, json = {"objtype":"input", "input":text, "output":output})
        return response.json()




