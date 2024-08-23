'''
Actualiza el archivo data.json con la información de la API de Waze.
'''

import json
import requests

def read_json(filename='data.json'):
    '''
    Lee un archivo json y lo convierte en un diccionario.
    '''
    with open(filename, 'r') as file:
        d = json.load(file)
        file.close()
        return d

def write_json(dat, filename='data.json'):
    '''
    Escribe un diccionario en un archivo json.
    '''
    with open(filename, 'w') as file:
        json.dump(dat, file, indent=4)
        file.close()



url = "https://www.waze.com/row-partnerhub-api/partners/18532407453/waze-feeds/d44195e2-2952-4b2f-8539-af8e85b661c5?format=1"
response = requests.get(url, timeout=10)

f = read_json()
data = response.json()


lists = [(key, value) for _, (key, value) in \
         enumerate(zip([i for i in f if isinstance(f[i], list)], \
                       [f[i] for i in f if isinstance(f[i], list)]))]

f = dict(lists)

lists = [(key, value) for _, (key, value) in \
         enumerate(zip([i for i in data if isinstance(data[i], list)],\
                        [data[i] for i in data if isinstance(data[i], list)]))]

data = dict(lists)


for i in data:
    for j in data[i]:
        if j['uuid'] not in [l['uuid'] for k in f for l in f[k]]:
            f[i].append(j)

write_json(f)
