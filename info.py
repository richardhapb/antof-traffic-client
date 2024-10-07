'''
Actualiza el archivo waze.json con la información de la API de Waze.
'''

import json
import time
import datetime
import pytz
import requests

timezone = pytz.timezone("America/Santiago")
# 15000 = 2.5 minutos, el promedio entre periodos
now = int(datetime.datetime.now(timezone).timestamp()) * 1000 - 150000

def read_json(filename='data/waze.json'):
    '''
    Lee un archivo json y lo convierte en un diccionario.
    '''
    with open(filename, 'r', encoding="utf8") as file:
        d = json.load(file)
        file.close()
        return d

def write_json(dat, filename='data/waze.json'):
    '''
    Escribe un diccionario en un archivo json.
    '''
    with open(filename, 'w', encoding="utf8") as file:
        json.dump(dat, file, indent=4)
        file.close()

def verify_endings(datf, datr):

    try:
        irregularities = datf['irregularities']
        del datf['irregularities']
    except KeyError:
        irregularities = []
    try:
        r_irregularities = datr['irregularities']
        del datr['irregularities']
    except KeyError:
        r_irregularities = []

    for i in datf:
        for ij, j in enumerate(datf[i]):
            try:
                if j['uuid'] not in [l['uuid'] for k in datr for l in datr[k]]:
                    if "endreport" not in j:
                        datf[i][ij]['endreport'] = now
            except KeyError:
                print(json.dumps(j, indent=2))
                if j['id'] not in [l['id'] for l in r_irregularities]:
                    if "endreport" not in j:
                        datf[i][ij]['endreport'] = now

    return datf


def main():
    '''
    Actualiza el archivo waze.json con la información de la API de Waze.
    '''
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

    try:
        irregularities = f['irregularities']
        del f['irregularities']
    except KeyError:
        irregularities = []
    
    for i in data:
        for j in data[i]:
            try:
                if j['uuid'] not in [l['uuid'] for k in f for l in f[k]]:
                    f[i].append(j)
            except KeyError:
                if j['id'] not in [l['id'] for l in irregularities]:
                    irregularities.append(j)

    f['irregularities'] = irregularities
    f = verify_endings(f, data)
    write_json(f)

if __name__ == '__main__':
    # Se ejecuta cada 5 minutos.
    while True:
        print("Actualizando...")
        main()
        time.sleep(60 * 5)
