
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data_met(client_id, source, elements, start_date, end_date):
  
    # Define endpoint and parameters
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
        'sources': source,
        'elements': ','.join(elements),
        'referencetime': start_date.tz_localize('Europe/Oslo').tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S') + '/' + end_date.tz_localize('Europe/Oslo').tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S'),
        'timeresolutions':'PT1H',
    }
    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()

    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json['data']
        print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])

    # Convert json to dataframe format
    df = pd.DataFrame()
    print('len data = {}'.format(len(data)))
    for i in range(len(data)):
        row = pd.DataFrame(data[i]['observations'])
        row['referenceTime'] = data[i]['referenceTime']
        row['sourceId'] = data[i]['sourceId']
        df = df.append(row)

    df = df.reset_index()

    # Convert the time value to something Python understands
    df['referenceTime'] = pd.to_datetime(df['referenceTime'],utc=True).dt.tz_convert('Europe/Oslo')
    df=df.set_index('referenceTime')
    df.index=df.index.tz_localize(None)

    output = pd.DataFrame()
    output['time'] = pd.date_range(start_date,end_date,freq='H',closed='left')
    output = output.set_index('time')
    #output['time'] = df['referenceTime'].unique()
    #output.set_index('time',inplace=True)
    output.index = output.index.tz_localize(None)
    for e in elements:
        #output[e] = df['value'][df['elementId']==e].values
        output = output.join(df[df['elementId']==e]['value'])
        output[e] = output['value']
        output=output.drop(['value'],axis=1)
    
    return output
