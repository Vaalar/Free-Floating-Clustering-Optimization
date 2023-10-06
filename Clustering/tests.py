import pandas as pd
from datetime import datetime, timedelta

file = pd.read_csv('Clustering/Datasets/primeras_ultimas.csv')

groups = file.groupby('id')

for group_name, group in groups:
    interval = group.groupby('route_code')
    for interval_code, interval_groups in interval:
        trip_interval = []
        for idx, entry in interval_groups.iterrows():
            trip_interval.append(datetime.strptime((entry)['timestamp'], '%Y-%m-%d %H:%M:%S%z'))
        trip_interval[0] = trip_interval[0] + timedelta(minutes=30)
        print(interval.groups)
        
        input()