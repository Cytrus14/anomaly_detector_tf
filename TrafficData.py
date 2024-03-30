import mmh3
import os
import threading
import numpy as np
import pandas as pd

class TrafficData:
    def __init__(self):
        columns=["source", "destination", "protocol", "_time_delta_sum", "source_port", 
        "destination_port", 'avg_time_delta', "bytes_total", 'bytes_avg', "frames_total"]
        # DataFames for storing captured traffic data
        self.intermediate_traffic_df = pd.DataFrame(columns=columns)
        self.traffic_data_buffer = pd.DataFrame(columns=columns)
        
        # Locks for data objects
        self.intermediate_traffic_lock = threading.Lock()
        self.traffic_data_buffer_lock = threading.Lock()

        # Flags for writing captured traffic to file
        self.header_written = False
        self.output_file_exists_check = False

        self.data_points_captured_count = 0

    def update_intermediate_traffic_data(self, tshark_line):
        line = tshark_line.decode('utf-8') # Convert byte stream to string
        fields = line.split('\t') # Split tshark output into individual fields
        hash_id = mmh3.hash(fields[0] + fields[1] + fields[4])
        self.intermediate_traffic_lock.acquire()
        if hash_id not in self.intermediate_traffic_df.index:
            new_row = {}
            # sometimes tshark concatenates source and destination
            # splitting by ',' should fix this issue
            source_temp = fields[0].split(',')
            source_temp = source_temp[0]
            destination_temp = fields[1].split(',')
            if len(destination_temp) > 1:
                destination_temp = destination_temp[1]
            else:
                destination_temp = destination_temp[0]
            # check whether source are designation addresses are not empty
            if source_temp == '':
                source_temp = '0.0.0.0'
            if destination_temp == '':
                destination_temp = '0.0.0.0'
            new_row['source'] = source_temp
            new_row['destination'] = destination_temp
            new_row['protocol'] = fields[4]
            # if the tcp protocol is used
            if len(fields[5]) > 0: 
                new_row['source_port'] = int(fields[5])
                new_row['destination_port'] = int(fields[6])
            # if the udp protocol is used
            if len(fields[7]) > 0:
                new_row['source_port'] = int(fields[7])
                new_row['destination_port'] = int(fields[8])
            new_row['bytes_total'] = 0
            new_row['_time_delta_sum'] = 0
            new_row['frames_total'] = 0
            self.intermediate_traffic_df.loc[hash_id] = new_row

        self.intermediate_traffic_df.loc[hash_id, 'bytes_total'] += int(fields[2])
        self.intermediate_traffic_df.loc[hash_id, '_time_delta_sum'] += float(fields[3])
        self.intermediate_traffic_df.loc[hash_id, 'frames_total'] += 1
        self.intermediate_traffic_lock.release()

    def buffer_traffic_data(self):
        self.intermediate_traffic_lock.acquire()
        traffic_data = self.intermediate_traffic_df.copy(deep=True)
        # clear the intermediate traffic data
        self.intermediate_traffic_df.drop(self.intermediate_traffic_df.index, inplace=True)
        self.intermediate_traffic_lock.release()
        # calculate the final values for selected fields
        for hash_id, row in traffic_data.iterrows():
            if row['frames_total'] != 0:
                row['avg_time_delta'] = row['_time_delta_sum'] / row['frames_total']
                row['bytes_avg'] = row['bytes_total'] / row['frames_total']
                self.traffic_data_buffer.loc[self.data_points_captured_count] = row
                self.data_points_captured_count += 1
    
    def get_buffer_size(self):
        return len(self.traffic_data_buffer)
    
    def write_buffer_to_csv(self):
        if os.path.exists("baseline_traffic.csv") and not self.output_file_exists_check:
            os.remove("baseline_traffic.csv")
            self.output_file_exists_check = True
        columns=["source", "destination", "protocol", "source_port", 
        "destination_port", 'avg_time_delta', "bytes_total", 'bytes_avg', "frames_total"]
        self.traffic_data_buffer_lock.acquire()
        if not self.header_written:
            self.traffic_data_buffer.to_csv('baseline_traffic.csv', mode='a', columns=columns, header=True, index=False)
            self.header_written = True
        else:
            self.traffic_data_buffer.to_csv('baseline_traffic.csv', mode='a', columns=columns, header=False, index=False)
        # clear the buffer
        self.traffic_data_buffer.drop(self.traffic_data_buffer.index, inplace=True)
        self.traffic_data_buffer_lock.release()
    
    def get_captured_data_points_count(self):
        return self.data_points_captured_count
    
    def get_traffic_data_buffer(self):
        self.traffic_data_buffer_lock.acquire()
        traffic_data_buffer_copy = self.traffic_data_buffer.copy(deep=True)
        self.traffic_data_buffer_lock.release()
        return traffic_data_buffer_copy
    
    def clear_traffic_data_buffer(self):
        self.traffic_data_buffer_lock.acquire()
        self.traffic_data_buffer.drop(self.traffic_data_buffer.index, inplace=True)
        self.traffic_data_buffer_lock.release()

