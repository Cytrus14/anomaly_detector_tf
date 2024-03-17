import copy
import mmh3
import numpy as np
import pandas as pd
from Capture_window import CaptureWindow

class Connections:
    def __init__(self):
        self.capture_windows = {} # capture windows used for data processing
        self.capture_windows_visual = {} # capture windows used for visualization
    
    def update_capture_windows(self, line):
        line = line.decode('utf-8') # Convert byte stream to string
        fields = line.split('\t') # Split tshark output into individual fields
        # Create a capture windows key by comining the source ip, destination ip and protocol fields
        key = str(mmh3.hash(fields[0] + fields[1] + fields[4]))
        if key not in self.capture_windows:
            self.capture_windows[key] = CaptureWindow()
            self.capture_windows[key].source = fields[0]
            self.capture_windows[key].destination = fields[1]
            self.capture_windows[key].protocol = fields[4]
            # if the tcp protocol is used
            if len(fields[5]) > 0: 
                self.capture_windows[key].source_port = int(fields[5])
                self.capture_windows[key].destination_port = int(fields[6])
            # if the udp protocol is used
            if len(fields[7]) > 0:
                self.capture_windows[key].source_port = int(fields[7])
                self.capture_windows[key].destination_port = int(fields[8])
        self.capture_windows[key].bytes_total += int(fields[2])
        self.capture_windows[key]._time_delta_sum += float(fields[3])
        self.capture_windows[key].frames_total += 1
        self.capture_windows_visual[key] = copy.copy(self.capture_windows[key])
    
    def finalize_capture_windows(self):
        for key in self.capture_windows.keys():
            if self.capture_windows[key].frames_total != 0:
                self.capture_windows[key].avg_time_delta = self.capture_windows[key]._time_delta_sum / self.capture_windows[key].frames_total
                self.capture_windows[key].bytes_avg = self.capture_windows[key].bytes_total / self.capture_windows[key].frames_total
    
    def clear_capture_windows(self):
        self.capture_windows = {}
    
    def get_capture_windows(self):
        return copy.copy(self.capture_windows)
    
    def get_capture_window_visual(self):
        return copy.copy(self.capture_windows_visual)
    
    def clear_capture_windows_visual(self):
        self.capture_windows_visual = {}
    
    def convert_to_pd_dataframe(self):
        rows = []
        for key in self.capture_windows.keys():
            row = []
            # if len(self.capture_windows[key].source) > len("185.213.154.117"):
            #     print(self.capture_windows[key].source)
            if self.capture_windows[key].source == '':
                row.append(np.NaN)
            else:
                row.append(self.capture_windows[key].source)
            if self.capture_windows[key].destination == '':
                row.append(np.NaN)
            else:
                row.append(self.capture_windows[key].destination)
            row.append(self.capture_windows[key].protocol)
            row.append(self.capture_windows[key].bytes_avg)
            row.append(self.capture_windows[key].bytes_total)
            row.append(self.capture_windows[key].avg_time_delta)
            row.append(self.capture_windows[key].source_port)
            row.append(self.capture_windows[key].destination_port)
            row.append(self.capture_windows[key].frames_total)
            rows.append(row)
        df = pd.DataFrame(rows, columns=['source', 'destination', 'protocol', 'bytes_avg', 'bytes_total',
            'avg_time_delta', 'source_port', 'destination_port', 'frames_total'])
        return df

    def add_predictions(self, predictions):
        for idx, key in enumerate(self.capture_windows.keys()):
            self.capture_windows[key].classification = predictions[idx]
            self.capture_windows_visual[key] = copy.copy(self.capture_windows[key])
    