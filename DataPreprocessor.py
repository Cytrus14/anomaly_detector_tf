import os
import pickle
import subprocess
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from Autoencoder import Autoencoder

class DataPreprocessor:
    def __init__(self):
        self.port_hasher = FeatureHasher(n_features=12, input_type="string")
        self.protocol_hasher = FeatureHasher(n_features=16, input_type="string")
        self.bytes_avg_scaler = None
        self.bytes_total_scaler = None
        self.avg_time_delta_scaler = None
        self.frames_total_scaler = None

    def _fill_nans(self, df):
        df['source'] = df['source'].fillna("0.0.0.0")
        df['destination'] = df['destination'].fillna("0.0.0.0")
        return df
    
    def _save_with_pickle(self, obj, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(obj, file)
    
    def _load_with_pickle(self, load_path):
        with open(load_path, 'rb') as file:
            return pickle.load(file)

    def _transform_ip_address(self, ip_addr):
        min_ip_octet = 0
        max_ip_octet = 255
        octets = ip_addr.split(',')[0] # if the are two ip address use only the first one
        #print(octets)
        octets = ip_addr.split(".")
        transformed_octets = []
        for idx, octet in enumerate(octets):
            octet = int(octet)
            normalized_octet = (octet - min_ip_octet) / (max_ip_octet - min_ip_octet)
            transformed_octets.append(normalized_octet)
        return transformed_octets
    
    def _transform_generic_numeric_column(self, df, column):
        data = np.array(df[column])
        data = data.reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        transformed_df = pd.DataFrame(data, columns=[column])
        return (transformed_df, scaler)
    
    def _transform_generic_numeric_column_with_scaler(self, df, column, scaler):
        data = np.array(df[column])
        data = data.reshape(-1, 1)
        data = scaler.transform(data)
        transformed_df = pd.DataFrame(data, columns=[column])
        return transformed_df
    
    def _transform_port_number(self, df, column):
        featureHasher = self.port_hasher
        data_sparse_matrix = featureHasher.fit_transform(df[column].apply(lambda x: [str(x)]))
        data_list = data_sparse_matrix.toarray()
        transformed_df = pd.DataFrame(data_list)
        return transformed_df
    
    def load_scalers(self):
        self.bytes_avg_scaler = self._load_with_pickle(os.path.join('model', 'scalers', 'bytes_avg_scaler.pkl'))
        self.bytes_total_scaler = self._load_with_pickle(os.path.join('model', 'scalers', 'bytes_total_scaler.pkl'))
        self.avg_time_delta_scaler = self._load_with_pickle(os.path.join('model', 'scalers', 'avg_time_delta_scaler.pkl'))
        self.frames_total_scaler = self._load_with_pickle(os.path.join('model', 'scalers', 'frames_total_scaler.pkl'))
    
    def preprocess_train_data(self, df_input):
        # fill nans
        df_input = self._fill_nans(df_input)
        # handle the source ip address
        source_transformed = pd.DataFrame()
        source_transformed['transformed_source'] = df_input['source'].apply(lambda ip: self._transform_ip_address(ip))
        source_transformed = source_transformed['transformed_source'].apply(pd.Series)
        source_transformed.columns = ["source_octet_1", "source_octet_2", "source_octet_3", "source_octet_4"]
        # handle the destination ip address
        destination_transformed = pd.DataFrame()
        destination_transformed['transformed_destination'] = df_input['destination'].apply(lambda ip: self._transform_ip_address(ip))
        destination_transformed = destination_transformed['transformed_destination'].apply(pd.Series)
        destination_transformed.columns = ["destination_octet_1", "destination_octet_2", "destination_octet_3", "destination_octet_4"]
        #combined = pd.concat([df_input, result], axis=1)
        # handle the protocol field
        featureHasher = self.protocol_hasher
        protocol_sparse_matrix = featureHasher.fit_transform(df_input['protocol'].apply(lambda x: [x]))
        protocol_list = protocol_sparse_matrix.toarray()
        protocol_transformed = pd.DataFrame(protocol_list)
        # handle port fields
        source_port_transformed = self._transform_port_number(df_input, 'source_port')
        destination_port_port_transformed = self._transform_port_number(df_input, 'destination_port')
        # handle generic numeric fields
        bytes_avg_transformed, bytes_avg_scaler = self._transform_generic_numeric_column(df_input, "bytes_avg")
        self._save_with_pickle(bytes_avg_scaler, os.path.join('model', 'scalers', 'bytes_avg_scaler.pkl'))
        bytes_total_transformed, bytes_total_scaler = self._transform_generic_numeric_column(df_input, "bytes_total")
        self._save_with_pickle(bytes_total_scaler, os.path.join('model', 'scalers', 'bytes_total_scaler.pkl'))
        avg_time_delta_transformed, avg_time_delta_scaler = self._transform_generic_numeric_column(df_input, "avg_time_delta")
        self._save_with_pickle(avg_time_delta_scaler, os.path.join('model', 'scalers', 'avg_time_delta_scaler.pkl'))
        frames_total_transformed, frames_total_scaler = self._transform_generic_numeric_column(df_input, "frames_total")
        self._save_with_pickle(frames_total_scaler, os.path.join('model', 'scalers', 'frames_total_scaler.pkl'))
        # Concat all the created dataframes
        df_combined = pd.concat([source_transformed, destination_transformed, protocol_transformed,
            source_port_transformed,destination_port_port_transformed, bytes_avg_transformed,
            bytes_total_transformed, avg_time_delta_transformed, frames_total_transformed], axis=1)
        return df_combined


    def preprocess_inference_data(self, df_input):
        df_input = self._fill_nans(df_input)
        # handle the source ip address
        source_transformed = pd.DataFrame()
        source_transformed['transformed_source'] = df_input['source'].apply(lambda ip: self._transform_ip_address(ip))
        source_transformed = source_transformed['transformed_source'].apply(pd.Series)
        source_transformed.columns = ["source_octet_1", "source_octet_2", "source_octet_3", "source_octet_4"]
        # handle the destination ip address
        destination_transformed = pd.DataFrame()
        destination_transformed['transformed_destination'] = df_input['destination'].apply(lambda ip: self._transform_ip_address(ip))
        destination_transformed = destination_transformed['transformed_destination'].apply(pd.Series)
        destination_transformed.columns = ["destination_octet_1", "destination_octet_2", "destination_octet_3", "destination_octet_4"]
        # handle the protocol field
        featureHasher = self.protocol_hasher
        protocol_sparse_matrix = featureHasher.fit_transform(df_input['protocol'].apply(lambda x: [x]))
        protocol_list = protocol_sparse_matrix.toarray()
        protocol_transformed = pd.DataFrame(protocol_list)
        # handle port fields
        source_port_transformed = self._transform_port_number(df_input, 'source_port')
        destination_port_port_transformed = self._transform_port_number(df_input, 'destination_port')
        # handle generic numeric fields
        bytes_avg_transformed = self._transform_generic_numeric_column_with_scaler(df_input, "bytes_avg", self.bytes_avg_scaler)
        bytes_total_transformed = self._transform_generic_numeric_column_with_scaler(df_input, "bytes_total", self.bytes_total_scaler)
        avg_time_delta_transformed = self._transform_generic_numeric_column_with_scaler(df_input, "avg_time_delta", self.avg_time_delta_scaler)
        frames_total_transformed = self._transform_generic_numeric_column_with_scaler(df_input, "frames_total", self.frames_total_scaler)
        # Concat all the created dataframes
        df_combined = pd.concat([source_transformed, destination_transformed, protocol_transformed,
            source_port_transformed,destination_port_port_transformed, bytes_avg_transformed,
            bytes_total_transformed, avg_time_delta_transformed, frames_total_transformed], axis=1)
        return df_combined