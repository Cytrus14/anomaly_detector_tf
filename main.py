import argparse
import os
import subprocess
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.style import Style
from Autoencoder import Autoencoder
from DataPreprocessor import DataPreprocessor
from TrafficData import TrafficData

cycle_timer = 0

def keras_predict(model, data, threshold=1.0):
    pred = model(data)
    loss = tf.keras.losses.mae(pred, data)
    return tf.math.less(loss, threshold)

def capture_baseline_traffic(traffic_data, console, cycle_timer=0):
    traffic_data.buffer_traffic_data()
    cycle_timer += 1
    # write buffer content to a csv file every 20 cycles
    if cycle_timer >= 20 and traffic_data.get_buffer_size() > 0:
        traffic_data.write_buffer_to_csv()
        cycle_timer = 0
        console.log(f"{traffic_data.get_captured_data_points_count()} data points captured")
    threading.Timer(0.1, capture_baseline_traffic, (traffic_data, console, cycle_timer)).start()

def analyze_traffic(traffic_data, console, cycle_timer=0, dataPreprocessor=None, model=None, threshold=1.0):
    traffic_data.buffer_traffic_data()
    cycle_timer += 1
    if cycle_timer == 20:
        captured_data = traffic_data.get_traffic_data_buffer()
        traffic_data.clear_traffic_data_buffer()
        # Perform inference
        if len(captured_data) > 0:
            captured_data.reset_index(drop=True, inplace=True)
            captured_data_preprocessed = dataPreprocessor.preprocess_inference_data(captured_data)
            predictions = keras_predict(model, captured_data_preprocessed, threshold)
            predictions = predictions.numpy()
            captured_data['class'] = predictions
            with console.capture() as capture:
                draw_table(captured_data, console)
            console.clear()
            print(capture.get())
        cycle_timer = 0
    threading.Timer(0.1, analyze_traffic, [traffic_data, console, cycle_timer, dataPreprocessor, model, threshold]).start()

def draw_table(captured_data, console):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="blue")
    table.add_column("Class")
    table.add_column("Source", justify="right")
    table.add_column("Destination", justify="right")
    table.add_column("Protocol", justify="right")
    table.add_column("Bytes Total", justify="right")
    table.add_column("Avg Time Delta (ms)", justify="right")
    table.add_column("Frames total", justify="right")

    style_normal = Style(bgcolor="green", bold=True)
    style_anomalous = Style(bgcolor="red", bold=True)
    for idx, row in captured_data.iterrows():
        if row['class']:
            class_row = Text("Normal")
            class_row.stylize(style_normal)
        else:
            class_row = Text("Anomaly")
            class_row.stylize(style_anomalous)
        table.add_row(
            str(idx),
            class_row,
            row['source'],
            row['destination'],
            row['protocol'],
            str(row['bytes_total']),
            str(round(row['avg_time_delta'] * 1000, 4)),
            str(row['frames_total'])
        )
    console.print(table)

def tshark_capture(interface, console, mode, dataPreprocessor=None, model=None, threshold=1.0):
    #connections = Connections()
    #console = Console()
    cmd = [
        "tshark", "-l", "-i", interface, "-T", "fields",
        "-e", "ip.src",             # frame source ip
        "-e", "ip.dst",             # frame destination ip
        "-e", "frame.len",          # frame length
        "-e", "frame.time_delta",   # frame capture time
        "-e", "frame.protocols",    # frame protocol
        "-e", "tcp.srcport",        # tcp source port
        "-e", "tcp.dstport",        # tcp destination port
        "-e", "udp.srcport",        # udp source port
        "-e", "udp.dstport"         # udp destination port
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,

    )

    traffic_data = TrafficData()
    if mode == 'baseline-capture':
        capture_baseline_thread = threading.Thread(target=capture_baseline_traffic, args=(traffic_data, console, 0))
        capture_baseline_thread.start()
    elif mode == 'analysis':
        analyze_traffic_thread = threading.Thread(target=analyze_traffic, args=(traffic_data, console, 0, dataPreprocessor, model, threshold))
        analyze_traffic_thread.start()

    for tshark_output_line in process.stdout:
        traffic_data.update_intermediate_traffic_data(tshark_output_line)

    return_code = process.wait()
    print(return_code)

console = Console()
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--interface", type=str, help="Network interface for traffic analysis/capture")
parser.add_argument("-c", "--capture-traffic", action="store_true", help="Capture baseline traffic")
parser.add_argument("-f", "--fit-model", action="store_true", help="Fit the model using captured baseline traffic")
parser.add_argument("-t", "--threshold", type=float, help="Set threshold for traffic analysis")
parser.add_argument("-e", "--epochs", type=int, help="Epochs for fitting the model")
args = parser.parse_args()

interface = args.interface
# Traffic capture mode
if args.capture_traffic and not args.fit_model:
    if interface == None:
        print("Error: No network interface specified. Select a valid interface with -i or --interface")
        exit()
    with console.status("Capturing traffic...") as status:
        tshark_capture(interface, console, 'baseline-capture')

# Train model mode
elif not args.capture_traffic and args.fit_model:
    preprocessor = DataPreprocessor()
    df = pd.read_csv("baseline_traffic.csv")
    df = preprocessor.preprocess_train_data(df)
    model = Autoencoder()
    model.compile(optimizer="adam", loss="mae")
    epochs = 50
    if args.epochs != None:
        epochs = args.epochs
    history = model.fit(df, df, epochs=epochs, batch_size=24)
    print(model.summary())
    model.save(os.path.join('model', "model.keras"))
    print("Model trained and saved!")

# Anomaly detection mode
elif not args.capture_traffic and not args.fit_model:
    if interface == None:
        print("Error: No network interface specified. Select a valid interface with -i or --interface")
        exit()
    threshold = 1.0
    if args.threshold != None:
        threshold = args.threshold
    dataPreprocessor = DataPreprocessor()
    dataPreprocessor.load_scalers()
    model = tf.keras.models.load_model(os.path.join('model', 'model.keras'))
    tshark_capture(interface, console, 'analysis', dataPreprocessor=dataPreprocessor, model=model, threshold=threshold)
else:
    print("Error: Flags -c (--capture-traffic) and -f (--fit-model) cannot be used simouteniously.")
