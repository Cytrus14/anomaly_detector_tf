import argparse
import copy
import csv
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
from Connections import Connections
from DataPreprocessor import DataPreprocessor

def keras_predict(model, data, threshold=0.12):
    pred = model(data)
    loss = tf.keras.losses.mae(pred, data)
    return tf.math.less(loss, threshold)

cycle_timer = 0
def write_buffer_to_csv(buffer, console):
    if not hasattr(write_buffer_to_csv, "header_written"):
        write_buffer_to_csv.header_written = False
    if not hasattr(write_buffer_to_csv, "output_file_exists_check"):
        write_buffer_to_csv.output_file_exists_check = False
    if not hasattr(write_buffer_to_csv, "id_counter"):
        write_buffer_to_csv.id_counter = 0

    if os.path.exists("baseline_traffic.csv") and not write_buffer_to_csv.output_file_exists_check:
        os.remove("baseline_traffic.csv")
        write_buffer_to_csv.output_file_exists_check = True


    fieldnames = ["id", "source", "destination", "protocol", "bytes_avg", "bytes_total",
        "avg_time_delta", "source_port", "destination_port", "frames_total"]
    with open("baseline_traffic.csv", "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not write_buffer_to_csv.header_written:
            writer.writeheader()
            write_buffer_to_csv.header_written = True
        for window_captures in buffer:
            for key in window_captures.keys():
                dict_temp = {field: getattr(window_captures[key], field) for field in fieldnames[1:]}
                dict_temp["id"] = write_buffer_to_csv.id_counter
                write_buffer_to_csv.id_counter += 1
                writer.writerow(dict_temp)

    console.log(f"{write_buffer_to_csv.id_counter} data points captured")

def capture_baseline_traffic(connections, buffer, console, cycle_timer=0):
    connections.finalize_capture_windows()
    capture_windows = connections.get_capture_windows()
    if len(capture_windows.keys()) > 0:
        buffer.append(capture_windows)
    connections.clear_capture_windows()
    cycle_timer += 1
    # write buffer content to a csv file every 20 cycles
    if cycle_timer >= 20 and len(buffer) > 0:
        write_buffer_to_csv(buffer, console)
        buffer = []
        cycle_timer = 0
    threading.Timer(0.1, capture_baseline_traffic, (connections, buffer, console, cycle_timer)).start()


def analyze_connections(connections, console, cycle_timer=0, dataPreprocessor=None, model=None):
    #connections = copy.deepcopy(connections)
    connections.finalize_capture_windows()
    capture_windows = connections.get_capture_windows()
    df = connections.convert_to_pd_dataframe()
    if len(df.index) > 0:
        df = dataPreprocessor.preprocess_inference_data(df)
        predictions = keras_predict(model, df)
        predictions = predictions.numpy()
        connections.add_predictions(predictions)

    cycle_timer += 1
    if cycle_timer == 20:
        capture_windows = connections.get_capture_window_visual()
        with console.capture() as capture:
            draw_table(console, connections, capture_windows)
        console.clear()
        print(capture.get())
        connections.clear_capture_windows_visual()
        cycle_timer = 0
    connections.clear_capture_windows()
    threading.Timer(0.1, analyze_connections, [connections, console, cycle_timer, dataPreprocessor, model]).start()


def draw_table(console, connections, capture_windows_visual):
    table = Table(show_header=True, header_style="bold magenta")
    #table.add_column("Date", style="dim", width=12)
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
    #capture_windows_visual = connections.get_capture_window_visual()
    for idx, key in enumerate(capture_windows_visual.keys()):
        if capture_windows_visual[key].classification:
            class_row = Text("Normal")
            class_row.stylize(style_normal)
        else:
            class_row = Text("Anomaly")
            class_row.stylize(style_anomalous)
        table.add_row(
            str(idx),
            class_row,
            capture_windows_visual[key].source,
            capture_windows_visual[key].destination,
            capture_windows_visual[key].protocol,
            str(capture_windows_visual[key].bytes_total),
            str(round(capture_windows_visual[key].avg_time_delta * 1000, 4)),
            str(capture_windows_visual[key].frames_total)
        )
    console.print(table)


def tshark_capture(interface, console, mode, dataPreprocessor=None, model=None):
    connections = Connections()
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

    if mode == 'baseline-capture':
        buffer = []
        capture_baseline_thread = threading.Thread(target=capture_baseline_traffic, args=(connections, buffer, console, 0))
        capture_baseline_thread.start()
    elif mode == 'analysis':
        analyze_connections_thread = threading.Thread(target=analyze_connections, args=(connections, console, 0, dataPreprocessor, model))
        analyze_connections_thread.start()

    for line in process.stdout:
        connections.update_capture_windows(line)

    return_code = process.wait()
    print(return_code)


console = Console()
parser = argparse.ArgumentParser()
parser.add_argument("interface", help="Network interface for traffic analysis/capture")
parser.add_argument("-c", "--capture-traffic", action="store_true", help="Capture baseline traffic")
parser.add_argument("-t", "--train-model", action="store_true", help="Train the model using captured baseline traffic")
args = parser.parse_args()

interface = args.interface
# Traffic capture mode
if args.capture_traffic and not args.train_model:
    with console.status("Capturing traffic...") as status:
        tshark_capture(interface, console, 'baseline-capture')

# Train model mode
elif not args.capture_traffic and args.train_model:
    preprocessor = DataPreprocessor()
    df = pd.read_csv("baseline_traffic.csv")
    df = preprocessor.preprocess_train_data(df)
    model = Autoencoder()
    model.compile(optimizer="adam", loss="mae")
    history = model.fit(df, df, epochs=50, batch_size=24)
    model.save(os.path.join('model', "model.keras"))
    print("Model trained and saved!")

# Anomaly detection mode
elif not args.capture_traffic and not args.train_model:
    dataPreprocessor = DataPreprocessor()
    dataPreprocessor.load_scalers()
    model = tf.keras.models.load_model(os.path.join('model', 'model.keras'))
    tshark_capture(interface, console, 'analysis', dataPreprocessor=dataPreprocessor, model=model)
else:
    print("Error: Flags -c (--capture-traffic) and -t (--train_model) cannot be used simouteniously.")

# console = Console()
# draw_table(console)
#console.clear()