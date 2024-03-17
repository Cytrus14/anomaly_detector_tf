class CaptureWindow:
    def __init__(self):
        self.source = ""
        self.destination = ""
        self.protocol = ""
        self.bytes_avg = 0
        self.bytes_total = 0
        self.avg_time_delta = 0
        self.source_port = 0
        self.destination_port = 0
        self.frames_total = 0
        self.classification = None
        self._time_delta_sum = 0
