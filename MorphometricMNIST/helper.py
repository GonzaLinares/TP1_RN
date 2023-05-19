
from tensorflow.keras.callbacks import CSVLogger

def callbackH(location):
    csv_logger = CSVLogger(location, separator=',', append=False)
    return csv_logger