import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")


# Expected Output
# TensorFlow has access to the following devices:
# [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# TensorFlow version: 2.8.0