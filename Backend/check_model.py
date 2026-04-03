import tensorflow as tf
import numpy as np

interp = tf.lite.Interpreter(model_path='isl_twohand_mlp.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
print('Input shape:', inp[0]['shape'])
print('Input dtype:', inp[0]['dtype'])
print('Output shape:', out[0]['shape'])
print('Num ops:', len(interp._get_ops_details()))