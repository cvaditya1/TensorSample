import tensorflow as tf
import librosa
import os
import numpy as np

TFLite_model = 'lite-model_ASR_TFLite_pre_trained_models_English_1.tflite'

# audio = 'speech_whistling2.wav'
audio = 'This is automation.wav'
signal, _ = librosa.load(os.path.expanduser(audio), sr=16000, mono=False)

interpreter = tf.lite.Interpreter(model_path=TFLite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(input_details[0]["index"], signal.shape)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]["index"], signal)
interpreter.set_tensor(
    input_details[1]["index"],
    np.array(0).astype('int32')
)
interpreter.set_tensor(
    input_details[2]["index"],
    np.zeros([1,2,1,320]).astype('float32')
)
interpreter.invoke()
hyp = interpreter.get_tensor(output_details[0]["index"])

final_text = "".join([chr(u) for u in hyp if u != 0])
print(final_text)