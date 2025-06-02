import openvino as ov
import os

core = ov.Core()

tflite_model_name = "selfie_multiclass_256x256"
tflite_model_path = tflite_model_name + ".tflite"

ir_model_path = tflite_model_name + ".xml"

if os.path.exists(ir_model_path) == False:
    if os.path.exists(tflite_model_path) == False:
        raise FileNotFoundError(f"Model file {tflite_model_path} not found")
    ov_model = ov.convert_model(tflite_model_path)
    ov.save_model(ov_model, ir_model_path)
    print("Converting Model to OpenVINO IR")
else:
    ov_model = core.read_model(ir_model_path)
    print("Model already exists")

print(f"Model input info: {ov_model.inputs}")

print(f"Model output info: {ov_model.outputs}")
