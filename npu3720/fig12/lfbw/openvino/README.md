
# Model
```
cd openvino
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

# Convert
```
python convert_model.py
```

# Run
```bash
# in the directory ./Linux-Fake-Background-Webcam

# use GPU
python3 lfbw/lfbw.py --ov-device GPU

# use NPU
python3 lfbw/lfbw.py --ov-device NPU
```



# Play Fake Selfie Video
```bash
ffmpeg -loop 1 -re -i selfies.jpg -f v4l2 -vcodec rawvideo -pix_fmt yuv420p /dev/video0
```


# Show Result
```bash
ffplay /dev/video2
```