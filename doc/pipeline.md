
Steps to run inference and object position estimation. 


1. Calibrate the intrisincs. 

2. Calibrate the extrinsics. 

3. Train the detection model. 


4. Export to ONNX
```bash
python export.py --weights ../experiments/runs/yolo_100/potato_rgbdif/weights/best.pt  --include onnx --batch-size 1 --device cpu --simplify --opset 11
```
Notes:
--simplify flag gave some versions error when using onnx


5. Run inference in the boat (CPU)


6. Optimize and convert for tensorrt engine execution
```bash
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine
```

7. Running on the vessel



