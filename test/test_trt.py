import cv2
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# print(f"tensorrt {trt.__version__}")
# print(f"tensorrt {trt.__file__}")
# print(f"pycuda {pycuda.__file__}")

# TensorRT Logger
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

batch = 1
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []



def load_engine(engine_path):
    """Load a pre-built TensorRT engine from disk."""
    print(f"Loading TensorRT engine from {engine_path}...")
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt.Logger(trt.Logger.VERBOSE))
    engine = runtime.deserialize_cuda_engine(engine_data)
    if not engine:
        raise ValueError(f"Failed to load the engine from {engine_path}")
    print("TensorRT engine loaded successfully.\n")

    print("create buffers")
    for binding in engine:
        print(binding)
        print(f"Binding: {binding}")

        size = trt.volume(engine.get_tensor_shape(binding)) * batch
        print(f"Size: {size}")
        host_mem = cuda.pagelocked_empty(shape=[size],dtype=np.float32)
        print(f"Host mem: {host_mem}")
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        print(f"Cuda mem: {cuda_mem}")

        bindings.append(int(cuda_mem))
        if engine.get_tensor_mode(binding)==trt.TensorIOMode.INPUT:
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    print("Buffers created")
    
    stream = cuda.Stream()
    context = engine.create_execution_context()
    return engine, stream, context


def preprocess_image(image_path, target_size=(640, 640)):
    """Read, resize, and preprocess the image."""
    print("Reading and preprocessing image...")
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)

    # Normalize and convert BGR to RGB
    img_resized = img_resized[..., ::-1] / 255.0  # BGR to RGB and normalize to [0, 1]
    
    # Convert to CHW format and add batch dimension
    img_input = np.transpose(img_resized, (2, 0, 1))[None, ...].astype(np.float32)

    return img_input

def run_inference(engine, stream, context, img_input):
    """Run inference on the input image."""
    print("Running inference...")


    np.copyto(host_inputs[0], img_input.ravel())

    # Execute the inference

    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    # stream.synchronize()
    context.execute_v2(bindings)
    # stream.synchronize()
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    return host_outputs[0]

def postprocess_results(output, confidence_threshold=0.2, image_size=(1024, 1224)):
    """Postprocess the raw model output."""
    print("Postprocessing results...")
    # Filter out low-confidence boxes
    print(output.shape)
    print(output[:10])

    output = np.array(output).reshape(-1,6)
    original_boxes = output[output[:, 4] > confidence_threshold]
    
    # Rescale boxes to original image size
    original_boxes[:, 0] *= image_size[1] / 640  # x1
    original_boxes[:, 1] *= image_size[0] / 640  # y1
    original_boxes[:, 2] *= image_size[1] / 640  # x2
    original_boxes[:, 3] *= image_size[0] / 640  # y2

    # Convert box coordinates from xywh to xyxy
    converted_boxes = convert_xywh_to_xyxy(original_boxes)

    # Apply Non-Maximum Suppression (NMS)
    cv2_boxes = converted_boxes[:, :4]
    cv2_scores = converted_boxes[:, 4]
    indices = cv2.dnn.NMSBoxes(cv2_boxes.tolist(), cv2_scores.tolist(), confidence_threshold, 0.4)

    # Filter selected boxes
    XcYcWH_boxes = original_boxes[indices]
    XYXY_boxes = converted_boxes[indices]
    return XcYcWH_boxes, XYXY_boxes

def convert_xywh_to_xyxy(boxes):
    """Convert bounding boxes from xywh to xyxy format."""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.column_stack([x1, y1, x2, y2, boxes[:, 4]])


def draw_boxes_on_image(img, XcYcWH_boxes, XYXY_boxes, color=(0, 255, 0), marker_color=(0, 0, 255)):
    """Draw bounding boxes and centers on the image."""
    # print("Drawing bounding boxes on image...")
    print(f"\n\nnumber of detections:{XcYcWH_boxes}\n\n")
    for box in XYXY_boxes:
        # Draw bounding box
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        
    for box in XcYcWH_boxes:
        # Draw center
        xc, yc = box[:2]
        cv2.drawMarker(img, (int(xc), int(yc)), marker_color, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
        
    return img

if __name__ == "__main__":
    engine_path = '../config/best.engine'  # Path to the pre-built TensorRT engine
    image_path = 'test_input.png'

    # Load the pre-built TensorRT engine
    engine, stream, context = load_engine(engine_path)

    # Preprocess the image
    img_input = preprocess_image(image_path)
    # print(f"Image input shape: {img_input.shape}")
    # print(f"Image type: {type(img_input)}")
    # print(f"Type of image input: {img_input.dtype}")
    # print(f"Image input: {img_input[:10]}")

    # Run inference with TensorRT engine
    output = run_inference(engine, stream, context, img_input)
    print(f"Inference output shape: {output.shape}")
    print(f"Inference output type: {type(output)}")
    print(f"Inference output: {output[:10]}")

    # print(f"Inference output shape: {output.shape}")
    # print(f"Inference output type: {type(output)}")
    # print(f"Inference output: {output[:10]}")

    # Postprocess the output
    XcYcWH_boxes, XYXY_boxes = postprocess_results(output)

    # Load original image for drawing boxes
    img = cv2.imread(image_path)

    # Draw the boxes on the image
    img = draw_boxes_on_image(img, XcYcWH_boxes, XYXY_boxes)

    # Save the result image
    cv2.imwrite('result_trt.png', img)
    print("Result image saved as 'result_trt.png'.")


