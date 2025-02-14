import onnx
import onnxruntime as ort
import cv2
import numpy as np
import time

def load_and_check_model(onnx_model_path):
    """Load and check the ONNX model."""
    # print("Loading model...")
    model = onnx.load(onnx_model_path)
    # onnx.checker.check_model(model)
    # print("Model is valid.")
    # return model

def prepare_session(onnx_model_path):
    """Prepare the ONNX Runtime session."""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Adjust number of threads for CPU
    sess_options.inter_op_num_threads = 1

    # Load the model
    # print("Creating inference session...")
    session = ort.InferenceSession(onnx_model_path, sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    return session, input_name, output_name

def preprocess_image(image_path, target_size=(640, 640)):
    """Load, resize, and preprocess the image for inference."""
    # print("Reading and preprocessing image...")
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)

    # Normalize and convert BGR to RGB
    img_resized = img_resized[..., ::-1] / 255.0  # BGR to RGB and normalize to [0, 1]
    
    # Convert to CHW format and add batch dimension
    img_input = np.transpose(img_resized, (2, 0, 1))[None, ...].astype(np.float32)
    return img_input

def run_inference(session, input_name, output_name, img_input):
    """Run inference on the image and measure time."""
    outputs = session.run([output_name], {input_name: img_input})
    return outputs[0]

def postprocess_results(output, confidence_threshold=0.7, image_size=(1024, 1224)):
    """Postprocess the raw model output."""
    # print("Postprocessing results...")
    # Filter out low-confidence boxes
    original_boxes = output[output[:, :, 4] > confidence_threshold]
    
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
    onnx_model_path = '../models/yolov5/DiffObjDet.onnx'
    image_path = 'test_input.png'

    # Load and check model
    start_time = time.time()
    load_and_check_model(onnx_model_path)
    end_time = time.time() - start_time
    print(f"Load model time: {end_time:.4f} seconds")


    # Prepare inference session
    start_time = time.time()
    session, input_name, output_name = prepare_session(onnx_model_path)
    end_time = time.time() - start_time
    print(f"Prepare session time: {end_time:.4f} seconds")


    # Preprocess image
    start_time = time.time()
    img_input = preprocess_image(image_path)
    end_time = time.time() - start_time
    print(f"Preprocess image time: {end_time:.4f} seconds")

    # Run inference
    start_time = time.time()
    for i in range(10):
        output = run_inference(session, input_name, output_name, img_input)
    end_time = time.time() - start_time
    print(f"Run inference time: {end_time:.4f} seconds")

    # Postprocess the output
    start_time = time.time()
    XcYcWH_boxes, XYXY_boxes = postprocess_results(output)
    end_time = time.time() - start_time
    print(f"Postprocess results time: {end_time:.4f} seconds")

    # Load original image for drawing boxes
    img = cv2.imread(image_path)

    # Draw the boxes on the image
    img = draw_boxes_on_image(img, XcYcWH_boxes, XYXY_boxes)

    # Save the result image
    cv2.imwrite('result_onnx.png', img)
    print("Result image saved as 'result_onnx.png'.")
