import os
import cv2
import time
import detectron2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from ultralytics import YOLO
from cityscapesscripts.helpers.labels import labels

# Ensure the results directory exists
os.makedirs('./static/result', exist_ok=True)
dataset_dir = '../cityscapes'
train_json = os.path.join(dataset_dir, "train.json")
val_json = os.path.join(dataset_dir, "val.json")
test_json = os.path.join(dataset_dir, "test.json")
images_dir = os.path.join(dataset_dir, "images")

# Register the datasets
register_coco_instances("cityscapes_train_3", {}, train_json, images_dir)
register_coco_instances("cityscapes_val_3", {}, val_json, images_dir)
register_coco_instances("cityscapes_test_3", {}, test_json, images_dir)

# Get class names from cityscapesscripts and sort by trainId
class_names = [label.name for label in sorted(labels, key=lambda x: x.trainId) if label.trainId not in [-1, 255]]

# Register metadata with class names
MetadataCatalog.get("cityscapes_train_3").thing_classes = class_names
MetadataCatalog.get("cityscapes_val_3").thing_classes = class_names
MetadataCatalog.get("cityscapes_test_3").thing_classes = class_names


def get_mask_rcnn_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cityscapes_train_3",)
    cfg.DATASETS.TEST = ("cityscapes_val_3",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # Pick a good LR
    cfg.SOLVER.MAX_ITER = 36000  # Set number of iterations for approximately 12 epochs
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)  # Number of classes in the Cityscapes dataset
    cfg.OUTPUT_DIR = "../output"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Path to the trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def get_yolo_model():
    model = YOLO('yolov8n-seg')  # Load the YOLOv8 segmentation model by name (e.g., 'yolov8n-seg', 'yolov8s-seg', etc.)
    return model


def process_image_with_yolo(image_path):
    model = get_yolo_model()

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")

    start_time = time.time()
    results = model(image)
    inference_time = time.time() - start_time

    # Save the results with the specified naming convention
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    result_file_name = f"{name}_result_yolo{ext}"
    result_path = os.path.join('static', 'result', result_file_name)

    # Ensure the result directory exists
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Save the image
    cv2.imwrite(result_path, results[0].plot())

    return result_file_name, inference_time  # Return the relative path and inference time


def process_video_with_yolo(video_path, output_path):
    model = get_yolo_model()
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_inference_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        results = model(frame)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        frame_count += 1

        annotated_frame = results[0].plot()  # Render the frame with annotations using the plot method
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0

    # Save the results with the specified naming convention
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    result_file_name = f"{name}_result_yolo{ext}"
    result_path = os.path.join('static', 'result', result_file_name)
    os.rename(output_path, result_path)

    return result_file_name, avg_inference_time  # Return the relative path and average inference time


def process_image_with_mask_rcnn(image_path):
    model, cfg = get_mask_rcnn_model()

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")

    start_time = time.time()
    results = model(image)
    inference_time = time.time() - start_time

    # Save the results with the specified naming convention
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    result_file_name = f"{name}_result_maskrcnn{ext}"
    result_path = os.path.join('static', 'result', result_file_name)

    # Ensure the result directory exists
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Visualize the predictions
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    out = v.draw_instance_predictions(results["instances"].to("cpu"))

    # Save the output image
    cv2.imwrite(result_path, out.get_image()[:, :, ::-1])

    return result_file_name, inference_time  # Return the relative path and inference tim


def process_video_with_mask_rcnn(video_path, output_path, batch_size=4):
    model, cfg = get_mask_rcnn_model()

    # Create temporary directory for frames
    frames_dir = 'frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    inference_times = []

    # Split video into frames and save them
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Process each frame with Mask R-CNN
    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.jpg')
        frame = cv2.imread(frame_path)

        start_time = time.time()
        outputs = model(frame)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        annotated_frame = v.get_image()[:, :, ::-1]
        cv2.imwrite(frame_path, annotated_frame)

    # Combine processed frames into a video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.jpg')
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

    # Clean up frames directory
    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.jpg')
        os.remove(frame_path)
    os.rmdir(frames_dir)

    avg_inference_time = np.mean(inference_times)

    return output_path, avg_inference_time
