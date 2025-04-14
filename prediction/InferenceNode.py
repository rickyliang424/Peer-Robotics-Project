import os
import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO, SAM, FastSAM

import torch
from copy import deepcopy
from ultralytics.data.augment import LetterBox
from ultralytics.utils.plotting import Annotator, colors

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.subscription = self.create_subscription(Image, '/robot1/zed2i/left/image_rect_color', self.image_callback, 10)
        self.publisher_det = self.create_publisher(Image, '/pallet_detection', 10)
        self.publisher_seg = self.create_publisher(Image, '/ground_segmentation', 10)
        self.bridge = CvBridge()

        self.model_det = YOLO("./models/best4.engine", task='detect')
        self.model_seg = FastSAM("./models/FastSAM-s.engine")
        self.get_logger().info("InferenceNode initialized.")
    
    def annotation(self, img, type):
        annotator = Annotator(deepcopy(img), None, None, "Arial.ttf", False, example='')
        if type == 'det':
            for i, d in enumerate(reversed(self.res.boxes)):
                d_conf, id = float(d.conf), None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + self.res.names[1]
                label = (f"{name} {d_conf:.2f}")
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(0, True), rotated=None)
        else:
            img = LetterBox(self.res.masks.shape[1:])(image=annotator.result())
            im_gpu = (torch.as_tensor(img, dtype=torch.float16, device=self.res.masks.data.device).permute(2,0,1).flip(0).contiguous()/255)
            annotator.masks(self.res.masks.data, colors=[colors(1, True)], im_gpu=im_gpu)
        return annotator.result()
    
    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite("./temp.jpg", cv_img)
        img = cv2.resize(cv2.imread("./temp.jpg"), (640, 640))
        
        # Detection
        results_det = self.model_det(img)
        self.res = results_det[0]
        xyxy = self.res.boxes.xyxy.cpu().numpy()
        cls = self.res.boxes.cls.cpu().numpy()
        box = xyxy[cls == 0, :]
        if box.shape[0] > 1:
            conf = self.res.boxes.conf.cpu().numpy()
            box = box[np.argmax(conf)]
                
        # Segmentation with detection box
        results_seg = self.model_seg(img, imgsz=640, bboxes=box)
        self.res.boxes = self.res.boxes[cls == 1.0]
        self.res.masks = results_seg[0].masks
        
        output_img_det = self.annotation(img, 'det')
        output_img_seg = self.annotation(img, 'seg')
        
        # Convert back to ROS Image and publish
        output_msg_det = self.bridge.cv2_to_imgmsg(output_img_det, encoding='bgr8')
        output_msg_seg = self.bridge.cv2_to_imgmsg(output_img_seg, encoding='bgr8')
        self.publisher_det.publish(output_msg_det)
        self.publisher_seg.publish(output_msg_seg)

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        os.remove("./temp.jpg")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
