from ultralytics import YOLO, settings
from pathlib import Path
import argparse

class Trainer:
    def __init__(self):
        self.project_folder = Path(__file__).resolve().parent
        self.model_config = Path(self.project_folder, "yolo11m.yaml")
        self.data_config = Path(self.project_folder, "pallets.yaml")
        self.setup()

    def setup(self):
        settings.update({"datasets_dir": str(self.project_folder)})
        dataset_folder = Path(self.project_folder, "dataset")

    def train(self, pretrained_model_path=None):
        model = YOLO(self.model_config)
        if pretrained_model_path:
            model.load(pretrained_model_path)
        model.train(data = self.data_config,
                    project = "pallets",
                    imgsz = 640,
                    device = [0],
                    batch = 4,
                    epochs = 100,
                    amp = True,
                    save = True,
                    workers = 8,
                    verbose = True,
                    val = True,
                    optimizer = 'auto',
                    # optimizer = 'AdamW',
                    # lr0 = 2e-5,
                    # lrf = 0.01,
                    # momentum = 0.9,
                    # weight_decay = 0.001,
                    hsv_h = 0.02,
                    hsv_s = 0.8,
                    hsv_v = 0.5,
                    translate = 0.0,
                    scale = 0.2,
                    shear = 1.0,
                    perspective = 0.0001,
                    fliplr = 0.5,
                    )
        
    def validate(self, model_path):
        model = YOLO(model_path)
        model.val(data=self.data_config,
                  project="pallets",
                  imgsz=640,
                  device=[0],
                  batch=4,
                  verbose=True)

def run_train(model_path=None):
    """
    Runs the training logic.
    Parameters:
        model_path (str): Optional path to a pre-trained model. If provided, the training will start from this model.
    """
    trainer = Trainer()
    trainer.setup()
    trainer.train(pretrained_model_path=model_path)

def run_val(model_path):
    """
    Runs validation logic.
    """
    trainer = Trainer()
    trainer.setup()
    trainer.validate(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trainig YOLO model", add_help=True)
    parser.add_argument("--model_path", "-p", type=str, default=None, help="path to a pretrained model")
    args = parser.parse_args()

    model_path = args.model_path
    run_train(model_path)
    # run_val(model_path)
