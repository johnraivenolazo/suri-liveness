import os
import torch

DATA_PATH = os.environ.get("DATA_PATH", "./data")


def get_num_classes(spoof_categories):
    assert isinstance(
        spoof_categories, list
    ), "spoof_categories expected to be list of spoof labels lists, got {}".format(
        spoof_categories
    )
    num_classes = len(spoof_categories)
    return num_classes


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


class TrainConfig(object):
    def __init__(
        self,
        input_size=128,
        batch_size=256,
        spoof_categories=[[0], [1, 2, 3, 7, 8, 9]],
        class_balancing=None,
        crop_dir="data",
        output_dir="./output",
    ):
        self.lr = 1e-1
        self.milestones = [10, 15, 22, 30]
        self.gamma = 0.1
        self.epochs = 50
        self.momentum = 0.9
        self.batch_size = batch_size
        self.valid_size = 0.2
        self.class_balancing = class_balancing
        self.output_dir = output_dir

        self.input_size = input_size
        self.train_path = "{}/{}/train".format(DATA_PATH, crop_dir)
        self.labels_path = "{}/{}/metas/labels/train_label.json".format(
            DATA_PATH, crop_dir
        )
        self.spoof_categories = spoof_categories

        self.num_classes = get_num_classes(spoof_categories)
        self.num_channels = 3
        self.embedding_size = 128
        self.kernel_size = get_kernel(input_size, input_size)
        self.fourier_size = [2 * s for s in self.kernel_size]

        self.log_loss_per_epoch = 10

    def set_job(self, name, device_id=0):
        self.device = (
            "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        )
        self.job_name = "AntiSpoofing_{}_{}".format(name, self.input_size)

        self.model_path = os.path.join(self.output_dir, name)
        self.log_path = os.path.join(self.model_path, "logs")

        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        print(f"Output directory: {self.model_path}")


class PretrainedConfig(object):
    def __init__(self, model_path, device_id=0, input_size=128, num_classes=2):
        self.model_path = model_path
        self.device = (
            "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        )
        self.input_size = input_size
        self.kernel_size = get_kernel(input_size, input_size)
        self.num_classes = num_classes


class TestConfig(PretrainedConfig):
    def __init__(
        self,
        model_path,
        device_id=0,
        input_size=128,
        batch_size=1,
        spoof_categories=[[0], [1, 2, 3, 7, 8, 9]],
        crop_dir="data",
    ):
        super().__init__(
            model_path, device_id, input_size, get_num_classes(spoof_categories)
        )
        self.test_path = "{}/{}/test".format(DATA_PATH, crop_dir)
        self.labels_path = "{}/{}/metas/labels/test_label.json".format(
            DATA_PATH, crop_dir
        )
        self.spoof_categories = spoof_categories
        self.batch_size = batch_size
