import os
from subprocess import check_output

from images.camvid import CamvidDataset
from images.cifar10 import Cifar10Dataset
from images.cityscapes import CityscapesDataset
from images.isbi_em_stacks import IsbiEmStacksDataset
from images.kitti import KITTIdataset
from images.mscoco import MSCocoDataset
from images.pascalvoc import PascalVOCdataset
from images.polyps912 import Polyps912Dataset
from images.scene_parsing_MIT import SceneParsingMITDataset
from images.gta5 import GTA5Dataset
from images.gta5tiny import GTA5TinyDataset

from videos.change_detection import ChangeDetectionDataset
from videos.davis import DavisDataset
from videos.gatech import GatechDataset

cwd = os.path.join(__path__[0], os.path.pardir)
__version__ = check_output('git rev-parse HEAD', cwd=cwd,
                           shell=True).strip().decode('ascii')

__all__ = [
    "CamvidDataset",
    "Cifar10Dataset",
    "CityscapesDataset",
    "IsbiEmStacksDataset",
    "KITTIdataset",
    "MSCocoDataset",
    "PascalVOCdataset",
    "Polyps912Dataset",
    "SceneParsingMITDataset",
    "ChangeDetectionDataset",
    "DavisDataset",
    "GatechDataset",
    "GTA5Dataset",
    "GTA5TinyDataset"
    ]
