import numpy as np
import os
import time

from dataset_loaders.parallel_loader import ThreadedDataset
from getpass import getuser

floatX = 'float32'

path = ('Tmp/' + getuser() + '/gta5')

class GTA5Dataset(ThreadedDataset):
    '''The GTA5 semantic segmentation dataset

    The GTA5 dataset [1] consists of 24966 densely labelled frames split into
    10 parts for convenience. The class labels are compatible with the CamVid
    and CityScapes datasets.

    The dataset should be downloaded from [1]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.

     References
    ----------
    .. [1] https://download.visinf.tu-darmstadt.de/data/from_games/
    '''
    name = 'gta5'
    # optional arguments
    data_shape = (1052, 1914, 3)
    mean = [0, 0, 0]
    std = [1, 1, 1]
    max_files = 512

    mapping_type = 'cityscapes'

    if mapping_type == 'camvid':
        non_void_nclasses = 11
        _void_labels = [11]

        _cmap = {
            0: (128, 128, 128),    # sky
            1: (128, 0, 0),        # building
            2: (192, 192, 128),    # column_pole
            3: (128, 64, 128),     # road
            4: (0, 0, 192),        # sidewalk
            5: (128, 128, 0),      # Tree
            6: (192, 128, 128),    # SignSymbol
            7: (64, 64, 128),      # Fence
            8: (64, 0, 128),       # Car
            9: (64, 64, 0),        # Pedestrian
            10: (0, 128, 192),     # Bicyclist
            11: (0, 0, 0)}         # Void
        _mask_labels = {0: 'sky', 1: 'building', 2: 'column_pole',
                        3: 'road', 4: 'sidewalk', 5: 'tree', 6: 'sign',
                        7: 'fence', 8: 'car', 9: 'pedestrian',
                        10: 'byciclist', 11: 'void'}
    else:
        non_void_nclasses = 19
        _void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        GTclasses = range(34)
        GTclasses = GTclasses + [-1]

        _mask_labels = {
            0: 'unlabeled',
            1: 'ego vehicle',
            2: 'rectification border',
            3: 'out of roi',
            4: 'static',
            5: 'dynamic',
            6: 'ground',
            7: 'road',
            8: 'sidewalk',
            9: 'parking',
            10: 'rail track',
            11: 'building',
            12: 'wall',
            13: 'fence',
            14: 'guard rail',
            15: 'bridge',
            16: 'tunnel',
            17: 'pole',
            18: 'polegroup',
            19: 'traffic light',
            20: 'traffic sign',
            21: 'vegetation',
            22: 'terrain',
            23: 'sky',
            24: 'person',
            25: 'rider',
            26: 'car',
            27: 'truck',
            28: 'bus',
            29: 'caravan',
            30: 'trailer',
            31: 'train',
            32: 'motorcycle',
            33: 'bicycle',
            -1: 'license plate'
        }
        _cmap = {
            0: (0, 0, 0),           # unlabeled
            1: (0, 0, 0),           # ego vehicle
            2: (0, 0, 0),           # rectification border
            3: (0, 0, 0),           # out of roi
            4: (0, 0, 0),           # static
            5: (0, 0, 0),           # dynamic
            6: (0, 0, 0),           # ground
            7: (128, 64, 128),      # road
            8: (244, 35, 232),      # sidewalk
            9: (0, 0, 0),           # parking
            10: (0, 0, 0),          # rail track
            11: (70, 70, 70),       # building
            12: (102, 102, 156),    # wall
            13: (190, 153, 153),    # fence
            14: (0, 0, 0),          # guard rail
            15: (0, 0, 0),          # bridge
            16: (0, 0, 0),          # tunnel
            17: (153, 153, 153),    # pole
            18: (0, 0, 0),          # polegroup
            19: (250, 170, 30),     # traffic light
            20: (220, 220,  0),     # traffic sign
            21: (107, 142, 35),     # vegetation
            22: (152, 251, 152),    # terrain
            23: (0, 130, 180),      # sky
            24: (220, 20, 60),      # person
            25: (255, 0, 0),        # rider
            26: (0, 0, 142),        # car
            27: (0, 0, 70),         # truck
            28: (0, 60, 100),       # bus
            29: (0,  0, 0),         # caravan
            30: (0,  0, 0),         # trailer
            31: (0, 80, 100),       # train
            32: (0, 0, 230),        # motorcycle
            33: (119, 11, 32),      # bicycle
            -1: (0, 0, 0)         # license plate
            # 5: (111, 74, 0),        # dynamic
            # 6: (81,  0, 81),        # ground
            # 9: (250, 170, 160),     # parking
            # 10: (230, 150, 140),    # rail track
            # 14: (180, 165, 180),    # guard rail
            # 15: (150, 100, 100),    # bridge
            # 16: (150, 120, 90),     # tunnel
            # 18: (153, 153, 153),    # polegroup
            # 29: (0,  0, 90),        # caravan
            # 30: (0,  0, 110),       # trailer
            }

    _filenames = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            self._prefix_list = np.unique(np.array([el[:6]
                                                    for el in self.filenames]))

        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            # Get file names for this set
            filenames = []
            import scipy.io
            split = scipy.io.loadmat(os.path.join(path, 'split.mat'))
            split = split[self.which_set + "Ids"]

            for i in range(1, self.max_files):
                filenames.append(str(i).zfill(5)+'.png')
            # for id in split:
            #     filenames.append(str(id[0]).zfill(5)+'.png')
            self._filenames = filenames
            # print(filenames)
        return self._filenames

    def __init__(self, which_set='train', *args, **kwargs):
	print(path)
	self.which_set = "val" if which_set == "valid" else which_set
        self.image_path = os.path.join(path, "images")
        self.mask_path = os.path.join(path, "labels")
	print(self.image_path)
	print(self.mask_path)
        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(GTA5Dataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_subset_names = {}
        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            per_subset_names[prefix] = [el for el in filenames if
                                        el.startswith(prefix)]
        return per_subset_names

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from skimage import io
        from PIL import Image
        import numpy as np
        X = []
        Y = []
        F = []

        for prefix, frame in sequence:
            img = io.imread(os.path.join(self.image_path, frame))
            img = img.astype(floatX) / 255.

            # mask = io.imread(os.path.join(self.mask_path, frame))
            mask = Image.open(os.path.join(self.mask_path, frame))
            mask = np.array(mask)
            mask = mask.astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(frame)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():

    trainiter = GTA5Dataset(
        which_set='train',
        batch_size=10,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=True)

    validiter = GTA5Dataset(
        which_set='valid',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    nbatches = trainiter.nbatches
    print("Train %d" % (train_nsamples))

    valid_nsamples = validiter.nsamples
    print("Valid %d" % (valid_nsamples))

    # Simulate training
    max_epochs = 2
    start_training = time.time()
    for epoch in range(max_epochs):
        start_epoch = time.time()
        for mb in range(nbatches):
            start_batch = time.time()
            trainiter.next()
            print("Minibatch {}: {} seg".format(mb, (time.time() -
                                                     start_batch)))
        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
