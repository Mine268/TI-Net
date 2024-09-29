from .FreiHand import FreiHand
from .DexYCB import DexYCB

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    """Data loader wrapper with background prefetching"""

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
