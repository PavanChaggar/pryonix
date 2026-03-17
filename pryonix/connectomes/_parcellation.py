import numpy as np
from ._graphml import load_parcellation
# Define the Region class
class Region:
    def __init__(self, ID, Label, Cortex, Lobe, Hemisphere, x, y, z):
        self.ID = ID
        self.Label = Label
        self.Cortex = Cortex
        self.Lobe = Lobe
        self.Hemisphere = Hemisphere
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Region {self.ID}: {self.Hemisphere} {self.Label}"

    def get_coords(self):
        return np.array([self.x, self.y, self.z])

# Define the Parcellation class
class Parcellation:
    def __init__(self, regions=None):
        if regions is None:
            regions = []
        self.regions = regions

    @classmethod
    def from_lists(cls, ids, labels, cortex, lobes, hemispheres, xs, ys, zs):
        regions = [
            Region(ids[i], labels[i], cortex[i], lobes[i], hemispheres[i], xs[i], ys[i], zs[i])
            for i in range(len(ids))
        ]
        return cls(regions)

    @classmethod
    def from_path(cls, path):
        # Replace with your own logic to load parcellation from a path
        ids, labels, cortex, lobes, hemispheres, xs, ys, zs = load_parcellation(path)
        return cls.from_lists(ids, labels, cortex, lobes, hemispheres, xs, ys, zs)

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.regions[idx]
        elif isinstance(idx, list):
            return Parcellation([self.regions[_idx] for _idx in idx])
        else:
            raise TypeError(f"Indexing type {type(idx)} not supported")

    def __iter__(self):
        for region in self.regions:
            yield region

    def get_coords(self):
        return np.array([region.get_coords() for region in self.regions])

    def filter(self, func):
        filtered_regions = filter(func, self.regions)
        return Parcellation(list(filtered_regions))

    def __repr__(self):
        return f"Parcellation with {len(self)} regions"

# Helper functions (similar to the Julia versions)
def get_node_id(parc):
    return [roi.ID.item() for roi in parc]

def get_label(parc):
    return [roi.Label for roi in parc]

def get_cortex(parc):
    return [roi.Cortex for roi in parc]

def get_lobe(parc):
    return [roi.Lobe for roi in parc]

def get_hemisphere(parc):
    return [roi.Hemisphere for roi in parc]

def get_coords(parc):
    return [roi.get_coords() for roi in parc]