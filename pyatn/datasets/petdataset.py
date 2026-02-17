import polars as pl
import numpy as np
from datetime import datetime
from typing import List, Generic, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Type variables for generic classes
ScanDataType = TypeVar('ScanDataType', bound='PETScanData')
SubjectType = TypeVar('SubjectType', bound='PETSubject')

# Generic base classes
@dataclass
class PETScanData(ABC):
    """Base class for PET scan data"""
    Date: datetime
    SUVR: np.ndarray  # 1D array of SUVR values
    Volume: np.ndarray  # 1D array of volume values
    Ref_SUVR: float
    Ref_Vol: float
    CL: float | None
    @abstractmethod
    def __repr__(self):
        pass


@dataclass
class PETSubject(ABC, Generic[ScanDataType]):
    """Base class for PET subjects"""
    ID: int
    n_scans: int
    scan_dates: List[datetime]
    Data: List[ScanDataType]

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def get_suvr(self) -> np.ndarray:
        """Get SUVR for a given subject by index"""
        return np.vstack([scan.SUVR for scan in self.Data])

    def get_ref_suvr(self) -> np.ndarray:
        """Get reference SUVR for a given subject by index"""
        return np.array([scan.Ref_SUVR for scan in self.Data])

    def get_ref_vol(self) -> np.ndarray:
        """Get reference volume for a given subject by index"""
        return np.array([scan.Ref_Vol for scan in self.Data])

    def get_vol(self) -> np.ndarray:
        """Get volume for a given subject by index"""
        return np.vstack([scan.Volume for scan in self.Data])

    def get_dates(self) -> List[datetime]:
        """Get scan dates for a given subject by index"""
        return self.scan_dates

    def get_times(self) -> np.ndarray:
        """Get time in years since first scan for a given subject by index"""
        dates = np.array(self.scan_dates)
        days = (dates - np.min(dates)).astype('timedelta64[D]').astype(int)
        return days / 365

    def get_id(self) -> int:
        """Get ID for a given subject by index"""
        return self.ID

    def calc_suvr(self, max_norm: bool = False) -> np.ndarray:
        """Calculate SUVR for a given subject by index"""
        suvr = self.get_suvr()
        ref_suvr = self.get_ref_suvr()
        result = (suvr.T / ref_suvr).T
        if max_norm:
            return result / np.max(result)
        return result

    def get_initial_conditions(self, max_norm: bool = False) -> np.ndarray:
        """Get initial conditions (first SUVR) for a given subject by index"""
        suvr = self.calc_suvr(max_norm=max_norm)
        return suvr[0]

    def get_cl(self) -> np.ndarray:
        return np.array([scan.CL for scan in self.Data])

class PETDataset(ABC, Generic[SubjectType]):
    """Base class for PET datasets"""
    def __init__(self, n_subjects: int, SubjectData: List[SubjectType], rois: List[str]):
        self.n_subjects = n_subjects
        self.SubjectData = SubjectData
        self.rois = rois

    def get_suvr(self) -> List[np.ndarray]:
        """Get SUVR for all subjects"""
        return [sub.get_suvr() for sub in self.SubjectData]

    def get_ref_suvr(self) -> List[np.ndarray]:
        """Get reference SUVR for all subjects"""
        return [sub.get_ref_suvr() for sub in self.SubjectData]

    def get_ref_vol(self) -> List[np.ndarray]:
        """Get reference volume for all subjects"""
        return [sub.get_ref_vol() for sub in self.SubjectData]

    def get_vol(self) -> List[np.ndarray]:
        """Get volume for all subjects"""
        return [sub.get_vol() for sub in self.SubjectData]

    def get_dates(self) -> List[List[datetime]]:
        """Get scan dates for all subjects"""
        return [sub.get_dates() for sub in self.SubjectData]

    def get_times(self) -> List[np.ndarray]:
        """Get time in years since first scan for all subjects"""
        return [sub.get_times() for sub in self.SubjectData]

    def get_ids(self) -> List[int]:
        """Get IDs for all subjects"""
        return [sub.get_id() for sub in self.SubjectData]

    def calc_suvr(self, max_norm: bool = False) -> List[np.ndarray]:
        """Calculate SUVR for all subjects"""
        return [sub.calc_suvr(max_norm=max_norm) for sub in self.SubjectData]

    def get_initial_conditions(self, max_norm: bool = False) -> List[np.ndarray]:
        """Get initial conditions (first SUVR) for all subjects"""
        return [sub.get_initial_conditions(max_norm=max_norm) for sub in self.SubjectData]

    def get_cl(self) -> List[np.ndarray]:
        return [sub.get_cl() for sub in self.SubjectData]
    
    def __iter__(self):
        return iter(self.SubjectData)

    def __len__(self):
        return len(self.SubjectData)
        
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.SubjectData[idx]
        elif isinstance(idx, slice):
            sd = self.SubjectData[idx]
            return self.__class__(len(sd), sd, self.rois)
        elif isinstance(idx, list):
            sd = [self.SubjectData[i] for i in idx]
            return self.__class__(len(sd), sd, self.rois)
        else:
            raise TypeError("Invalid index type")

    @abstractmethod
    def __repr__(self):
        pass

