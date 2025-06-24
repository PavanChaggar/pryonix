import polars as pl
import numpy as np
from datetime import datetime
from typing import List
from dataclasses import dataclass

@dataclass
class ADNIScanData:
    Date: datetime
    SUVR: List[float]
    Volume: List[float]
    Ref_SUVR: float
    Ref_Vol: float

    def __repr__(self):
        d = self.Date
        return f"ADNI Scan: {d}."

@dataclass
class ADNISubject:
    ID: int
    n_scans: int
    scan_dates: List[datetime]
    Data: List[ADNIScanData]

    @classmethod
    def from_dataframe(cls, subid, df, roi_names, reference_region="inferiorcerebellum"):
        """
        Class method to create an ADNISubject from a Polars DataFrame.
        """
        sub = df.filter(pl.col("RID") == subid)
        
        if "EXAMDATE" in sub.columns:
            dates = sub["EXAMDATE"].to_list()
            subdate = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        elif "SCANDATE" in sub.columns:
            dates = sub["SCANDATE"].to_list()
            subdate = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

        subsuvr = np.array([sub[suvr_name(roi)] for roi in roi_names]).T
        subvol = np.array([sub[vol_name(roi)] for roi in roi_names]).T
        subref_suvr = sub[suvr_name(reference_region)].to_list()
        subref_vol = sub[vol_name(reference_region)].to_list()
        
        n_scans = len(subdate)
        
        if n_scans == subsuvr.shape[0] == subvol.shape[0]:
            subject_data = [
                ADNIScanData(subdate[i], subsuvr[i, :], subvol[i, :], subref_suvr[i], subref_vol[i])
                for i in range(n_scans)
            ]
            return cls(subid, n_scans, subdate, subject_data)
        return None
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.Data[idx]
        elif isinstance(idx, slice):
            _data = self.Data[idx]
            _times = self.scan_dates[idx]
            return ADNISubject(self.ID, len(_data), _times, _data)
        elif isinstance(idx, list):
            _data = [self.Data[i] for i in idx]
            _times = [self.scan_dates[i] for i in idx]
            return ADNISubject(self.ID, len(_data), _times, _data)
        else:
            raise TypeError("Invalid index type")
    
    def __repr__(self):
        id = self.ID
        n_scans = self.n_scans
        return f"ADNI Subject {id} with {n_scans} scans."

    def get_suvr(self):
        """ Get SUVR for a given subject by index """
        return np.vstack([scan.SUVR for scan in self.Data])
    
    def get_ref_suvr(self):
        """ Get reference SUVR for a given subject by index """
        return [scan.Ref_SUVR for scan in self.Data]

    def get_ref_vol(self):
        """ Get reference volume for a given subject by index """
        return [scan.Ref_Vol for scan in self.Data]
    
    def get_vol(self):
        """ Get volume for a given subject by index """
        return np.vstack([scan.Volume for scan in self.Data])

    def get_dates(self):
        """ Get scan dates for a given subject by index """
        return self.scan_dates

    def get_times(self):
        """ Get time in years since first scan for a given subject by index """
        dates = np.array(self.scan_dates)
        days = (dates - np.min(dates)).astype('timedelta64[D]').astype(int)
        return days / 365

    def get_id(self):
        """ Get ID for a given subject by index """
        return self.ID

    def calc_suvr(self, max_norm=False):
        """ Calculate SUVR for a given subject by index """
        suvr = self.get_suvr()
        ref_suvr = self.get_ref_suvr()
        result = (suvr.T / ref_suvr).T
        if max_norm:
            return result / np.max(result)
        return result
    
    def get_initial_conditions(self, max_norm=False):
        """ Get initial conditions (first SUVR) for a given subject by index """
        suvr = self.calc_suvr(max_norm=max_norm)
        return suvr[0]


class ADNIDataset:
    def __init__(self, n_subjects, SubjectData, rois):
        self.n_subjects = n_subjects
        self.SubjectData = SubjectData
        self.rois = rois

    @classmethod
    def from_dataframe(cls, df, roi_names, min_scans=1, max_scans=np.inf, reference_region="inferiorcerebellum", qc=True):
        """
        Class method to create an ADNIDataset from a Polars DataFrame.
        """
        if qc:
            df = df.filter(pl.col("qc_flag") == 2)  # check QC status
        
        subjects = df["RID"].unique().to_list()
        n_scans = [df.filter(pl.col("RID") == sub).height for sub in subjects]
        
        multi_subs = [sub for sub, scans in zip(subjects, n_scans) if min_scans <= scans <= max_scans]
        
        adnisubjects = []
        for sub in multi_subs:
            subject = ADNISubject.from_dataframe(sub, df, roi_names, reference_region)
            if isinstance(subject, ADNISubject):
                adnisubjects.append(subject)
        
        return cls(len(adnisubjects), adnisubjects, roi_names)

    def get_suvr(self):
        """ Get SUVR for a given subject by index """
        return [sub.get_suvr() for sub in self.SubjectData]

    def get_ref_suvr(self):
        """ Get reference SUVR for a given subject by index """
        return [sub.get_ref_suvr() for sub in self.SubjectData]

    def get_ref_vol(self):
        """ Get reference volume for a given subject by index """
        return [sub.get_ref_vol() for sub in self.SubjectData]

    def get_vol(self):
        """ Get volume for a given subject by index """
        return [sub.get_vol() for sub in self.SubjectData]

    def get_dates(self):
        """ Get scan dates for a given subject by index """
        return [sub.get_dates() for sub in self.SubjectData]

    def get_times(self):
        """ Get time in years since first scan for a given subject by index """
        return [sub.get_times() for sub in self.SubjectData]

    def get_ids(self):
        """ Get ID for a given subject by index """
        return [sub.get_id() for sub in self.SubjectData]

    def calc_suvr(self, max_norm=False):
        """ Calculate SUVR for a given subject by index """
        return [sub.calc_suvr(max_norm=max_norm) for sub in self.SubjectData]

    def get_initial_conditions(self, max_norm):
        """ Get initial conditions (first SUVR) for a given subject by index """
        return [sub.get_initial_conditions(max_norm=max_norm) for sub in self.SubjectData]

    def __repr__(self):
        n_subs = self.n_subjects
        n_scans = sum([sub.n_scans for sub in self.SubjectData])
        return f"ADNI data set with {n_subs} subjects and {n_scans} scans."
    
    def __iter__(self):
        return iter(self.SubjectData)

    def __len__(self):
        return len(self.SubjectData)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.SubjectData[idx]
        elif isinstance(idx, slice) or isinstance(idx, list):
            sd = [self.SubjectData[i] for i in idx]
            return ADNIDataset(len(sd), sd, self.rois)
        else:
            raise TypeError("Invalid index type")


def suvr_name(roi):
    return f"{roi.upper()}_SUVR"

def vol_name(roi):
    return f"{roi.upper()}_VOLUME"