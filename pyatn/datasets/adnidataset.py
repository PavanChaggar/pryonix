from .petdataset import PETDataset, PETSubject, PETScanData
import polars as pl
import numpy as np
from datetime import datetime
from typing import List
from dataclasses import dataclass

# ADNI-specific classes
@dataclass
class ADNIScanData(PETScanData):
    """ADNI-specific scan data"""
    def __repr__(self):
        d = self.Date
        return f"ADNI Scan: {d}."


@dataclass
class ADNISubject(PETSubject[ADNIScanData]):
    """ADNI-specific subject class"""
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
        subref_suvr = np.array(sub[suvr_name(reference_region)].to_list(), dtype=float)
        subref_vol = np.array(sub[vol_name(reference_region)].to_list(), dtype=float)
        
        n_scans = len(subdate)
        
        if n_scans == subsuvr.shape[0] == subvol.shape[0]:
            subject_data = [
                ADNIScanData(
                    subdate[i], 
                    subsuvr[i, :].astype(float), 
                    subvol[i, :].astype(float), 
                    float(subref_suvr[i]), 
                    float(subref_vol[i])
                )
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


class ADNIDataset(PETDataset[ADNISubject]):
    """ADNI dataset class inheriting from PETDataset"""
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

    def __repr__(self):
        n_subs = self.n_subjects
        n_scans = sum([sub.n_scans for sub in self.SubjectData])
        return f"ADNI data set with {n_subs} subjects and {n_scans} scans."


# Helper functions
def suvr_name(roi):
    return f"{roi.upper()}_SUVR"

def vol_name(roi):
    return f"{roi.upper()}_VOLUME"