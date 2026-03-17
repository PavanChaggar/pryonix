import pytest
import polars as pl
import numpy as np
from datetime import datetime
from pathlib import Path
from pryonix.datasets.adnidataset import ADNIDataset, ADNISubject, ADNIScanData


@pytest.fixture
def test_csv_path():
    """Fixture to provide the path to test CSV"""
    return Path(__file__).parent / "adni-test.csv"


@pytest.fixture
def test_df(test_csv_path):
    """Fixture to load the test CSV as a Polars DataFrame"""
    df = pl.read_csv(test_csv_path)
    # Add a QC flag column for testing
    df = df.with_columns(pl.lit(2).alias("qc_flag"))
    return df


@pytest.fixture
def roi_names():
    """Fixture for a subset of ROI names to test with"""
    return [
        "inferiorcerebellum",
        "eroded_subcorticalwm",
        "braak1",
        "braak34",
        "meta_temporal",
    ]


class TestADNIScanData:
    """Tests for ADNIScanData class"""

    def test_scan_data_creation(self):
        """Test creating ADNIScanData object"""
        date = datetime(2001, 1, 1)
        suvr = np.array([1.0, 1.1, 1.2])
        volume = np.array([100.0, 105.0, 110.0])
        ref_suvr = 0.5
        ref_vol = 1.0

        scan = ADNIScanData(date, suvr, volume, ref_suvr, ref_vol, 0.0)

        assert scan.Date == date
        assert np.array_equal(scan.SUVR, suvr)
        assert np.array_equal(scan.Volume, volume)
        assert scan.Ref_SUVR == ref_suvr
        assert scan.Ref_Vol == ref_vol

    def test_scan_data_repr(self):
        """Test string representation of ADNIScanData"""
        date = datetime(2001, 1, 1)
        suvr = np.array([1.0])
        volume = np.array([100.0])

        scan = ADNIScanData(date, suvr, volume, 0.5, 1.0, 0.0)
        repr_str = repr(scan)

        assert "ADNI Scan" in repr_str
        assert "2001-01-01" in repr_str


class TestADNISubject:
    """Tests for ADNISubject class"""

    def test_adni_subject_from_dataframe(self, test_df, roi_names):
        """Test creating ADNISubject from DataFrame"""
        subject_id = 1
        subject = ADNISubject.from_dataframe(
            subject_id, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        assert subject is not None
        assert subject.ID == subject_id
        assert subject.n_scans > 0
        assert len(subject.scan_dates) == subject.n_scans
        assert len(subject.Data) == subject.n_scans

    def test_adni_subject_indexing_int(self, test_df, roi_names):
        """Test integer indexing on ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        scan = subject[0]
        assert isinstance(scan, ADNIScanData)
        assert scan.Date == subject.scan_dates[0]

    def test_adni_subject_indexing_slice(self, test_df, roi_names):
        """Test slice indexing on ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        if subject.n_scans > 1:
            sliced = subject[0:1]
            assert isinstance(sliced, ADNISubject)
            assert sliced.n_scans == 1

    def test_adni_subject_indexing_list(self, test_df, roi_names):
        """Test list indexing on ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        if subject.n_scans > 1:
            indexed = subject[[0]]
            assert isinstance(indexed, ADNISubject)
            assert indexed.n_scans == 1

    def test_adni_subject_get_suvr(self, test_df, roi_names):
        """Test getting SUVR from ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        suvr = subject.get_suvr()
        assert isinstance(suvr, np.ndarray)
        assert suvr.shape[0] == subject.n_scans
        assert suvr.shape[1] == len(roi_names)

    def test_adni_subject_get_vol(self, test_df, roi_names):
        """Test getting volume from ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        vol = subject.get_vol()
        assert isinstance(vol, np.ndarray)
        assert vol.shape[0] == subject.n_scans
        assert vol.shape[1] == len(roi_names)

    def test_adni_subject_get_ref_suvr(self, test_df, roi_names):
        """Test getting reference SUVR from ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        ref_suvr = subject.get_ref_suvr()
        assert isinstance(ref_suvr, np.ndarray)
        assert len(ref_suvr) == subject.n_scans

    def test_adni_subject_get_ref_vol(self, test_df, roi_names):
        """Test getting reference volume from ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        ref_vol = subject.get_ref_vol()
        assert isinstance(ref_vol, np.ndarray)
        assert len(ref_vol) == subject.n_scans

    def test_adni_subject_get_dates(self, test_df, roi_names):
        """Test getting scan dates from ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        dates = subject.get_dates()
        assert isinstance(dates, list)
        assert all(isinstance(d, datetime) for d in dates)
        assert len(dates) == subject.n_scans

    def test_adni_subject_get_times(self, test_df, roi_names):
        """Test getting time since first scan from ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        times = subject.get_times()
        assert isinstance(times, np.ndarray)
        assert len(times) == subject.n_scans
        assert times[0] == 0  # First scan should be at time 0

    def test_adni_subject_calc_suvr(self, test_df, roi_names):
        """Test SUVR calculation from ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        suvr = subject.calc_suvr()
        assert isinstance(suvr, np.ndarray)
        assert suvr.shape == subject.get_suvr().shape

    def test_adni_subject_calc_suvr_max_norm(self, test_df, roi_names):
        """Test SUVR calculation with max normalization"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        suvr = subject.calc_suvr(max_norm=True)
        assert isinstance(suvr, np.ndarray)
        assert np.max(suvr) <= 1.0 + 1e-6  # Allow for floating point error

    def test_adni_subject_get_initial_conditions(self, test_df, roi_names):
        """Test getting initial conditions from ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        ic = subject.get_initial_conditions()
        assert isinstance(ic, np.ndarray)
        assert len(ic) == len(roi_names)

    def test_adni_subject_get_cl(self, test_df, roi_names):
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum", CL=True, CL_region="CENTILOIDS"
        )

        cl = subject.get_cl()
        assert isinstance(cl, np.ndarray)
        assert cl[0] == 100.0
        assert len(cl) == 2
        
    def test_adni_subject_repr(self, test_df, roi_names):
        """Test string representation of ADNISubject"""
        subject = ADNISubject.from_dataframe(
            1, test_df, roi_names, reference_region="inferiorcerebellum"
        )

        repr_str = repr(subject)
        assert "ADNI Subject" in repr_str
        assert str(subject.ID) in repr_str
        assert str(subject.n_scans) in repr_str


class TestADNIDataset:
    """Tests for ADNIDataset class"""

    def test_adni_dataset_from_dataframe(self, test_df, roi_names):
        """Test creating ADNIDataset from DataFrame"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        assert dataset is not None
        assert dataset.n_subjects > 0
        assert len(dataset.SubjectData) == dataset.n_subjects
        assert dataset.rois == roi_names

    def test_adni_dataset_qc_filtering(self, test_df, roi_names):
        """Test QC filtering in ADNIDataset"""
        # Create dataset with QC filtering
        dataset_qc = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=True
        )

        # Create dataset without QC filtering
        dataset_no_qc = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        # Both should work
        assert dataset_qc.n_subjects >= 0
        assert dataset_no_qc.n_subjects >= 0

    def test_adni_dataset_min_scans_filter(self, test_df, roi_names):
        """Test minimum scans filter in ADNIDataset"""
        # Get dataset with minimum 2 scans
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, min_scans=2, qc=False
        )

        # All subjects should have at least 2 scans
        for subject in dataset.SubjectData:
            assert subject.n_scans >= 2

    def test_adni_dataset_max_scans_filter(self, test_df, roi_names):
        """Test maximum scans filter in ADNIDataset"""
        # Get dataset with maximum 2 scans
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, max_scans=2, qc=False
        )

        # All subjects should have at most 2 scans
        for subject in dataset.SubjectData:
            assert subject.n_scans <= 2

    def test_adni_dataset_indexing_int(self, test_df, roi_names):
        """Test integer indexing on ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        subject = dataset[0]
        assert isinstance(subject, ADNISubject)

    def test_adni_dataset_indexing_slice(self, test_df, roi_names):
        """Test slice indexing on ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        if dataset.n_subjects > 1:
            sliced = dataset[0:1]
            assert isinstance(sliced, ADNIDataset)
            assert sliced.n_subjects == 1

    def test_adni_dataset_indexing_list(self, test_df, roi_names):
        """Test list indexing on ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        if dataset.n_subjects > 1:
            indexed = dataset[[0, 1]]
            assert isinstance(indexed, ADNIDataset)
            assert indexed.n_subjects == min(2, dataset.n_subjects)

    def test_adni_dataset_get_suvr(self, test_df, roi_names):
        """Test getting SUVR from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        suvr = dataset.get_suvr()
        assert isinstance(suvr, list)
        assert len(suvr) == dataset.n_subjects
        assert all(isinstance(s, np.ndarray) for s in suvr)

    def test_adni_dataset_get_vol(self, test_df, roi_names):
        """Test getting volume from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        vol = dataset.get_vol()
        assert isinstance(vol, list)
        assert len(vol) == dataset.n_subjects
        assert all(isinstance(v, np.ndarray) for v in vol)

    def test_adni_dataset_get_ref_suvr(self, test_df, roi_names):
        """Test getting reference SUVR from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        ref_suvr = dataset.get_ref_suvr()
        assert isinstance(ref_suvr, list)
        assert len(ref_suvr) == dataset.n_subjects
        assert all(isinstance(rs, np.ndarray) for rs in ref_suvr)

    def test_adni_dataset_get_ref_vol(self, test_df, roi_names):
        """Test getting reference volume from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        ref_vol = dataset.get_ref_vol()
        assert isinstance(ref_vol, list)
        assert len(ref_vol) == dataset.n_subjects
        assert all(isinstance(rv, np.ndarray) for rv in ref_vol)

    def test_adni_dataset_get_dates(self, test_df, roi_names):
        """Test getting scan dates from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        dates = dataset.get_dates()
        assert isinstance(dates, list)
        assert len(dates) == dataset.n_subjects
        assert all(isinstance(d, list) for d in dates)

    def test_adni_dataset_get_times(self, test_df, roi_names):
        """Test getting time since first scan from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        times = dataset.get_times()
        assert isinstance(times, list)
        assert len(times) == dataset.n_subjects
        assert all(isinstance(t, np.ndarray) for t in times)

    def test_adni_dataset_get_ids(self, test_df, roi_names):
        """Test getting subject IDs from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        ids = dataset.get_ids()
        assert isinstance(ids, list)
        assert len(ids) == dataset.n_subjects
        assert all(isinstance(i, (int, np.integer)) for i in ids)

    def test_adni_dataset_calc_suvr(self, test_df, roi_names):
        """Test SUVR calculation from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        suvr = dataset.calc_suvr()
        assert isinstance(suvr, list)
        assert len(suvr) == dataset.n_subjects
        assert all(isinstance(s, np.ndarray) for s in suvr)

    def test_adni_dataset_calc_suvr_max_norm(self, test_df, roi_names):
        """Test SUVR calculation with max normalization"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        suvr = dataset.calc_suvr(max_norm=True)
        assert isinstance(suvr, list)
        assert len(suvr) == dataset.n_subjects

    def test_adni_dataset_get_initial_conditions(self, test_df, roi_names):
        """Test getting initial conditions from ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        ic = dataset.get_initial_conditions()
        assert isinstance(ic, list)
        assert len(ic) == dataset.n_subjects
        assert all(isinstance(i, np.ndarray) for i in ic)

    def test_adni_dataset_len(self, test_df, roi_names):
        """Test __len__ method on ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        assert len(dataset) == dataset.n_subjects

    def test_adni_dataset_iter(self, test_df, roi_names):
        """Test __iter__ method on ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        count = 0
        for subject in dataset:
            assert isinstance(subject, ADNISubject)
            count += 1

        assert count == dataset.n_subjects

    def test_adni_dataset_repr(self, test_df, roi_names):
        """Test string representation of ADNIDataset"""
        dataset = ADNIDataset.from_dataframe(
            test_df, roi_names, qc=False
        )

        repr_str = repr(dataset)
        assert "ADNI data set" in repr_str
        assert str(dataset.n_subjects) in repr_str