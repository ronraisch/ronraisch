from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
import os
from logging import warning
from typing import Any, Type, List

from enum import Enum, auto

ROOT_DIR = Path('..')


class Columns(Enum):
    SAMPLE_ID = 'מספר דגימה'
    SLIDE_ID = 'מספר סלייד'
    FP_POSITION = 'מספר טביעה'
    IMPRINT_DATE = 'תאריך הטבעה'
    IMPRINT_TIME = 'שעת הטבעה'
    SAMPLE_DATE = 'תאריך דגימה'
    SAMPLE_TIME = 'שעת דגימה'
    DONOR_AGE = 'גיל תורם'
    DONOR_SEX = 'מין תורם'
    DONOR_NAME = 'תורם'
    QUALITY = 'איכות'
    COMMENTS = 'הערות'
    TARGET = auto()


class Records:

    records_mapping = {col.value: col for col in Columns}

    def __init__(self, filename: str):
        self.filename = filename
        self.records = self._load_records()
        self._rename_columns()

    def _load_records(self) -> pd.DataFrame:
        """
        Load records from a CSV file.
        """
        file_path = ROOT_DIR / 'data' / self.filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        df = pd.read_csv(file_path)
        if df.empty:
            warning(f"File {file_path} is empty.")
        return df

    def _rename_columns(self):
        """
        Rename columns to match the Records class mapping.
        """
        self.records.rename(columns=self.records_mapping, inplace=True)

    def __assert_column(self, column_name: Columns):
        """
        Assert that a column exists in the records.
        """
        if column_name not in self.records.columns:
            raise ValueError(
                f"Column {column_name} does not exist in the records.")

    def _remove_samples_with_quality_comments(self, drop_quality: bool = True):
        """
        Remove rows with quality comments. it is assumed that quality comments are always bad comments.
        """
        self.__assert_column(Columns.QUALITY)
        self.records = self.records[self.records[Columns.QUALITY.name].isna()]
        if drop_quality:
            self.records.drop(columns=[Columns.QUALITY.name], inplace=True)

    def _remove_bad_samples(self, bad_samples: List[int]):
        """
        Remove samples that are in the bad_samples list.
        """
        self.__assert_column(Columns.SAMPLE_ID)
        self.records = self.records[~self.records[Columns.SAMPLE_ID].isin(
            bad_samples)]

    def _remove_bad_slides(self, bad_slides: List[int]):
        """
        Remove slides that are in the bad_slides list.
        """
        self.__assert_column(Columns.SLIDE_ID)
        self.records = self.records[~self.records[Columns.SLIDE_ID].isin(
            bad_slides)]

    def _remove_column(self, column_name: Columns):
        """
        Remove a column from the records.
        """
        self.__assert_column(column_name)
        self.records.drop(columns=[column_name.name], inplace=True)

    def _remove_columns(self, columns: List[Columns]):
        """
        Remove multiple columns from the records.
        """
        for column in columns:
            self._remove_column(column)

    def _remove_rows_nan(self, nan_values: List[str]):
        for nan_value in nan_values:
            self.records = self.records.replace(nan_value, np.nan)
        self.records.dropna(axis=0, how='any', inplace=True)

    def __calc_full_time(self, date_column: Columns, time_column) -> pd.Series:
        """
        Calculate full time from date and time columns.
        """
        self.__assert_column(date_column)
        self.__assert_column(time_column)
        date_str = self.records[date_column].astype(str) + ' ' + self.records[time_column].astype(str)
        return pd.to_datetime(date_str, format='mixed', dayfirst=True, yearfirst=False)

        
    def _set_target(self):
        sample_full =  self.__calc_full_time(Columns.SAMPLE_DATE, Columns.SAMPLE_TIME)
        imprint_full = self.__calc_full_time(Columns.IMPRINT_DATE, Columns.IMPRINT_TIME)
        self.records[Columns.TARGET] =  (sample_full - imprint_full).dt.total_seconds() / 60/ 60/ 24
        self.records[Columns.TARGET] = self.records[Columns.TARGET].astype(float)


    def _second_iteration_main(self):
        bad_samples = [1528]
        bad_slides = [156, 150, 159, 149, 158]
        self._remove_samples_with_quality_comments()
        self._remove_bad_samples(bad_samples)
        self._remove_bad_slides(bad_slides)
        redundant_columns = [Columns.SAMPLE_ID, Columns.DONOR_NAME,
                             Columns.DONOR_AGE, Columns.DONOR_SEX, Columns.COMMENTS]
        self._remove_columns(redundant_columns)
        self._remove_rows_nan(['x', 'X'])
        self._set_target()
        self._remove_columns([Columns.IMPRINT_DATE, Columns.IMPRINT_TIME, Columns.SAMPLE_DATE, Columns.SAMPLE_TIME])

    def preprocess(self):
        """
        Main preprocessing function.
        """
        self._second_iteration_main()
