from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
import re

from lightning.data.dataset_utils import generate_random_keep


class FilterBase:
    def __init__(self, field_name: str):
        self.field_name = field_name

    def find_indices(self, data: pd.DataFrame) -> pd.Series:
        """
        The method locates the indices which when applied to the input data, these are reduced according to a strategy.
        The strategy is implemented in the derived class.
        The indices are 1 if the data entry should be retained, else 0.

        @param data: Input data in form of list of dicts
        @return: The indices of the filtered data
        """
        # Check for correct field name. The check could be done before conversion, however I prefer to convert in order
        # to also check for inconsistent data values (rare case)
        if self.field_name not in data.columns:
            raise NotImplementedError(f'Name {self.field_name} must be among {list(data.columns.array)}')

        kept_indices = self._find_indices_impl(data)
        if (kept_indices != 0).sum() == 0:
            raise RuntimeError(f'Applied filter "{type(self).__name__}" kept none indices')

        return kept_indices

    @abstractmethod
    def _find_indices_impl(self, data: pd.DataFrame) -> pd.Series:
        pass


class ListFilter(FilterBase):
    """
    Keep data entries whose field_name value is among values which are given as a list
    """

    def __init__(self, field_name: str, values: List):
        super().__init__(field_name)
        self.values = values

    def _find_indices_impl(self, data: pd.DataFrame) -> pd.Series:
        data_col = data[self.field_name]
        return data_col.isin(self.values)


class RegexListFilter(FilterBase):
    """
    Keep data entries whose field_name value is matches one of the values which are given as a list using a regex
    """
    def __init__(self, field_name: str, regex_values: List[str]):
        super().__init__(field_name)
        self.regex_values = regex_values

    def _find_indices_impl(self, data: pd.DataFrame) -> pd.Series:
        data_col = data[self.field_name]

        # The set below contains the actual description names that were matched from the regex values
        # It is set in order to avoid multiple matched of the same original value
        resolved_values = set()

        # Find unique values such that we can compare against them
        unique_values = data_col.unique()

        # Check all given regex values against all unique values
        for r_val in self.regex_values:
            for val in unique_values:
                if re.search(r_val, val):
                    resolved_values.add(val)

        # Return the actual
        return data_col.isin(resolved_values)


class RangeFilter(FilterBase):
    """
    Keep data entries whose field_name value is among values which are given as a range
    """

    def __init__(self, field_name: str, range_: range):
        super().__init__(field_name)

        if range_.step != 1:
            raise NotImplementedError('Ranges of step 1 are the only supported')

        self.start = range_.start
        # For the case of range we want to convert exclusive stop value to inclusive end number
        self.stop = range_.stop - 1

    def _find_indices_impl(self, data: pd.DataFrame) -> pd.Series:
        data_col = data[self.field_name]
        return data_col.between(self.start, self.stop, inclusive='both')


class GlobalRandomKeepFilter(FilterBase):
    """
    Randomly sample how many of the data entries will be kept among all data_entries, with probability keep_probability
    """

    def __init__(self, field_name: str, keep_probability=0.5):
        super().__init__(field_name)
        self.keep_probability = keep_probability

    def _find_indices_impl(self, data: pd.DataFrame) -> pd.Series:
        data_col = data[self.field_name]
        keep_indices = generate_random_keep(n_samples=len(data_col), keep_probability=self.keep_probability)
        return pd.Series(keep_indices)


class LocalRandomKeepFilter(FilterBase):
    """
    Randomly sample how many of the data entries will be kept among each individual values data_entries,
    with probability keep_probability. E.g. if field_name corresponds to class number, entries per class
    are iterated and sampling is on those entries.
    """

    def __init__(self, field_name: str, keep_probability=0.5):
        super().__init__(field_name)
        self.keep_probability = keep_probability

    def _find_indices_impl(self, data: pd.DataFrame) -> pd.Series:
        data_col = data[self.field_name]
        keep_indices = np.ones(len(data_col)) * -1
        labels = data_col.values
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_indices = np.where(labels == label)
            random_ones = generate_random_keep(n_samples=len(label_indices[0]), keep_probability=self.keep_probability)
            keep_indices[label_indices] = random_ones

        return pd.Series(keep_indices)


class ConditionFilter(FilterBase):
    """
    Filters the dataset based on a condition which holds for a field.

    Examples:
        (a) retain only classes which have at least n samples
    """
    def _find_indices_impl(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class FilterManager:
    def __init__(self, indices_filters: List[FilterBase]):
        self.indices_filters = indices_filters

    def apply_filters(self, data: List[Dict]) -> List[Dict]:
        # Convert to dataframe
        data_df = pd.DataFrame(data)

        # Initially consider all indices
        selected_indices = pd.Series(np.ones(shape=(data_df.shape[0],), dtype=bool))

        # Progressively remove indices by applying a 'boolean and' operator
        # I chose to raise exceptions if the condition is wrong rather than bypassing the specific condition, because
        # I assume this should never be intentional or accidental
        for indices_filter in self.indices_filters:
            selected_indices &= indices_filter.find_indices(data_df)

        # Filter data based on indices
        data_df = data_df.loc[selected_indices]

        # Convert back to list of dicts
        filtered_data = data_df.to_dict('records')
        return filtered_data


def filter_by(data, conditions_list: List[FilterBase]):
    filter_manager = FilterManager(conditions_list)
    return filter_manager.apply_filters(data)