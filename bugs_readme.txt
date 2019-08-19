- extract_subjects.py fails with integer overflow error from pandas, while computing patient age at admission (add_age_to_icustays)
- reason seems to be nanosecond resolution added in pandas 0.23.0 - this prevents pandas from being able to handle Timedelta greater than 106,751 days,
  which is exceeded by patients > 89 years of age, since MIMIC-III sets their DOB as 300 years before date of admission
Ref:
- https://github.com/pandas-dev/pandas/issues/12727
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html
- pd.Timedelta.max = 106,751 days
- 300 years = 109,500 days



- preprocessing.py/make_phenotype_label_matrix fails due to using deprecated/altered pandas function Dataframe.ix
- fix: use Dataframe.reindex on array input (sort_values.values)
