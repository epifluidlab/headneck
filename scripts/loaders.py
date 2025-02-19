import os
import glob
import numpy as np
import pandas as pd
from inmoose.pycombat import pycombat_norm

class HNSCCFeatureHandler:
	def __init__(self, metadata_file: str, valid_indices_file: str):
		with open(valid_indices_file, "r") as f:
			self.valid_ids = {line.strip() for line in f}
		self.metadata = pd.read_csv(metadata_file)
		self.metadata = self.metadata[self.metadata["ID"].isin(self.valid_ids)]
		self.screen = self.metadata[self.metadata["Type of Visit"]=="Screen"]
		self.dayzero = self.metadata[self.metadata["Type of Visit"]=="Day 0"]
		self.adjwk1 = self.metadata[self.metadata["Type of Visit"]=="Adj Wk 1"]
		self.screen_ids, self.day0_ids, self.adjwk1_ids = set(self.screen["ID"]), set(self.dayzero["ID"]), set(self.adjwk1["ID"])
		self.institute1 = self.metadata[self.metadata["Institute"]==1]
		self.institute2 = self.metadata[self.metadata["Institute"]==2]
		self.institute3 = self.metadata[self.metadata["Institute"]==3]
		self.institute4 = self.metadata[self.metadata["Institute"]==4]
		self.institute5 = self.metadata[self.metadata["Institute"]==5]
		self.institute6 = self.metadata[self.metadata["Institute"]==6]
		self.institute1_ids, self.institute2_ids, self.institute3_ids, self.institute4_ids, self.institute5_ids, self.institute6_ids = set(self.institute1["ID"]), set(self.institute2["ID"]), set(self.institute3["ID"]), set(self.institute4["ID"]), set(self.institute5["ID"]), set(self.institute6["ID"])
		self.responder = self.metadata[self.metadata["Treatment Response"]=="Responder"]
		self.non_responder = self.metadata[self.metadata["Treatment Response"]=="Non-Responder"]
		self.responder_ids, self.non_responder_ids = set(self.responder["ID"]), set(self.non_responder["ID"])
		self.metadata.set_index('ID', inplace=True)
		self.features = None
		self.data = None

	def load_feature_to_dataframe(self, file_pattern: str, skip_rows: int, feature_col_index: int, na_value: str = None):
		dfs=[]
		suffix = os.path.basename(file_pattern).replace('*', '')
		for file in glob.glob(file_pattern):
			filename = os.path.basename(file)
			col_name = filename.replace(suffix, '')
			df = pd.read_csv(file, delimiter='\t', usecols=[0, 1, 2, feature_col_index], skiprows=skip_rows, header=None, names=['chr', 'start', 'end', 'feature'])
			if na_value is not None:
				df['feature'] = df['feature'].replace(na_value, np.nan)
			df['feature'] = df['feature'].astype(float)
			df[col_name] = df['feature']
			df.drop(columns=['feature'], inplace=True)
			df['chr:start-end'] = df['chr'] + ':' + df['start'].astype(str) + '-' + df['end'].astype(str)
			df.set_index('chr:start-end', inplace=True)
			df.drop(columns=['chr', 'start', 'end'], inplace=True)
			dfs.append(df)
		final_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True, how='outer')
		for df in dfs[2:]:
			final_df = pd.merge(final_df, df, left_index=True, right_index=True, how='outer')
		final_df = final_df.dropna()
		self.features = final_df
		return final_df

	def normalize_zscore(self):
		if self.features is None:
			raise ValueError("No features loaded yet. Load features using the load_feature_to_dataframe method.")
		
		final_df = self.features.copy()
		
		for column in final_df.columns:
			if column in self.valid_ids:
				col_mean = final_df[column].mean()
				col_std = final_df[column].std()
				
				if col_std != 0:
					final_df[column] = (final_df[column] - col_mean) / col_std
				else:
					final_df[column] = 0
		
		self.features = final_df
		return final_df

	
	def normalize_total_sum(self):
		if self.features is None:
			raise ValueError("No features loaded yet. Load features using the load_feature_to_dataframe method.")
		final_df = self.features.copy()
		for column in final_df.columns:
			if column in self.valid_ids:
				column_sum = final_df[column].sum()
				final_df[column] = final_df[column] / column_sum
			else:
				final_df = final_df.drop(columns=column)
		self.features = final_df
		return final_df
	
	def normalize_min_max(self):
		if self.features is None:
			raise ValueError("No features loaded yet. Load features using the load_feature_to_dataframe method.")
		
		final_df = self.features.copy()
		
		for column in final_df.columns:
			if column in self.valid_ids:
				col_min = final_df[column].min()
				col_max = final_df[column].max()
				
				if col_max != col_min: 
					final_df[column] = (final_df[column] - col_min) / (col_max - col_min)
				else:
					final_df[column] = 0 
		
		self.features = final_df
		return final_df

	
	def merge_feature_metadata(self):
		if self.features is None:
			raise ValueError("No features loaded yet. Load features using the load_feature_to_dataframe method.")
		transposed_features = self.features.T
		#transposed_features.set_index('ID', inplace=True)
		merged_data = self.metadata.merge(transposed_features, left_index=True, right_index=True)
		#merged_data.set_index('ID', inplace=True)
		self.data = merged_data
		return merged_data

	def get_subset(self, *ids, df: pd.DataFrame = None,):
		if df is None:
			if self.data is None:
				raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
			df = self.data
		intersection = ids[0].copy()
		for id_set in ids[1:]:
			intersection &= id_set.copy()
		if df is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
		return df.loc[df.index.isin(intersection)]

	def get_raw_features(self, df: pd.DataFrame = None):
		if df is None:
			if self.data is None:
				raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
			df = self.data
		chr_columns = [col for col in df.columns if col.startswith('chr')]
		return df[chr_columns]
		
	def get_metadata_col(self, col_name: str, df: pd.DataFrame = None):
		if df is None:
			if self.data is None:
				raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
			df = self.data
		return df[col_name]

	def batch_correct(self):
		if self.data is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
		raw_data = self.get_raw_features()
		metadata_cols = self.data.drop(columns=raw_data.columns)
		raw_data = raw_data.T
		institute = self.get_metadata_col('Institute')
		#pilot = self.get_metadata_col('Pilot')==1
		#batch = self.get_metadata_col('Batches')
		treatment = list(self.get_metadata_col('Treatment Response'))
		corrected_features = pycombat_norm(raw_data, institute, covar_mod=treatment).T
		#corrected_features = pycombat_norm(corrected_features, batch, covar_mod=treatment)
		#corrected_features = pycombat_norm(corrected_features, pilot, covar_mod=treatment).T
		self.data = pd.concat([metadata_cols, corrected_features], axis=1)

	def filter_locations(self, locations_file: str):
		locations = pd.read_csv(locations_file, delimiter='\t', header=None, names=['chr', 'start', 'end'])
		location_strings = set(locations.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1))
		matching_columns = [col for col in self.data.columns if col in location_strings or not col.startswith('chr')]
		return self.data[matching_columns]

			