import os
import glob
import patsy
import math
import random
import joblib
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from patsy.contrasts import Sum
import matplotlib.pyplot as plt
from collections import Counter
from inmoose.pycombat import pycombat_norm
from itertools import combinations
from sklearn.base import clone
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from lifelines.statistics import logrank_test
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

class HNSCCFeatureHandler:
	def __init__(self, metadata_file: str, cv_indices_file: str, hold_out_indices_file: str):
		with open(cv_indices_file, "r") as f:
			self.valid_ids = {line.strip() for line in f}
		with open(hold_out_indices_file, "r") as f:
			self.hold_ids = {line.strip() for line in f}
		self.metadata = pd.read_csv(metadata_file)
		self.hold_metadata = self.metadata[self.metadata["ID"].isin(self.hold_ids)]
		self.metadata = self.metadata[self.metadata["ID"].isin(self.valid_ids)]
		self.screen = self.metadata[self.metadata["Type of Visit"]=="Screen"]
		self.dayzero = self.metadata[self.metadata["Type of Visit"]=="Day 0"]
		self.adjwk1 = self.metadata[self.metadata["Type of Visit"]=="Adj Wk 1"]
		self.screen_ids, self.day0_ids, self.adjwk1_ids = set(self.screen["ID"]), set(self.dayzero["ID"]), set(self.adjwk1["ID"])

		self.hold_screen = self.hold_metadata[self.hold_metadata["Type of Visit"]=="Screen"]
		self.hold_dayzero = self.hold_metadata[self.hold_metadata["Type of Visit"]=="Day 0"]
		self.hold_adjwk1 = self.hold_metadata[self.hold_metadata["Type of Visit"]=="Adj Wk 1"]
		self.hold_screen_ids, self.hold_day0_ids, self.hold_adjwk1_ids = set(self.hold_screen["ID"]), set(self.hold_dayzero["ID"]), set(self.hold_adjwk1["ID"])
		
		self.institute1 = self.metadata[self.metadata["Institute"]=="University of Cincinnati"]
		self.institute2 = self.metadata[self.metadata["Institute"]=="Ohio State University"]
		self.institute3 = self.metadata[self.metadata["Institute"]=="Medical University of South Carolina"]
		self.institute4 = self.metadata[self.metadata["Institute"]=="University of Michigan"]
		self.institute5 = self.metadata[self.metadata["Institute"]=="University of Louisville"]
		self.institute6 = self.metadata[self.metadata["Institute"]=="MD Anderson"]
		self.institute1_ids, self.institute2_ids, self.institute3_ids, self.institute4_ids, self.institute5_ids, self.institute6_ids = set(self.institute1["ID"]), set(self.institute2["ID"]), set(self.institute3["ID"]), set(self.institute4["ID"]), set(self.institute5["ID"]), set(self.institute6["ID"])
		self.train_ids = self.institute1_ids | self.institute2_ids | self.institute3_ids | self.institute4_ids
		self.test_ids = self.institute5_ids | self.institute6_ids
		self.responder = self.metadata[self.metadata["Treatment Response"]=="Responder"]
		self.non_responder = self.metadata[self.metadata["Treatment Response"]=="Non-Responder"]
		self.responder_ids, self.non_responder_ids = set(self.responder["ID"]), set(self.non_responder["ID"])
		self.metadata.set_index('ID', inplace=True)
		self.hold_metadata.set_index('ID', inplace=True)
		self.features = None
		self.data = None
		self.hold_data = None

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
			if column in self.valid_ids or column in self.hold_ids:
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
			if column in self.valid_ids or column in self.hold_ids:
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
			if column in self.valid_ids or column in self.hold_ids:
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
		merged_data = self.metadata.merge(transposed_features, left_index=True, right_index=True)
		self.data = merged_data
		merged_data = self.hold_metadata.merge(transposed_features, left_index=True, right_index=True)
		self.hold_data = merged_data
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
		chr_columns = [col for col in df.columns if ":" in col]
		return df[chr_columns]
		
	def get_metadata_col(self, col_name: str, df: pd.DataFrame = None):
		if df is None:
			if self.data is None:
				raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
			df = self.data
		return df[col_name]

	def batch_correct(self):
		def limma(exprs, covariate_matrix, design_matrix, rcond=1e-8):
			design_batch = np.hstack((covariate_matrix, design_matrix))
			coefficients, _, _, _ = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)
			beta = coefficients[-design_matrix.shape[1]:]
			return exprs - design_matrix.dot(beta).T, beta
		def limma_apply(exprs, covariate_matrix, design_matrix, beta, rcond=1e-8):
			if beta.shape[0] != design_matrix.shape[1]:
				raise ValueError("The number of columns in beta and design_matrix do not match.")
			corrected_exprs = exprs - design_matrix.dot(beta).T
			return corrected_exprs
		def limma_nocov(exprs, covariate_matrix, design_matrix, rcond=1e-8):
			coefficients, _, _, _ = np.linalg.lstsq(design_matrix, exprs.T, rcond=rcond)
			beta = coefficients[-design_matrix.shape[1]:]
			return exprs - design_matrix.dot(beta).T, beta
		def get_dmatrix(categories, all_levels):
			contrast = Sum()
			contrast_matrix = contrast.code_without_intercept(all_levels).matrix
			contrast_dict = {level: contrast_matrix[i] for i, level in enumerate(all_levels)}

			dmatrix = np.array([contrast_dict.get(cat, np.zeros(len(all_levels) - 1)) for cat in categories])
			return dmatrix
		
		if self.data is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
		raw_data = self.get_raw_features()
		metadata_cols = self.data.drop(columns=raw_data.columns)
		raw_data = raw_data.T
		batch1 = pd.Categorical(self.get_metadata_col('Institute'))
		#batch2 = pd.Categorical(self.get_metadata_col('Pilot'))
		batch3 = pd.Categorical(self.get_metadata_col('WGS Library Prep Date'))
		batch4 = pd.Categorical(self.get_metadata_col('cfDNA Isolation Date'))

		all_institutes = sorted(set(batch1.categories) | set(self.get_metadata_col('Institute', df=self.hold_data).unique()))
		all_prep_dates = sorted(set(batch3.categories) | set(self.get_metadata_col('WGS Library Prep Date', df=self.hold_data).unique()))
		all_isolation_dates = sorted(set(batch4.categories) | set(self.get_metadata_col('cfDNA Isolation Date', df=self.hold_data).unique()))

		contrast_key = {
			'Institute': get_dmatrix(batch1, all_institutes),
			'Library_Prep_Date': get_dmatrix(batch3, all_prep_dates),
			'cfDNA_Isolation_Date': get_dmatrix(batch4, all_isolation_dates),
		}

		design = np.concatenate([contrast_key[key] for key in contrast_key], axis=1)
		covariates = get_dmatrix(pd.Categorical(self.get_metadata_col('Treatment Response')), 
							 list(self.get_metadata_col('Treatment Response').unique()))
		corrected_data, beta = limma(raw_data, covariates, design)
		#corrected_data, beta = limma_nocov(raw_data, None, design)
		self.data = pd.concat([metadata_cols, corrected_data.T], axis=1)
		
		raw_data = self.get_raw_features(df=self.hold_data)
		metadata_cols = self.hold_data.drop(columns=raw_data.columns)
		raw_data = raw_data.T

		batch1 = pd.Categorical(self.get_metadata_col('Institute', df=self.hold_data))
		batch3 = pd.Categorical(self.get_metadata_col('WGS Library Prep Date', df=self.hold_data))
		batch4 = pd.Categorical(self.get_metadata_col('cfDNA Isolation Date', df=self.hold_data))

		# Use the same shared levels
		contrast_key = {
			'Institute': get_dmatrix(batch1, all_institutes),
			'Library_Prep_Date': get_dmatrix(batch3, all_prep_dates),
			'cfDNA_Isolation_Date': get_dmatrix(batch4, all_isolation_dates),
		}

		design = np.concatenate([contrast_key[key] for key in contrast_key], axis=1)

		corrected_data = limma_apply(raw_data, None, design, beta)
		self.hold_data = pd.concat([metadata_cols, corrected_data.T], axis=1)
		return self.data

	def filter_locations(self, locations_file: str):
		locations = pd.read_csv(locations_file, delimiter='\t', header=None, names=['chr', 'start', 'end'])
		location_strings = list(locations.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1))
		matching_columns = [col for col in self.data.columns if not ":" in col]
		matching_columns += location_strings
		return self.data[matching_columns]
	
	def batch_correct_pycombat(self):
		if self.data is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
		raw_data = self.get_raw_features()
		metadata_cols = self.data.drop(columns=raw_data.columns)
		raw_data = raw_data.T
		institute = self.get_metadata_col('Institute')
		lpd = self.get_metadata_col('WGS Library Prep Date')
		cid = self.get_metadata_col('cfDNA Isolation Date')
		batch = self.get_metadata_col('Batches')

		treatment = list(self.get_metadata_col('Treatment Response'))
		corrected_features = pycombat_norm(raw_data, institute, covar_mod=treatment).T
		#corrected_features = pycombat_norm(corrected_features, lpd, covar_mod=treatment)
		#corrected_features = pycombat_norm(corrected_features, cid, covar_mod=treatment).T
		self.data = pd.concat([metadata_cols, corrected_features], axis=1)

	def pca(self, label: str, scale=True, save_path: str = None, before=True, plot_ellipses=False):
		if self.data is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")

		raw_data = self.get_raw_features()
		pca = PCA(n_components=2)
		scaler = StandardScaler() if scale else FunctionTransformer(lambda x: x)
		X = pd.DataFrame(scaler.fit_transform(raw_data), index=raw_data.index, columns=raw_data.columns)

		pca_data = pca.fit_transform(X)
		pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'], index=X.index)
		pca_df[label] = self.get_metadata_col(label)

		# Apply mapping if label is 'Institute'
		if label == "Institute":
			institute_label_map = {
				"University of Cincinnati": "UC",
				"University of Michigan": "U-M",
				"MD Anderson": "MD Anderson",
				"University of Louisville": "UofL",
				"Ohio State University": "OSU",
				"Medical University of South Carolina": "MUSC"
			}
			pca_df[label] = pca_df[label].map(institute_label_map).fillna(pca_df[label])

		unique_labels = pca_df[label].unique()
		colors = sns.color_palette("tab10", len(unique_labels))
		color_map = {label_val: colors[i] for i, label_val in enumerate(unique_labels)}

		plt.figure(figsize=(6, 3))
		for label_l, color in color_map.items():
			subset = pca_df[pca_df[label] == label_l]
			plt.scatter(subset['PC1'], subset['PC2'], c=color, label=label_l, s=2)

			if plot_ellipses and len(subset) > 1:
				mean = subset[['PC1', 'PC2']].mean()
				cov = subset[['PC1', 'PC2']].cov()
				lambda_, v = np.linalg.eig(cov)
				lambda_ = np.sqrt(lambda_)
				angle = np.rad2deg(np.arccos(v[0, 0]))
				ellipse = plt.matplotlib.patches.Ellipse(
					xy=mean, width=lambda_[0]*2, height=lambda_[1]*2, angle=angle,
					edgecolor=color, facecolor=color, alpha=0.2, linewidth=1
				)
				plt.gca().add_patch(ellipse)

		plt.xlabel('PC1')
		plt.ylabel('PC2')
		plt.ylim(-25, 25)
		plt.legend(title=label, bbox_to_anchor=(1.05, 1), loc='upper left')
		ax = plt.gca()
		if before:
			plt.title('Before Batch Correction')
		else:
			plt.title('After Batch Correction')
		for spine in ['top', 'right']:
			ax.spines[spine].set_visible(False)
		plt.tight_layout()

		if save_path is not None:
			plt.savefig(save_path, dpi=1000)

		plt.show()

	def train_test_model(self, model, hyperparameter_dict, train, test, hold=None, k=10, name=None):
		cv_split=10
		if hold is None:
			hold_data_for_ml_eval = pd.concat([self.get_raw_features(df=self.hold_data), self.get_metadata_col('Treatment Response', df=self.hold_data)], axis=1) 
		else:
			hold_data_for_ml_eval = hold
		def load_split(split_no: int):
			def to_list(filepath):
				with open(filepath, "r") as f:
					return [line.strip() for line in f]
			train_split = to_list(f"/projects/b1198/epifluidlab/ravi/0401/headneck/notebooks/lists/train_fold_{split_no}.txt")
			val_split = to_list(f"/projects/b1198/epifluidlab/ravi/0401/headneck/notebooks/lists/test_fold_{split_no}.txt")
			return train_split, val_split

		results = []
		best_overall_model = None
		best_overall_score = -np.inf
		best_overall_params = None
		feature_selection_counts = Counter()

		plt.figure(figsize=(8, 16), dpi=300)

		for split_no in range(cv_split):
			print(f"Processing split {split_no + 1}/{cv_split}...")
			train_split, val_split = load_split(split_no)
			X_train, y_train = train.loc[train_split].drop(columns=['Treatment Response']), train.loc[train_split]['Treatment Response']=='Responder'
			X_val, y_val = train.loc[val_split].drop(columns=['Treatment Response']), train.loc[val_split]['Treatment Response']=='Responder'

			selector = SelectKBest(f_classif, k=k)
			X_train_selected = selector.fit_transform(X_train, y_train)
			X_val_selected = selector.transform(X_val)
			X_test_selected = selector.transform(test.drop(columns=['Treatment Response']))

			selected_features = selector.get_support()
			selected_features = X_train.columns[selected_features]
			feature_selection_counts.update(selected_features)

			best_model = None
			best_score = -np.inf

			for params in ParameterGrid(hyperparameter_dict):
				model.set_params(**params)
				model.fit(X_train_selected, y_train)
				y_pred = model.predict(X_val_selected)
				score = balanced_accuracy_score(y_val, y_pred)
				if score > best_score:
					best_score = score
					best_model = model
					best_params = params

			if best_score > best_overall_score:
				best_overall_score = best_score
				best_overall_model = best_model
				best_overall_params = best_params

			y_prob = best_model.predict_proba(X_test_selected)[:, 1]
			fpr, tpr, _ = roc_curve(test['Treatment Response']=='Responder', y_prob)
			plt.subplot(4, 2, 1)
			plt.plot(fpr, tpr, color='grey', alpha=0.1)

			y_pred = best_model.predict(X_test_selected)
			score = balanced_accuracy_score(test['Treatment Response']=='Responder', y_pred)
			results.append(
				{
					"split_no": split_no,
					"best_params": best_params,
					"validation_score": best_score,
					"test_score": score,
				}
			)

		top_features = [feature for feature, count in feature_selection_counts.most_common(k)]
		X_train, y_train = train.drop(columns=['Treatment Response']), train['Treatment Response']=='Responder'
		X_test, y_test = test.drop(columns=['Treatment Response']), test['Treatment Response']=='Responder'

		X_test_screen, y_test_screen = self.get_subset(self.screen_ids, df=test).drop(columns=['Treatment Response']), self.get_subset(self.screen_ids, df=test)['Treatment Response']=='Responder'
		X_test_day0, y_test_day0 = self.get_subset(self.day0_ids, df=test).drop(columns=['Treatment Response']), self.get_subset(self.day0_ids, df=test)['Treatment Response']=='Responder'
		X_test_adjwk1, y_test_adjwk1 = self.get_subset(self.adjwk1_ids, df=test).drop(columns=['Treatment Response']), self.get_subset(self.adjwk1_ids, df=test)['Treatment Response']=='Responder'

		X_train_selected = X_train.loc[:, top_features]
		X_test_selected = X_test.loc[:, top_features]
		best_overall_model.set_params(**best_overall_params)
		best_overall_model.fit(X_train_selected, y_train)

		for idx, (X, y, title) in enumerate([(X_test, y_test, 'Overall'), (X_test_screen, y_test_screen, 'Screen'), (X_test_day0, y_test_day0, 'Day 0'), (X_test_adjwk1, y_test_adjwk1, 'Adj Week 1')], start=1):
			X_selected = X.loc[:, top_features]
			y_prob = best_overall_model.predict_proba(X_selected)[:, 1]
			fpr, tpr, _ = roc_curve(y, y_prob)
			plt.subplot(4, 2, 2 * idx - 1)
			plt.plot(fpr, tpr, color='mediumpurple')
			plt.text(0.4, 0.05, f'ROC AUC: {roc_auc_score(y, y_prob):.2f}', fontsize=12, color='black')
			plt.plot([0, 1], [0, 1], color='black', alpha=0.3, linestyle='--')
			plt.title(f'{title} ROC Curve')
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.xlim(0, 1)
			plt.ylim(0, 1)

			y_pred = best_overall_model.predict(X_selected)
	
			y_labels = np.where(y, 'Responder', 'Non-Responder')
			y_pred_labels = np.where(y_pred, 'Responder', 'Non-Responder')
			cm = confusion_matrix(y_labels, y_pred_labels, labels=['Responder', 'Non-Responder'])
			plt.subplot(4, 2, 2 * idx)
			sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
					xticklabels=['Responder', 'Non-Responder'],
					yticklabels=['Responder', 'Non-Responder'])
			plt.title(f'{title} Confusion Matrix')
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')

		plt.tight_layout()
		if name is not None:
			plt.savefig(name+'.pdf', dpi=300)
		plt.show()

		X_test, y_test = hold_data_for_ml_eval.drop(columns=['Treatment Response']), hold_data_for_ml_eval['Treatment Response']=='Responder'
		X_test_screen, y_test_screen = self.get_subset(self.hold_screen_ids, df=hold_data_for_ml_eval).drop(columns=['Treatment Response']), self.get_subset(self.hold_screen_ids, df=hold_data_for_ml_eval)['Treatment Response']=='Responder'
		X_test_day0, y_test_day0 = self.get_subset(self.hold_day0_ids, df=hold_data_for_ml_eval).drop(columns=['Treatment Response']), self.get_subset(self.hold_day0_ids, df=hold_data_for_ml_eval)['Treatment Response']=='Responder'
		X_test_adjwk1, y_test_adjwk1 = self.get_subset(self.hold_adjwk1_ids, df=hold_data_for_ml_eval).drop(columns=['Treatment Response']), self.get_subset(self.hold_adjwk1_ids, df=hold_data_for_ml_eval)['Treatment Response']=='Responder'

		plt.figure(figsize=(8, 16), dpi=300)
		for idx, (X, y, title) in enumerate([(X_test, y_test, 'Overall'), (X_test_screen, y_test_screen, 'Screen'), (X_test_day0, y_test_day0, 'Day 0'), (X_test_adjwk1, y_test_adjwk1, 'Adj Week 1')], start=1):
			X_selected = X.loc[:, top_features]
			y_prob = best_overall_model.predict_proba(X_selected)[:, 1]
			fpr, tpr, _ = roc_curve(y, y_prob)
			plt.subplot(4, 2, 2 * idx - 1)
			plt.plot(fpr, tpr, color='mediumpurple')
			plt.text(0.4, 0.05, f'ROC AUC: {roc_auc_score(y, y_prob):.2f}', fontsize=12, color='black')
			plt.plot([0, 1], [0, 1], color='black', alpha=0.3, linestyle='--')
			plt.title(f'{title} ROC Curve')
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.xlim(0, 1)
			plt.ylim(0, 1)

			y_pred = best_overall_model.predict(X_selected)
	
			y_labels = np.where(y, 'Responder', 'Non-Responder')
			y_pred_labels = np.where(y_pred, 'Responder', 'Non-Responder')
			cm = confusion_matrix(y_labels, y_pred_labels, labels=['Responder', 'Non-Responder'])
			plt.subplot(4, 2, 2 * idx)
			sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
					xticklabels=['Responder', 'Non-Responder'],
					yticklabels=['Responder', 'Non-Responder'])
			plt.title(f'{title} Confusion Matrix')
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')

		plt.tight_layout()
		if name is not None:
			plt.savefig(name+'.hold.out.pdf', dpi=300)
		#plt.show()

		if name.startswith("Overall"):
			feature_importances = pd.DataFrame({
				'Feature': X_train_selected.columns,
				'Importance': best_overall_model.feature_importances_
			}).sort_values(by='Importance', ascending=False)

			plt.figure(figsize=(4, 2), dpi=300)
			sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='Purples_r')
			plt.title('Feature Importance')
			plt.tight_layout()
			plt.savefig(name+'.feature.importance.pdf', dpi=300)

		y_prob = best_overall_model.predict_proba(X_test.loc[:, top_features])[:,1]
		y_pred = best_overall_model.predict(X_test.loc[:, top_features])
		score = balanced_accuracy_score(y_test, y_pred)
		results.append(
			{
				"split_no": "Overall",
				"best_params": best_overall_params,
				"validation_score": best_overall_score,
				"test_score": score,
				"predictions": y_pred.tolist(),
				"true_labels": y_test.tolist(),
			}
		)
		all_data = pd.concat([train, test])
		print(f"Best Overall Model: {best_overall_model}")
		print(f"Top Features: {top_features}")
		all_data['Predicted Treatment Probabilities'] = best_overall_model.predict_proba(all_data.drop(columns=['Treatment Response']).loc[:, top_features])[:, 1]
		all_data['Predicted Treatment Response'] = best_overall_model.predict(all_data.drop(columns=['Treatment Response']).loc[:, top_features])

		if hold is not None:
			self.hold_data['Predicted Treatment Probabilities'] = best_overall_model.predict_proba(hold.drop(columns=['Treatment Response']).loc[:, top_features])[:, 1]
			self.hold_data['Predicted Treatment Response'] = best_overall_model.predict(hold.drop(columns=['Treatment Response']).loc[:, top_features])
		else:
			self.hold_data['Predicted Treatment Probabilities'] = best_overall_model.predict_proba(self.hold_data.drop(columns=['Treatment Response']).loc[:, top_features])[:, 1]
			self.hold_data['Predicted Treatment Response'] = best_overall_model.predict(self.hold_data.drop(columns=['Treatment Response']).loc[:, top_features])
		screen_dictionary_preds = {}
		day0_dictionary_preds = {}
		adjwk1_dictionary_preds = {}
		for index, prediction in zip(list(self.screen_ids), all_data.loc[list(self.screen_ids)]['Predicted Treatment Response']):
			screen_dictionary_preds[index] = "Responder" if prediction else "Non-Responder"
		for index, prediction in zip(list(self.day0_ids), all_data.loc[list(self.day0_ids)]['Predicted Treatment Response']):
			day0_dictionary_preds[index] = "Responder" if prediction else "Non-Responder"
		for index, prediction in zip(list(self.adjwk1_ids), all_data.loc[list(self.adjwk1_ids)]['Predicted Treatment Response']):
			adjwk1_dictionary_preds[index] = "Responder" if prediction else "Non-Responder"
		
		preds_dictionary = [screen_dictionary_preds, day0_dictionary_preds, adjwk1_dictionary_preds]
		return results, best_overall_model, top_features, preds_dictionary, all_data, self.hold_data[['Patient Number', 'Type of Visit', 'Treatment Response', 'Predicted Treatment Probabilities', 'Predicted Treatment Response']]

	
	def train_test_model_nested(self, model, hyperparameter_dict, train, test, hold=None, k=10, name=None):
		cv_split = 10
		# Prepare hold-out data if not provided.
		if hold is None:
			hold_data_for_ml_eval = pd.concat([
				self.get_raw_features(df=self.hold_data),
				self.get_metadata_col('Treatment Response', df=self.hold_data)
			], axis=1)
		else:
			hold_data_for_ml_eval = hold

		def load_split(split_no: int):
			def to_list(filepath):
				with open(filepath, "r") as f:
					return [line.strip() for line in f]
			train_split = [to_list(f"/projects/b1198/epifluidlab/ravi/0401/headneck/notebooks/lists/train_fold_{split_no}_{i}.txt")
						for i in range(cv_split)]
			val_split = [to_list(f"/projects/b1198/epifluidlab/ravi/0401/headneck/notebooks/lists/val_fold_{split_no}_{i}.txt")
						for i in range(cv_split)]
			test_split = to_list(f"/projects/b1198/epifluidlab/ravi/0401/headneck/notebooks/lists/test_fold_{split_no}.txt")
			return train_split, val_split, test_split

		results = []          # to store fold-wise metrics and chosen hyperparameters
		fold_predictions = {}
		outer_selected_features_list = []  # collect outer-selected features per fold
		outer_best_params_list = []        # collect best hyperparameters per outer fold

		plt.figure(figsize=(8, 16), dpi=300)

		def evaluate_params(params, train_values, train_labels, inner_train_splits, inner_val_splits):
			scores = []
			for inner in range(len(inner_train_splits)):
				inner_train_idx = inner_train_splits[inner]
				inner_val_idx = inner_val_splits[inner]

				X_train_inner = train_values.loc[inner_train_idx]
				y_train_inner = train_labels[inner_train_idx]
				X_val_inner = train_values.loc[inner_val_idx]
				y_val_inner = train_labels[inner_val_idx]

				selector = SelectKBest(f_classif, k=k)
				X_train_sel = selector.fit_transform(X_train_inner, y_train_inner)
				X_val_sel = selector.transform(X_val_inner)

				curr_model = clone(model)
				curr_model.set_params(**params, random_state=np.random.RandomState(40))
				curr_model.fit(X_train_sel, y_train_inner)
				y_pred_inner = curr_model.predict(X_val_sel)
				score_inner = balanced_accuracy_score(y_val_inner, y_pred_inner)
				scores.append(score_inner)
			return np.mean(scores)
		param_grid = list(ParameterGrid(hyperparameter_dict))
		# Outer CV loop
		for split_no in range(cv_split):
			print(f"Processing outer split {split_no + 1}/{cv_split}...")
			inner_train_splits, inner_val_splits, outer_test_indices = load_split(split_no)
			best_score_outer = -np.inf
			best_params_outer = None
			milestones = [0, 25, 50, 75, 100]
			next_milestone_index = 0
			# Grid-search over hyperparameters (inner CV)
			train_values = train.drop(columns=['Treatment Response'])
			train_labels = (train['Treatment Response'] == 'Responder')
			
			# Parallel evaluation of parameter sets
			def eval_wrapper(p):
				return (p, evaluate_params(p, train_values, train_labels, inner_train_splits, inner_val_splits))

			parallel_results = Parallel(n_jobs=16, verbose=10)(
				delayed(eval_wrapper)(params) for params in param_grid
			)

			# Get best params from results
			best_params_outer, best_score_outer = max(parallel_results, key=lambda x: x[1])

			# After inner CV, use all inner splits (train+val) for the outer training step.
			outer_train_idx = list(set(sum(inner_train_splits, []) + sum(inner_val_splits, [])))
			X_outer_train = train.loc[outer_train_idx].drop(columns=['Treatment Response'])
			y_outer_train = (train.loc[outer_train_idx]['Treatment Response'] == 'Responder')

			# Outer feature selection using best hyperparameters from inner CV.
			outer_selector = SelectKBest(f_classif, k=k)
			X_outer_train_sel = outer_selector.fit_transform(X_outer_train, y_outer_train)
			outer_selected_features = list(X_outer_train.columns[outer_selector.get_support()])
			# Record the outer-selected features for consensus later.
			outer_selected_features_list.append(outer_selected_features)

			# Record the best hyperparameters for this fold.
			outer_best_params_list.append(best_params_outer)

			# Train the model using the outer training data.
			best_model_outer = clone(model)
			best_model_outer.set_params(**best_params_outer, random_state=np.random.RandomState(40))
			best_model_outer.fit(X_outer_train_sel, y_outer_train)

			# Evaluate on the outer test split.
			X_outer_test = train.loc[outer_test_indices].drop(columns=['Treatment Response'])
			y_outer_test = (train.loc[outer_test_indices]['Treatment Response'] == 'Responder')
			X_outer_test_sel = outer_selector.transform(X_outer_test)
			y_prob_outer = best_model_outer.predict_proba(X_outer_test_sel)[:, 1]
			y_pred_outer = best_model_outer.predict(X_outer_test_sel)

			# Compute ROC curve.
			fpr, tpr, _ = roc_curve(y_outer_test, y_prob_outer)

			# Plot the ROC (for all folds, using a grey, low-alpha line).
			plt.subplot(4, 2, 1)
			plt.plot(fpr, tpr, color='grey', alpha=0.1)

			# Compute additional metrics.
			acc = accuracy_score(y_outer_test, y_pred_outer)
			f1 = f1_score(y_outer_test, y_pred_outer)
			precision = precision_score(y_outer_test, y_pred_outer)
			recall = recall_score(y_outer_test, y_pred_outer)
			balanced_acc = balanced_accuracy_score(y_outer_test, y_pred_outer)
			roc_auc = roc_auc_score(y_outer_test, y_prob_outer)

			# Record fold metrics.
			results.append({
				"split_no": split_no,
				"best_params": best_params_outer,
				"inner_validation_score": best_score_outer,
				"outer_accuracy": acc,
				"outer_f1": f1,
				"outer_precision": precision,
				"outer_recall": recall,
				"outer_balanced_accuracy": balanced_acc,
				"outer_roc_auc": roc_auc,
				"outer_fpr": fpr.tolist(),
				"outer_tpr": tpr.tolist(),
				"predictions": y_pred_outer.tolist(),
				"true_labels": y_outer_test.tolist()
			})
			fold_predictions[split_no] = {
				"predictions": y_prob_outer.tolist(),
				"true_labels": y_outer_test.tolist(),
				"outer_test_indices": outer_test_indices
			}
		# Add a legend to the ROC plot.
		legend_elements = [Patch(facecolor='lightgray', edgecolor='none', label='CV Folds')]
		plt.legend(handles=legend_elements, loc='lower left', frameon=False)

		# Compute consensus feature set from outer folds.
		outer_feature_counts = Counter()
		for feat_list in outer_selected_features_list:
			outer_feature_counts.update(feat_list)
		consensus_features = [feat for feat, _ in outer_feature_counts.most_common(k)]
		
		# Compute consensus hyperparameters from outer folds.
		best_outer_fold = max(results, key=lambda x: x["outer_balanced_accuracy"])
		params_df = pd.DataFrame([fold["best_params"] for fold in results])
		mode_series = params_df.mode().iloc[0]
		consensus_hyper_params = best_outer_fold['best_params']
		
		print("Consensus Features:")
		print(consensus_features)
		print("Consensus Hyperparameters:")
		print(consensus_hyper_params)

		# Retrain final model on full training data using the consensus settings.
		X_train_final = train.drop(columns=['Treatment Response'])[consensus_features]
		y_train_final = (train['Treatment Response'] == 'Responder')
		final_model = clone(model)
		final_model.set_params(**consensus_hyper_params, random_state=np.random.RandomState(40))
		final_model.fit(X_train_final, y_train_final)

		# Evaluate final model on test data.
		X_test_final = test.drop(columns=['Treatment Response'])
		y_test_final = (test['Treatment Response'] == 'Responder')
		y_prob_final = final_model.predict_proba(X_test_final.loc[:, consensus_features])[:, 1]
		y_pred_final = final_model.predict(X_test_final.loc[:, consensus_features])
		overall_acc = accuracy_score(y_test_final, y_pred_final)
		overall_f1 = f1_score(y_test_final, y_pred_final)
		overall_precision = precision_score(y_test_final, y_pred_final)
		overall_recall = recall_score(y_test_final, y_pred_final)
		overall_bal_acc = balanced_accuracy_score(y_test_final, y_pred_final)
		overall_roc_auc = roc_auc_score(y_test_final, y_prob_final)
		fpr_final, tpr_final, _ = roc_curve(y_test_final, y_prob_final)

		results.append({
			"split_no": "Overall",
			"best_params": consensus_hyper_params,
			"validation_score": np.mean([d["inner_validation_score"] for d in results if d["split_no"] != "Overall"]),
			"test_accuracy": overall_acc,
			"test_f1": overall_f1,
			"test_precision": overall_precision,
			"test_recall": overall_recall,
			"test_balanced_accuracy": overall_bal_acc,
			"test_roc_auc": overall_roc_auc,
			"test_fpr": fpr_final.tolist(),
			"test_tpr": tpr_final.tolist(),
			"predictions": y_pred_final.tolist(),
			"true_labels": y_test_final.tolist(),
		})

		# Plot ROC curves and confusion matrices on several subsets (test, hold, etc.)
		for idx, (X, y, title) in enumerate([
			(test.drop(columns=['Treatment Response']), y_test_final, 'Overall'),
			(self.get_subset(self.screen_ids, df=test).drop(columns=['Treatment Response']),
			self.get_subset(self.screen_ids, df=test)['Treatment Response'] == 'Responder', 'Screen'),
			(self.get_subset(self.day0_ids, df=test).drop(columns=['Treatment Response']),
			self.get_subset(self.day0_ids, df=test)['Treatment Response'] == 'Responder', 'Day 0'),
			(self.get_subset(self.adjwk1_ids, df=test).drop(columns=['Treatment Response']),
			self.get_subset(self.adjwk1_ids, df=test)['Treatment Response'] == 'Responder', 'Adj Week 1')
		], start=1):
			X_sel = X.loc[:, consensus_features]
			y_prob = final_model.predict_proba(X_sel)[:, 1]
			y_pred = final_model.predict(X_sel)
			results.append({
				"split_no": f"Test - {title}",
				"accuracy": accuracy_score(y, y_pred),
				"f1": f1_score(y, y_pred),
				"precision": precision_score(y, y_pred),
				"recall": recall_score(y, y_pred),
				"balanced_accuracy": balanced_accuracy_score(y, y_pred),
				"roc_auc": roc_auc_score(y, y_prob)
			})
			fpr, tpr, _ = roc_curve(y, y_prob)
			plt.subplot(4, 2, 2 * idx - 1)
			plt.plot(fpr, tpr, color='mediumpurple')
			plt.text(0.55, 0.05, f'ROC AUC: {roc_auc_score(y, y_prob):.2f}', fontsize=12, color='red')
			plt.plot([0, 1], [0, 1], color='black', alpha=0.3, linestyle='--')
			plt.title(f'{title} ROC Curve')
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.xlim(-0.03, 1)
			plt.ylim(0, 1.03)

			y_pred = final_model.predict(X_sel)
			y_labels = np.where(y, 'Responder', 'Non-Responder')
			y_pred_labels = np.where(y_pred, 'Responder', 'Non-Responder')
			cm = confusion_matrix(y_labels, y_pred_labels, labels=['Responder', 'Non-Responder'])
			plt.subplot(4, 2, 2 * idx)
			sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
						xticklabels=['Responder', 'Non-Responder'],
						yticklabels=['Responder', 'Non-Responder'])
			plt.title(f'{title} Confusion Matrix')
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')

		plt.tight_layout()
		if name is not None:
			plt.savefig(name + '.pdf', dpi=300)
		plt.show()

		# Evaluate on hold-out data.
		X_hold = hold_data_for_ml_eval.drop(columns=['Treatment Response'])
		y_hold = (hold_data_for_ml_eval['Treatment Response'] == 'Responder')
		# If you have subsets of hold-out data, get them as below.
		X_hold_screen = self.get_subset(self.hold_screen_ids, df=hold_data_for_ml_eval).drop(columns=['Treatment Response'])
		y_hold_screen = (self.get_subset(self.hold_screen_ids, df=hold_data_for_ml_eval)['Treatment Response'] == 'Responder')
		X_hold_day0 = self.get_subset(self.hold_day0_ids, df=hold_data_for_ml_eval).drop(columns=['Treatment Response'])
		y_hold_day0 = (self.get_subset(self.hold_day0_ids, df=hold_data_for_ml_eval)['Treatment Response'] == 'Responder')
		X_hold_adjwk1 = self.get_subset(self.hold_adjwk1_ids, df=hold_data_for_ml_eval).drop(columns=['Treatment Response'])
		y_hold_adjwk1 = (self.get_subset(self.hold_adjwk1_ids, df=hold_data_for_ml_eval)['Treatment Response'] == 'Responder')

		plt.figure(figsize=(8, 16), dpi=300)
		for idx, (X, y, title) in enumerate([
			(X_hold, y_hold, 'Overall'),
			(X_hold_screen, y_hold_screen, 'Screen'),
			(X_hold_day0, y_hold_day0, 'Day 0'),
			(X_hold_adjwk1, y_hold_adjwk1, 'Adj Week 1')
		], start=1):
			X_sel = X.loc[:, consensus_features]
			y_prob = final_model.predict_proba(X_sel)[:, 1]
			fpr, tpr, _ = roc_curve(y, y_prob)
			y_pred = final_model.predict(X_sel)
			results.append({
				"split_no": f"Hold-out - {title}",
				"accuracy": accuracy_score(y, y_pred),
				"f1": f1_score(y, y_pred),
				"precision": precision_score(y, y_pred),
				"recall": recall_score(y, y_pred),
				"balanced_accuracy": balanced_accuracy_score(y, y_pred),
				"roc_auc": roc_auc_score(y, y_prob)
			})
			plt.subplot(4, 2, 2 * idx - 1)
			plt.plot(fpr, tpr, color='mediumpurple')
			plt.text(0.55, 0.05, f'ROC AUC: {roc_auc_score(y, y_prob):.2f}', fontsize=12, color='red')
			plt.plot([0, 1], [0, 1], color='black', alpha=0.3, linestyle='--')
			plt.title(f'{title} ROC Curve')
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.xlim(-0.03, 1)
			plt.ylim(0, 1.03)

			y_pred = final_model.predict(X_sel)
			y_labels = np.where(y, 'Responder', 'Non-Responder')
			y_pred_labels = np.where(y_pred, 'Responder', 'Non-Responder')
			cm = confusion_matrix(y_labels, y_pred_labels, labels=['Responder', 'Non-Responder'])
			plt.subplot(4, 2, 2 * idx)
			sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
						xticklabels=['Responder', 'Non-Responder'],
						yticklabels=['Responder', 'Non-Responder'])
			plt.title(f'{title} Confusion Matrix')
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')

		plt.tight_layout()
		if name is not None:
			plt.savefig(name + '.hold.out.pdf', dpi=300)

		# Save predictions on full data.
		all_data = pd.concat([train, test])
		print("Final Consensus Model:", final_model.get_params())
		print("Consensus Features:", consensus_features)
		all_data['Predicted Treatment Probabilities'] = final_model.predict_proba(
			all_data.drop(columns=['Treatment Response']).loc[:, consensus_features]
		)[:, 1]
		all_data['Predicted Treatment Response'] = final_model.predict(
			all_data.drop(columns=['Treatment Response']).loc[:, consensus_features]
		)

		if hold is not None:
			self.hold_data['Predicted Treatment Probabilities'] = final_model.predict_proba(
				hold.drop(columns=['Treatment Response']).loc[:, consensus_features]
			)[:, 1]
			self.hold_data['Predicted Treatment Response'] = final_model.predict(
				hold.drop(columns=['Treatment Response']).loc[:, consensus_features]
			)
		else:
			self.hold_data['Predicted Treatment Probabilities'] = final_model.predict_proba(
				self.hold_data.drop(columns=['Treatment Response']).loc[:, consensus_features]
			)[:, 1]
			self.hold_data['Predicted Treatment Response'] = final_model.predict(
				self.hold_data.drop(columns=['Treatment Response']).loc[:, consensus_features]
			)

		# Prepare dictionary of predictions per subgroup.
		screen_dictionary_preds = {}
		day0_dictionary_preds = {}
		adjwk1_dictionary_preds = {}
		for index, prediction in zip(list(self.screen_ids),
									all_data.loc[list(self.screen_ids)]['Predicted Treatment Response']):
			screen_dictionary_preds[index] = "Responder" if prediction else "Non-Responder"
		for index, prediction in zip(list(self.day0_ids),
									all_data.loc[list(self.day0_ids)]['Predicted Treatment Response']):
			day0_dictionary_preds[index] = "Responder" if prediction else "Non-Responder"
		for index, prediction in zip(list(self.adjwk1_ids),
									all_data.loc[list(self.adjwk1_ids)]['Predicted Treatment Response']):
			adjwk1_dictionary_preds[index] = "Responder" if prediction else "Non-Responder"

		preds_dictionary = [screen_dictionary_preds, day0_dictionary_preds, adjwk1_dictionary_preds]

		if name is not None:
			joblib.dump(final_model, name + ".model.pkl")
			with open(name + ".fold_predictions.pkl", "wb") as f:
				pickle.dump(fold_predictions, f)
		if name is not None:
			results_df = pd.DataFrame(results)
			results_df.to_csv(name + ".nested_cv_results.csv", index=False)
		return (results, final_model, consensus_features, preds_dictionary,
				all_data, self.hold_data[['Patient Number', 'Type of Visit', 'Treatment Response',
										'Predicted Treatment Probabilities', 'Predicted Treatment Response']])
		
	def survival_curve(self, response_dictionary, tumor_dictionary, name):
		if self.data is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
		if set(response_dictionary.keys()) != set(tumor_dictionary.keys()):
			missing_in_response = set(tumor_dictionary.keys()) - set(response_dictionary.keys())
			missing_in_tumor = set(response_dictionary.keys()) - set(tumor_dictionary.keys())
			
			raise ValueError(f"Key mismatch!\n"
							f"Missing in response_dictionary: {missing_in_response}\n"
							f"Missing in tumor_dictionary: {missing_in_tumor}")
		# Get survival and relapse data
		surv_time = self.get_metadata_col('Survival Months', df=pd.concat([self.data, self.hold_data], axis=0))
		surv_status = self.get_metadata_col('E_Survival', df=pd.concat([self.data, self.hold_data], axis=0))
		treatment_response = self.get_metadata_col('Stratification', df=pd.concat([self.data, self.hold_data], axis=0))
		diagnosis = self.get_metadata_col('Diagnosis', df=pd.concat([self.data, self.hold_data], axis=0))
		age = self.get_metadata_col('Age', df=pd.concat([self.data, self.hold_data], axis=0))
		gender = self.get_metadata_col('Gender', df=pd.concat([self.data, self.hold_data], axis=0))
		smoking = self.get_metadata_col('Smoking', df=pd.concat([self.data, self.hold_data], axis=0))
		alcohol = self.get_metadata_col('Alcohol', df=pd.concat([self.data, self.hold_data], axis=0))
		race = self.get_metadata_col('Race', df=pd.concat([self.data, self.hold_data], axis=0))
		ethnicity = self.get_metadata_col('Ethnicity', df=pd.concat([self.data, self.hold_data], axis=0))
		hpv = self.get_metadata_col('HPV', df=pd.concat([self.data, self.hold_data], axis=0))
		truth_response = self.get_metadata_col('Treatment Response', df=pd.concat([self.data, self.hold_data], axis=0))

		relapse_time = self.get_metadata_col('Relapse Months', df=pd.concat([self.data, self.hold_data], axis=0))
		relapse_status = self.get_metadata_col('E_Relapse', df=pd.concat([self.data, self.hold_data], axis=0))
		pdl1_ihc = self.get_metadata_col('PDL1 IHC', df=pd.concat([self.data, self.hold_data], axis=0))
		pdl1_ihc = pdl1_ihc.replace({0: "0", 1: "1-19", 2: ">20"})
		
		# Filter by class_dictionary
		surv_time = surv_time[surv_time.index.isin(response_dictionary.keys())]
		surv_status = surv_status[surv_status.index.isin(response_dictionary.keys())].astype(bool)
		treatment_response = treatment_response[treatment_response.index.isin(response_dictionary.keys())]
		relapse_time = relapse_time[relapse_time.index.isin(response_dictionary.keys())]
		relapse_status = relapse_status[relapse_status.index.isin(response_dictionary.keys())].astype(bool)
		pdl1_ihc = pdl1_ihc[pdl1_ihc.index.isin(response_dictionary.keys())]
		
		# Create DataFrame
		pairwise_results = {}
		merged_df = pd.concat([surv_time, surv_status, treatment_response, diagnosis, age, gender, smoking, alcohol, hpv, race, ethnicity, relapse_time, relapse_status, pdl1_ihc, truth_response], axis=1)
		merged_df.columns = ['Survival Months', 'E_Survival', 'Stratification', "Diagnosis", "Age", "Gender", "Smoking", "Alcohol", "HPV", "Race", "Ethnicity", 'Relapse Months', 'E_Relapse', 'PDL1 IHC', "Actual Treatment Response"]
		merged_df['E_Survival'] = merged_df['E_Survival'].astype(bool)
		merged_df['E_Relapse'] = merged_df['E_Relapse'].astype(bool)
		merged_df['Predicted Treatment Response'] = merged_df.index.map(response_dictionary)
		merged_df['Tumor Fraction'] = merged_df.index.map(tumor_dictionary)
		merged_df.dropna(inplace=True)
		
		# Plotting: Create subplots in a 2x3 layout
		fig, axes = plt.subplots(2, 3, figsize=(12, 8))
		
		#Set the colors for RISK STRATIFICATION.
		colors_strat = {"High": "firebrick", "Intermediate": "palegreen"}
		#Set the colors for TREATMENT RESPONSE.
		palette = sns.color_palette('colorblind')
		color0 = palette[0]
		color5 = palette[5]
		colors_resp = {"Responder": color0, "Non-Responder": color5}
		# Set the colors for PDL1 IHC.
		colors_pdl1 = {
			"0": "firebrick",
			"1-19": "gold",
			">20": "darkgreen"
		}
		# Set the colors for TUMOR FRACTION.
		colors_tumor = {
			"Low Tumor Fraction": "darkgreen",
			"High Tumor Fraction": "firebrick"
		}
		
		# Set up for two event types: Survival and Relapse
		for row, event_type in enumerate([("E_Relapse", "Relapse Months"), ("E_Survival", "Survival Months")]):
			for col, (group_col, colors, label) in enumerate([
				('Predicted Treatment Response', colors_resp, ''),
				('PDL1 IHC', colors_pdl1, ''),
				('Tumor Fraction', colors_tumor, ''),
			]):
				# Plot each group's curve
				for group in merged_df[group_col].unique():
					mask = merged_df[group_col] == group
					time, survival_prob, conf_int = kaplan_meier_estimator(
						merged_df[event_type[0]][mask],
						merged_df[event_type[1]][mask],
						conf_type="log-log",
					)
					axes[row, col].step(time, survival_prob, where="post", label=f"{label} = {group}",
										color=colors.get(group, "gray"), linewidth=1.5)
					# axes[row, col].fill_between(time, conf_int[0], conf_int[1], alpha=0.1, step="post",
					# 							color=colors.get(group, "gray"), edgecolor="none")
				
				# Create a pairwise matrix table of log-rank test results
				unique_groups = sorted(merged_df[group_col].unique())
				n = len(unique_groups)
				# Build an n x n matrix for results
				matrix_data = [[np.nan for _ in range(n)] for _ in range(n)]
				
				for i in range(n):
					for j in range(n):
						mask1 = merged_df[group_col] == unique_groups[i]
						mask2 = merged_df[group_col] == unique_groups[j]
						results = logrank_test(
							merged_df[event_type[1]][mask1],
							merged_df[event_type[1]][mask2],
							event_observed_A=merged_df[event_type[0]][mask1],
							event_observed_B=merged_df[event_type[0]][mask2]
						)
						p_value = results.p_value
						if p_value < 0.001:
							cell_text = f"{p_value:.4f}"
						elif p_value < 0.01:
							cell_text = f"{p_value:.3f}"
						elif p_value < 0.05:
							cell_text = f"{p_value:.2f}"
						else:
							cell_text = f"{p_value:.2f}"
						matrix_data[i][j] = float(cell_text)
				pairwise_results[(event_type[0], group_col)] = pd.DataFrame(
					matrix_data,
					index=unique_groups,
					columns=unique_groups
				)
				
				axes[row, col].set_ylim(0, 1.03)
				if col == 0:
					axes[row, col].set_ylabel("Estimated Probability of Event")
				else:
					axes[row, col].set_ylabel(None)
				axes[row, col].set_xlabel("Time in Months")
				
				# Adjust legend labels for clarity
				handles, labels_list = axes[row, col].get_legend_handles_labels()
				new_labels = [lbl.split('=')[1].strip() if '=' in lbl else lbl for lbl in labels_list]
				if len(new_labels) == 3:
					labels_list = new_labels
					new_labels = ["0", "1-19", ">20"]
					index_mapping = [labels_list.index(lab) if lab in labels_list else -1 for lab in new_labels]
					handles = [handles[i] for i in index_mapping]
				elif new_labels[0].startswith('N') or new_labels[0].startswith('R'):
					labels_list = new_labels
					new_labels = ["Responder", "Non-Responder"]
					index_mapping = [labels_list.index(lab) if lab in labels_list else -1 for lab in new_labels]
					handles = [handles[i] for i in index_mapping]
				else:
					labels_list = new_labels
					new_labels = ["Low Tumor Fraction", "High Tumor Fraction"]
					index_mapping = [labels_list.index(lab) if lab in labels_list else -1 for lab in new_labels]
					handles = [handles[i] for i in index_mapping]
				
				axes[row, col].legend(handles, new_labels, loc="lower left", frameon=False, fontsize=8)
		
		# Titles for each subplot
		axes[1, 0].set_title(f"Overall Survival (Predicted Response)", fontsize=10)
		axes[1, 1].set_title("Overall Survival (PDL1 IHC)", fontsize=10)
		axes[1, 2].set_title("Overall Survival (Tumor Fraction)", fontsize=10)
		axes[0, 0].set_title(f"Disease-Free Survival (Predicted Response)", fontsize=10)
		axes[0, 1].set_title("Disease-Free Survival (PDL1 IHC)", fontsize=10)
		axes[0, 2].set_title(f"Disease-Free Survival (Tumor Fraction)", fontsize=10)
		
		plt.tight_layout()
		plt.savefig(f"{name}.pdf", dpi=300)
		plt.show()
		return merged_df