import os
import glob
import patsy
import numpy as np
import pandas as pd
import seaborn as sns
from patsy.contrasts import Sum
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from lifelines.statistics import logrank_test
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix

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
		def limma(exprs, covariate_matrix, design_matrix, rcond=1e-8):
			design_batch = np.hstack((covariate_matrix, design_matrix))
			coefficients, _, _, _ = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)
			beta = coefficients[-design_matrix.shape[1]:]
			return exprs - design_matrix.dot(beta).T
		def get_dmatrix(categories):
			contrast = Sum()
			levels = list(categories.categories)
			contrast_matrix = contrast.code_without_intercept(levels).matrix
			if len(levels) == len(contrast_matrix):
				contrast_dict = {level: contrast_matrix[i] for i, level in enumerate(levels)}
			else:
				print("Error: The lengths of levels and contrast_matrix do not match.")
			
			dmatrix = []
			for element in categories:
				dmatrix.append(contrast_dict[element])
			return np.array(dmatrix)
		if self.data is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
		raw_data = self.get_raw_features()
		metadata_cols = self.data.drop(columns=raw_data.columns)
		raw_data = raw_data.T
		batch1 = pd.Categorical(self.get_metadata_col('Institute'))
		#batch2 = pd.Categorical(self.get_metadata_col('Pilot'))
		batch3 = pd.Categorical(self.get_metadata_col('WGS Library Prep Date'))
		batch4 = pd.Categorical(self.get_metadata_col('cfDNA Isolation Date'))
		batch_df = pd.DataFrame({
			'Institute': batch1.codes,
			#'Pilot': batch2.codes,
			'Library_Prep_Date': batch3.codes,
			'cfDNA_Isolation_Date': batch4.codes,
			'Treatment_Response': pd.Categorical(self.get_metadata_col('Treatment Response')).codes
		})
		print(batch_df.corr())

		contrast_key = {
			'Institute': get_dmatrix(batch1),
			#'Pilot': get_dmatrix(batch2),
			'Library_Prep_Date': get_dmatrix(batch3),
			'cfDNA_Isolation_Date': get_dmatrix(batch4),
		}
		design = np.concatenate([contrast_key[key] for key in contrast_key], axis=1)
		covariates = get_dmatrix(pd.Categorical(self.get_metadata_col('Treatment Response')))
		corrected_data = limma(raw_data, covariates, design)
		self.data = pd.concat([metadata_cols, corrected_data.T], axis=1)
		return self.data

	def filter_locations(self, locations_file: str):
		locations = pd.read_csv(locations_file, delimiter='\t', header=None, names=['chr', 'start', 'end'])
		location_strings = set(locations.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1))
		matching_columns = [col for col in self.data.columns if col in location_strings or not col.startswith('chr')]
		return self.data[matching_columns]

	def pca(self, label: str, scale=True, save_path: str = None, plot_ellipses=False):
		if self.data is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
		raw_data = self.get_raw_features()
		pca = PCA(n_components=2)
		scaler = StandardScaler() if scale==True else FunctionTransformer(lambda x: x)
		X =  pd.DataFrame(scaler.fit_transform(raw_data), index=raw_data.index, columns=raw_data.columns)
		pca_data = pca.fit_transform(X)
		pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
		pca_df.index = X.index
		pca_df[f'{label}'] = self.get_metadata_col(label)
		unique_labels = pca_df[label].unique()
		colors = sns.color_palette("tab10", len(unique_labels))
		color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
		plt.figure(figsize=(6, 3))
		for label_l, color in color_map.items():
			subset = pca_df[pca_df[label] == label_l]
			plt.scatter(subset['PC1'], subset['PC2'], c=color, label=label_l, s=2)
			if plot_ellipses:
				mean = subset[['PC1', 'PC2']].mean()
				cov = subset[['PC1', 'PC2']].cov()
				lambda_, v = np.linalg.eig(cov)
				lambda_ = np.sqrt(lambda_)
				ellipse = plt.matplotlib.patches.Ellipse(xy=mean, width=lambda_[0]*2, height=lambda_[1]*2, angle=np.rad2deg(np.arccos(v[0, 0])), edgecolor=color, facecolor=color, alpha=0.2, linewidth=1)
				plt.gca().add_patch(ellipse)
		plt.xlabel('PC1')
		plt.ylabel('PC2')
		plt.ylim(-25, 25)
		plt.legend(title=label, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['left'].set_visible(True)
		plt.gca().spines['bottom'].set_visible(True)
		plt.tight_layout()
		if save_path is not None:
			plt.savefig(save_path, dpi=1000)
		plt.show()

	def survival_curve(self, class_dictionary):
		if self.data is None:
			raise ValueError("No data loaded yet. Load data using the merge_feature_metadata method.")
		surv_time = self.get_metadata_col('Survival Months')
		surv_status = self.get_metadata_col('E_Survival')
		treatment_response = self.get_metadata_col('Treatment Response')
		relapse_time = self.get_metadata_col('Relapse Months')
		relapse_status = self.get_metadata_col('E_Relapse')

		# Drop all rows from surv_time, surv_status, relapse_time, and relapse_status that dont have a corresponding index in the key of class_dictionary.
		surv_time = surv_time[surv_time.index.isin(class_dictionary.keys())]
		surv_status = surv_status[surv_status.index.isin(class_dictionary.keys())]
		surv_status = surv_status.astype(bool)
		treatment_response = treatment_response[treatment_response.index.isin(class_dictionary.keys())]
		relapse_time = relapse_time[relapse_time.index.isin(class_dictionary.keys())]
		relapse_status = relapse_status[relapse_status.index.isin(class_dictionary.keys())]
		relapse_status = relapse_status.astype(bool)

		merged_df = pd.concat([surv_time, surv_status, treatment_response, relapse_time, relapse_status], axis=1)
		merged_df.columns = ['Survival Months', 'E_Survival', 'Actual Treatment Response', 'Relapse Months', 'E_Relapse']
		merged_df['Predicted Treatment Response'] = merged_df.index.map(class_dictionary)
		merged_df.dropna(inplace=True)
		data = merged_df.copy()
		fig, axes = plt.subplots(2, 2, figsize=(10, 10))
		colors = {
			"Responder": "green",
			"Non-Responder": "red"
		}
		for i, response_type in enumerate(["Actual Treatment Response", "Predicted Treatment Response"]):
			for j, event_type in enumerate([("E_Survival", "Survival Months"), ("E_Relapse", "Relapse Months")]):
				# Plot each treatment type
				for treatment_type in ("Responder", "Non-Responder"):
					mask_treat = data[response_type] == treatment_type
					time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
						data[event_type[0]][mask_treat],
						data[event_type[1]][mask_treat],
						conf_type="log-log",
					)
					axes[j, i].step(time_treatment, survival_prob_treatment, where="post",
									label=f"Treatment = {treatment_type}", color=colors[treatment_type])
					axes[j, i].fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25,
										step="post", color=colors[treatment_type])
				
				# Calculate p-value using log-rank test
				mask_responder = data[response_type] == "Responder"
				mask_non_responder = data[response_type] == "Non-Responder"
				
				results = logrank_test(
					data[event_type[1]][mask_responder],
					data[event_type[1]][mask_non_responder],
					event_observed_A=data[event_type[0]][mask_responder],
					event_observed_B=data[event_type[0]][mask_non_responder]
				)
				
				p_value = results.p_value
				
				# Add p-value text to the plot
				p_text = f"p = {p_value:.4f}"
				if p_value < 0.001:
					p_text = "p < 0.001 ***"
				elif p_value < 0.01:
					p_text = f"p = {p_value:.3f} **"
				elif p_value < 0.05:
					p_text = f"p = {p_value:.3f} *"
				else:
					p_text = f"p = {p_value:.3f} (ns)"
					
				axes[j, i].text(0.95, 0.95, p_text, transform=axes[j, i].transAxes,
							horizontalalignment='right', verticalalignment='top',
							bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
				
				axes[j, i].set_ylim(0, 1.03)
				axes[j, i].set_ylabel(r"Estimated Probability of Event $\hat{E}(t)$")
				axes[j, i].set_xlabel("Time in Months")
				axes[j, i].legend(loc="lower left")

		axes[0, 0].set_title("Actual - Survival")
		axes[0, 1].set_title("Predicted - Survival")
		axes[1, 0].set_title("Actual - Relapse")
		axes[1, 1].set_title("Predicted - Relapse")
		plt.tight_layout()
		plt.show()

	def train_test_model(self, model, hyperparameter_dict, train, test, cv_split=10, k=10):
		def load_split(split_no: int):
			def to_list(filepath):
				with open(filepath, "r") as f:
					return [line.strip() for line in f]
			train_split = to_list(f"/projects/b1198/epifluidlab/ravi/0130/headneck/notebooks/lists/train_fold_{split_no}.txt")
			val_split = to_list(f"/projects/b1198/epifluidlab/ravi/0130/headneck/notebooks/lists/val_fold_{split_no}.txt")
			return train_split, val_split

		results = []
		best_overall_model = None
		best_overall_score = -np.inf
		best_overall_params = None
		feature_selection_counts = Counter()

		plt.figure(figsize=(8, 16), dpi=300)

		for split_no in range(cv_split):
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
		plt.show()
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
		return results, best_overall_model, top_features, preds_dictionary, all_data



