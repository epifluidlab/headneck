library(arsenal)
library(readr)
library(dplyr)

file_path <- '/jet/home/rbandaru/ravi/headneck/metadata/RAW_HNSCC_METADATA.csv'
df <- read_csv(file_path)

# Select and clean the columns
selected_columns <- c("Patient Number", "Diagnosis", "Age", "Gender", "Smoking", "Alcohol", "Stratification", "Treatment Response")
df_selected <- df[selected_columns]
df_selected <- distinct(df_selected)
selected_columns <- c("Diagnosis", "Age", "Gender", "Smoking", "Alcohol", "Stratification", "Treatment Response")
df_selected <- df_selected[selected_columns]
df_selected[is.na(df_selected)] <- "Missing"

# Reorder the levels of 'Treatment Response'
df_selected$`Treatment Response` <- factor(df_selected$`Treatment Response`, levels = c("Responder", "Non-Responder", "Missing"))

# Create the summary table
summary_table <- tableby(`Treatment Response` ~ ., data = df_selected)

sink("/jet/home/rbandaru/ravi/headneck/hnscc_metadata_summary.md")
print(summary(summary_table))
sink()
