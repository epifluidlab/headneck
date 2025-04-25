library(arsenal)
library(readr)
library(dplyr)

file_path <- '../../Supplementary_Tables/ST1/RAW_HNSCC_METADATA_NEW.csv'
df <- read_csv(file_path)

selected_columns <- c("Patient Number", "Diagnosis", "Age", "Gender", "Smoking", "Alcohol", "Stratification", "Treatment Response")
df_selected <- df[selected_columns]
df_selected <- distinct(df_selected)
selected_columns <- c("Diagnosis", "Age", "Gender", "Smoking", "Alcohol", "Stratification", "Treatment Response")
df_selected <- df_selected[selected_columns]

df_selected$`Treatment Response` <- factor(df_selected$`Treatment Response`, levels = c("Responder", "Non-Responder", "Missing"))

summary_table <- tableby(`Treatment Response` ~ ., data = df_selected)

sink("./hnscc_metadata_summary.md")
print(summary(summary_table))
sink()
