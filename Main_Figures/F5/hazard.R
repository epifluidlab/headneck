# Load packages
library(dplyr)
library(survival)
library(gtsummary)
library(gt)
library(webshot)
library(rmarkdown)

# Set working directory
setwd('/Users/ravibandaru/Downloads')

# Load data
d <- read.csv("./survival_data.csv")

# Construct patient-level dataframe
df_patient <- data.frame(
  id            = d[, 1],
  age           = d$Age,
  gender        = factor(ifelse(d$Gender == "Male", 1, 2), labels = c("Male", "Female")),
  ethnicity     = factor(as.integer(as.factor(d$Ethnicity))),
  diagnosis     = factor(as.integer(as.factor(d$Diagnosis))),
  hpv           = factor(ifelse(d$HPV == "Unknown", 1, 2), labels = c("Unknown", "Known")),
  smoking       = factor(ifelse(d$Smoking == "No", 1, 2), labels = c("No", "Yes")),
  alcohol       = factor(ifelse(d$Alcohol == "No", 1, 2), labels = c("No", "Yes")),
  response      = factor(ifelse(d$REPINST_Predictions == "Responder", 1, 2), labels = c("Responder", "NonResponder")),
  stratification= factor(ifelse(d$Stratification == "Intermediate", 1, 2), labels = c("Intermediate", "High")),
  ihc           = factor(ifelse(d$PDL1.IHC == ">20", 1, ifelse(d$PDL1.IHC == "1-19", 2, 3)), labels = c(">20", "1-19", "0")),
  tf            = factor(ifelse(d$Tumor.Fraction == "Low Tumor Fraction", 1, 2), labels = c("Low", "High")),
  event_dfs     = as.integer(ifelse(d$E_Relapse, 1, 0)),
  event_os      = as.integer(ifelse(d$E_Survival, 1, 0)),
  relapse_time  = as.integer(d$`Relapse.Months..Adjuvant.` * 30),
  surv_time     = as.integer(d$Survival.Months..Adjuvant. * 30),
  repinst_prob  = as.numeric(d$REPINST_Probabilities),
  repinst_pred  = as.factor(d$REPINST_Predictions)
)

# Variables to analyze
vars <- c("age", "gender", "ethnicity", "diagnosis", "hpv", "smoking",
          "alcohol", "response", "stratification", "ihc", "tf")

# ---------- Univariate Cox: DFS ----------
uv_tbl_dfs <- df_patient %>%
  select(all_of(vars), relapse_time, event_dfs) %>%
  tbl_uvregression(
    method = coxph,
    y = Surv(relapse_time, event_dfs),
    exponentiate = TRUE
  )

gtsave(as_gt(uv_tbl_dfs), "cox_univariable_dfs.html")

# ---------- Univariate Cox: OS ----------
uv_tbl_os <- df_patient %>%
  select(all_of(vars), surv_time, event_os) %>%
  tbl_uvregression(
    method = coxph,
    y = Surv(surv_time, event_os),
    exponentiate = TRUE
  )

gtsave(as_gt(uv_tbl_os), "cox_univariable_os.html")

# ---------- Proportional Hazards Tests ----------
run_zph_tests <- function(time_col, event_col, vars, data, filename, model_label) {
  sink(filename)
  for (v in vars) {
    formula <- as.formula(paste0("Surv(", time_col, ", ", event_col, ") ~ ", v))
    model <- coxph(formula, data = data)
    zph <- cox.zph(model)
    
    cat("\n==============================\n")
    cat("Variable:", v, "\n")
    print(zph)
    cat("==============================\n")
  }
  sink()
}

run_zph_tests("relapse_time", "event_dfs", vars, df_patient, "cox_zph_univariable_dfs.txt", "DFS Univariable")
run_zph_tests("surv_time", "event_os", vars, df_patient, "cox_zph_univariable_os.txt", "OS Univariable")

# ---------- Multivariable Models ----------
run_multivariable_model <- function(time_col, event_col, formula_str, data, filename, model_desc) {
  sink(filename, append = TRUE)
  cat("\n==============================\n")
  cat("Model:", model_desc, "\n")
  model <- coxph(as.formula(paste0("Surv(", time_col, ", ", event_col, ") ~ ", formula_str)), data = data)
  zph <- cox.zph(model)
  print(zph)
  cat("==============================\n")
  sink()
}

# DFS models
run_multivariable_model("relapse_time", "event_dfs", "response + stratification", df_patient, "cox_zph_multivariable_dfs.txt", "response + stratification")
run_multivariable_model("relapse_time", "event_dfs", "response * stratification", df_patient, "cox_zph_multivariable_dfs.txt", "response * stratification")

# OS models
run_multivariable_model("surv_time", "event_os", "response + stratification", df_patient, "cox_zph_multivariable_os.txt", "response + stratification")
run_multivariable_model("surv_time", "event_os", "response * stratification", df_patient, "cox_zph_multivariable_os.txt", "response * stratification")
