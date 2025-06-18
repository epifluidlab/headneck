library(dplyr)
library(survival)
library(survminer)
library(ggplot2)
library(gridExtra)
library(emmeans)
library(gtsummary)
library(gt)

setwd('/Users/ravibandaru/Downloads')
d <- read.csv("./survival_data.csv")

# Format patient data
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

### --- DFS Cox Model --- ###
cox_dfs <- coxph(Surv(relapse_time, event_dfs) ~ response + stratification + response:stratification, data = df_patient)

emm_dfs_response <- emmeans(cox_dfs, ~ response)
emm_dfs_strat    <- emmeans(cox_dfs, ~ stratification)
emm_dfs_inter    <- emmeans(cox_dfs, ~ response * stratification)

### --- OS Cox Model --- ###
cox_os <- coxph(Surv(surv_time, event_os) ~ response + stratification + response:stratification, data = df_patient)

emm_os_response <- emmeans(cox_os, ~ response)
emm_os_strat    <- emmeans(cox_os, ~ stratification)
emm_os_inter    <- emmeans(cox_os, ~ response * stratification)

### --- Save Results to Text Files --- ###

## DFS Results
sink("DFS_emmeans_results.txt")
cat("=== Disease-Free Survival EMMEANS ===\n\n")

cat(">>> Response:\n")
print(pairs(emm_dfs_response))

cat("\n>>> Stratification:\n")
print(pairs(emm_dfs_strat))

cat("\n>>> Response * Stratification:\n")
print(pairs(emm_dfs_inter))
sink()

## OS Results
sink("OS_emmeans_results.txt")
cat("=== Overall Survival EMMEANS ===\n\n")

cat(">>> Response:\n")
print(pairs(emm_os_response))

cat("\n>>> Stratification:\n")
print(pairs(emm_os_strat))

cat("\n>>> Response * Stratification:\n")
print(pairs(emm_os_inter))
sink()
