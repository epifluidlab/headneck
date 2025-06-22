library(dplyr)
library(tidyr)
library(survival)
library(ggplot2)
library(gtsummary)
library(gt)
library(webshot)
library(rmarkdown)
library(emmeans)
library(broom)
library(stringr)

setwd('/Users/ravibandaru/Downloads')
d <- read.csv("./SURVIVAL_DATA.csv")

df_patient <- data.frame(
  id            = d[, 1],
  age           = d$Age,
  gender        = factor(ifelse(d$Gender == "Male", 1, 2), labels = c("Male", "Female")),
  ethnicity     = factor(as.integer(as.factor(d$Ethnicity))),
  diagnosis     = factor(as.integer(as.factor(d$Diagnosis))),
  hpv           = factor(ifelse(d$HPV == "Unknown", 2, 1), labels = c("No", "Unknown")),
  smoking       = factor(ifelse(d$Smoking == "No", 1, 2), labels = c("No", "Yes")),
  alcohol       = factor(ifelse(d$Alcohol == "No", 1, 2), labels = c("No", "Yes")),
  stratification= factor(ifelse(d$Stratification == "Intermediate", 1, 2), labels = c("Intermediate", "High")),
  response      = factor(ifelse(d$REPINST_Predictions == "Responder", 1, 2), labels = c("Responder", "Non-Responder")),
  ihc           = factor(ifelse(d$PDL1.IHC == ">20", 1, ifelse(d$PDL1.IHC == "1-19", 2, 3)), labels = c(">20", "1-19", "0")),
  tf            = factor(ifelse(d$Tumor.Fraction == "Low Tumor Fraction", 1, 2), labels = c("Low", "High")),
  
  event_dfs_new = as.integer(ifelse(d$E_Relapse, 1, 0)),
  time_dfs_new  = as.integer(d$Relapse.Months..Adjuvant. * 30.5),
  event_os_new  = as.integer(ifelse(d$E_Survival, 1, 0)),
  time_os_new   = as.integer(d$Survival.Months..Adjuvant. * 30.5),
  
  event_dfs_old = as.integer(ifelse(d$OLD.E_Relapse, 1, 0)),
  time_dfs_old  = as.integer(d$OLD.Relapse.Months..Adjuvant. * 30.5),
  event_os_old  = as.integer(ifelse(d$OLD.E_Survival, 1, 0)),
  time_os_old   = as.integer(d$OLD.Survival.Months..Adjuvant. * 30.5)
)

vars <- c("age", "gender", "hpv", "smoking",
          "alcohol", "response", "stratification", "ihc", "tf")

event_time_pairs <- list(
  "dfs_new" = list(event = "event_dfs_new", time = "time_dfs_new"),
  "os_new"  = list(event = "event_os_new", time = "time_os_new"),
  "dfs_old" = list(event = "event_dfs_old", time = "time_dfs_old"),
  "os_old"  = list(event = "event_os_old", time = "time_os_old")
)

perform_cox_analysis <- function(df, event_col, time_col, vars) {
  
  df_work <- df %>%
    mutate(
      event_current = .data[[event_col]],
      time_current = .data[[time_col]]
    )
  
  uv_tbl <- df_work %>%
    select(all_of(vars), event_current, time_current) %>%
    tbl_uvregression(
      method = coxph,
      y = Surv(time_current, event_current),
      exponentiate = TRUE
    )
  
  uv_df <- as_tibble(uv_tbl)
  
  event <- df[[event_col]]
  time <- df[[time_col]]
  
  # Cox odel with interaction
  cox_model <- coxph(Surv(time, event) ~ response:stratification, data = df)
  
  emm_response <- emmeans(cox_model, ~ response)
  emm_strat    <- emmeans(cox_model, ~ stratification)
  emm_inter    <- emmeans(cox_model, ~ response * stratification)
  
  hr_response <- contrast(emm_response, method = "pairwise") %>%
    summary(infer = TRUE) %>%
    as.data.frame() %>%
    transmute(
      contrast = contrast,
      HR       = exp(estimate),
      HR_lower = exp(asymp.UCL),
      HR_upper = exp(asymp.LCL),
      p_value  = p.value
    )

  hr_stratification <- contrast(emm_strat, method = "pairwise") %>%
    summary(infer = TRUE) %>%
    as.data.frame() %>%
    transmute(
      contrast = contrast,
      HR       = exp(estimate),
      HR_lower = exp(asymp.UCL),
      HR_upper = exp(asymp.LCL),
      p_value  = p.value
    )
  
  hr_response_stratification <- contrast(emm_inter, method = "pairwise") %>%
    summary(infer = TRUE) %>%
    as.data.frame() %>%
    transmute(
      contrast = contrast,
      HR       = exp(estimate),
      HR_lower = exp(asymp.UCL),
      HR_upper = exp(asymp.LCL),
      p_value  = p.value
    )
  
  uv_tidy <- uv_df %>%
    rename(
      contrast = '**Characteristic**',
      HR = '**HR**',
      ci = '**95% CI**',
      p_value = '**p-value**'
    ) %>%
    mutate(
      HR_lower = as.numeric(str_trim(str_extract(ci, "^[0-9.]+"))),
      HR_upper = as.numeric(str_trim(str_extract(ci, "(?<=,)\\s*[0-9.]+"))),
      HR = as.numeric(HR),
      p_value = as.numeric(str_replace(p_value, ">", ""))
    ) %>%
    select(contrast, HR, HR_lower, HR_upper, p_value) %>%
    mutate(source = "univariate")
  
  hr_response <- hr_response %>%
    mutate(source = "response") %>%
    select(contrast, HR, HR_lower, HR_upper, p_value, source)
  
  hr_stratification <- hr_stratification %>%
    mutate(source = "stratification") %>%
    select(contrast, HR, HR_lower, HR_upper, p_value, source)
  
  hr_response_stratification <- hr_response_stratification %>%
    mutate(source = "response_stratification") %>%
    select(contrast, HR, HR_lower, HR_upper, p_value, source)
  
  combined_df <- bind_rows(uv_tidy, hr_response, hr_stratification, hr_response_stratification)
  
  return(combined_df)
}

for (pair_name in names(event_time_pairs)) {
  
  cat("Processing:", pair_name, "\n")
  
  event_col <- event_time_pairs[[pair_name]]$event
  time_col <- event_time_pairs[[pair_name]]$time
  
  results <- perform_cox_analysis(df_patient, event_col, time_col, vars)
  
  filename <- paste0("hazard_", pair_name, ".csv")
  write.csv(results, file = filename, row.names = FALSE)
  
  cat("Results saved to:", filename, "\n\n")
}
