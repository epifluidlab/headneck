library(dplyr)
library(survival)
library(survminer)
library(ggplot2)

setwd('/Users/myt8099/Downloads')
d <- read.csv("./survival_data.csv")

df_patient_DFS <- data.frame(
  id = d[,1],
  age = d$Age,
  gender = ifelse(d$Gender == "Male", 1, 2),  # Male = 1 (reference), Female = 2
  ethnicity = as.integer(as.factor(d$Ethnicity)),
  diagnosis = as.integer(as.factor(d$Diagnosis)),
  hpv = ifelse(d$HPV == "Unknown", 1, 2),  # Unknown = 1 (reference), No = 2
  smoking = ifelse(d$Smoking == "No", 1, 2),  # No = 1 (reference), Yes = 2
  alcohol = ifelse(d$Alcohol == "No", 1, 2),  # No = 1 (reference), Yes = 2
  stratification = ifelse(d$Stratification == "Intermediate", 1, 2),
  response = ifelse(d$Predicted.Treatment.Response == "Responder", 1, 2),  # Responder = 1 (reference), Non-Responder = 2
  actual_response = ifelse(d$Actual.Treatment.Response == "Responder", 1, 2),  # Responder = 1 (reference), Non-Responder = 2
  ihc = ifelse(d$PDL1.IHC == ">20", 1, ifelse(d$PDL1.IHC == "1-19", 2, 3)),  # >20 = 1 (reference), 1-19 = 2, 0 = 3
  tf = ifelse(d$Tumor.Fraction == "Low Tumor Fraction", 1, 2),  # Low = 1, High = 2
  event = as.integer(ifelse(d$E_Relapse, 1, 0)),
  time_to_event = as.integer(d$Relapse.Months * 30)
)

cox_patient <- coxph(
  Surv(time_to_event, event) ~ age + gender + ethnicity + diagnosis + hpv + smoking + alcohol + stratification + response + tt(ihc) + tf,
  tt = function(x, t, ...) x * log(t),
  data = df_patient_DFS
)

output_file <- "DFS_NO_TIME_EFFECT.txt"
sink(output_file)
cat('DFS_NO_TIME_EFFECT\n\n')
print(summary(cox_patient))
sink()

summary_cox <- summary(cox_patient)
df_forest <- data.frame(
  variable = rownames(summary_cox$coefficients),
  coef = summary_cox$coefficients[, "coef"],
  HR = exp(summary_cox$coefficients[, "coef"]),
  lower_CI = exp(summary_cox$coefficients[, "coef"] - 1.96 * summary_cox$coefficients[, "se(coef)"]),
  upper_CI = exp(summary_cox$coefficients[, "coef"] + 1.96 * summary_cox$coefficients[, "se(coef)"]),
  p_value = summary_cox$coefficients[, "Pr(>|z|)"]
)
df_forest$variable <- gsub("response", "Predicted Non-Responder", df_forest$variable)
df_forest$variable <- gsub("gender", "Female", df_forest$variable)
df_forest$variable <- gsub("ethnicity", "Ethnicity", df_forest$variable)
df_forest$variable <- gsub("diagnosis", "Diagnosis Site", df_forest$variable)
df_forest$variable <- gsub("hpv", "No HPV", df_forest$variable)
df_forest$variable <- gsub("smoking", "Smoking", df_forest$variable)
df_forest$variable <- gsub("alcohol", "Alcohol", df_forest$variable)
df_forest$variable <- gsub("stratification", "High Risk Stratification", df_forest$variable)
df_forest$variable <- gsub("age", "Age", df_forest$variable)
df_forest$variable <- gsub("tf", "High Tumor Fraction", df_forest$variable)
df_forest$variable <- gsub("ihc", "Low PD-L1 IHC", df_forest$variable)

df_forest$variable <- sprintf("%s (p=%.3f)", df_forest$variable, df_forest$p_value)

p <- ggplot(df_forest, aes(x = reorder(variable, HR), y = HR)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower_CI, ymax = upper_CI), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  coord_flip() +
  scale_y_log10() +
  labs(title = "Hazard Ratios (Cox Model: Predicted Response)", x = "", y = "Hazard Ratio (log scale, 95% CI)") +
  theme_minimal(base_size = 12)

ggsave("DFS_Response_ForestPlot.pdf", plot = p, width = 8, height = 5, dpi = 300)

sink("DFS_NO_TIME_EFFECT_RESPONSE.txt")
cat('DFS_NO_TIME_EFFECT\n')
cox_patient <- coxph(
  Surv(time_to_event, event) ~ age + gender + ethnicity + diagnosis + hpv + smoking + alcohol + stratification + response,
  data = df_patient_DFS
)
print(summary(cox_patient))
zph_patient <- cox.zph(cox_patient)
print(zph_patient)
sink()  # close


sink("DFS_NO_TIME_EFFECT_IHC.txt")
cat('DFS_NO_TIME_EFFECT\n')
cox_patient <- coxph(
  Surv(time_to_event, event) ~ age + gender + ethnicity + diagnosis + hpv + smoking + alcohol + stratification + tt(ihc),
  tt = function(x, t, ...) x * log(t),
  data = df_patient_DFS
)
print(summary(cox_patient))
sink()  # close

sink("DFS_NO_TIME_EFFECT_TF.txt")
cat('DFS_NO_TIME_EFFECT\n')
cox_patient <- coxph(
  Surv(time_to_event, event) ~ age + gender + ethnicity + diagnosis + hpv + smoking + alcohol + stratification + tf,
  data = df_patient_DFS
)
print(summary(cox_patient))
zph_patient <- cox.zph(cox_patient)
print(zph_patient)
sink()  # close

surv_obj <- Surv(df_patient_DFS$time_to_event, df_patient_DFS$event)

km_fit <- survfit(surv_obj ~ response, data = df_patient_DFS)

km_plot <- ggsurvplot(
  km_fit,
  data = df_patient_DFS,
  legend.labs = c("Responder", "Non-Responder"),
  palette = c("#0173B2", "#CA9161"),
  title = "Disease-Free Survival\n(Predicted Response)",
  ylab = "Probability of Event",
  xlab = "Time (Months)"
)
km_plot$plot <- km_plot$plot + theme(plot.title = element_text(hjust = 0.5))
ggsave("DFS_Response.pdf", km_plot$plot, width = 5, height = 5, dpi = 300)


km_fit <- survfit(surv_obj ~ ihc, data = df_patient_DFS)

km_plot <- ggsurvplot(
  km_fit,
  data = df_patient_DFS,
  legend.labs = c(">20", "1-19", "0"),
  palette = c("#15B01A", "#FFD700", "#DC143C"),
  title = "Disease-Free Survival\n(PD-L1 IHC)",
  ylab = "Probability of Event",
  xlab = "Time (Months)"
)
km_plot$plot <- km_plot$plot + theme(plot.title = element_text(hjust = 0.5))
ggsave("DFS_IHC.pdf", km_plot$plot, width = 5, height = 5, dpi = 300)


km_fit <- survfit(surv_obj ~ tf, data = df_patient_DFS)

km_plot <- ggsurvplot(
  km_fit,
  data = df_patient_DFS,
  legend.labs = c("Low Tumor Fraction", "High Tumor Fraction"),
  palette = c("#15B01A", "#DC143C"),
  title = "Disease-Free Survival\n(Tumor Fraction)",
  ylab = "Probability of Event",
  xlab = "Time (Months)"
)

km_plot$plot <- km_plot$plot + theme(plot.title = element_text(hjust = 0.5))
ggsave("DFS_Tumor_Fraction.pdf", km_plot$plot, width = 5, height = 5, dpi = 300)


