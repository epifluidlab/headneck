library(dplyr)
library(survival)
library(broom)
library(ggplot2)
library(forcats)
library(stringr)

setwd('/Users/ravibandaru/Downloads')

d <- read.csv("./survival_data.csv")

df_patient <- data.frame(
  id = d[, 1],
  age = d$Age,
  gender = factor(ifelse(d$Gender == "Male", 1, 2), labels = c("Male", "Female")),
  ethnicity = factor(as.integer(as.factor(d$Ethnicity))),
  diagnosis = factor(as.integer(as.factor(d$Diagnosis))),
  hpv = factor(ifelse(d$HPV == "Unknown", 1, 2), labels = c("Unknown", "Known")),
  smoking = factor(ifelse(d$Smoking == "No", 1, 2), labels = c("No", "Yes")),
  alcohol = factor(ifelse(d$Alcohol == "No", 1, 2), labels = c("No", "Yes")),
  response = factor(ifelse(d$REPINST_Predictions == "Responder", 1, 2), labels = c("Responder", "NonResponder")),
  stratification = factor(ifelse(d$Stratification == "Intermediate", 1, 2), labels = c("Intermediate", "High")),
  ihc = factor(ifelse(d$PDL1.IHC == ">20", 1, ifelse(d$PDL1.IHC == "1-19", 2, 3)), labels = c(">20", "1-19", "0")),
  tf = factor(ifelse(d$Tumor.Fraction == "Low Tumor Fraction", 1, 2), labels = c("Low", "High")),
  event_dfs = as.integer(ifelse(d$E_Relapse, 1, 0)),
  event_os = as.integer(ifelse(d$E_Survival, 1, 0)),
  relapse_time = as.integer(d$`Relapse.Months..Adjuvant.` * 30),
  surv_time = as.integer(d$Survival.Months..Adjuvant. * 30),
  repinst_prob = as.numeric(d$REPINST_Probabilities),
  repinst_pred = as.factor(d$REPINST_Predictions)
)

cox_dfs <- coxph(Surv(relapse_time, event_dfs) ~ age + gender + hpv + smoking + alcohol + response + stratification + ihc + tf, data = df_patient)
cox_os <- coxph(Surv(surv_time, event_os) ~ age + gender + hpv + smoking + alcohol + response + stratification + ihc + tf, data = df_patient)

make_publication_forest_plot <- function(tidy_df, title_text) {
  var_labels <- c(
    "age" = "Age (continuous)",
    "genderFemale" = "Gender: Female (ref: Male)",
    "hpvKnown" = "HPV: Known (ref: Unknown)",
    "smokingYes" = "Smoking: Yes (ref: No)",
    "alcoholYes" = "Alcohol: Yes (ref: No)",
    "responseNonResponder" = "Predicted Response: Non-Responder (ref: Responder)",
    "stratificationHigh" = "Risk Stratification: High (ref: Intermediate)",
    "ihc1-19" = "PD-L1 IHC: 1-19% (ref: >20%)",
    "ihc0" = "PD-L1 IHC: 0% (ref: >20%)",
    "tfHigh" = "Tumor Fraction: High (ref: Low)"
  )
  
  plot_data <- tidy_df %>%
    filter(term != "(Intercept)") %>%
    mutate(
      clean_term = case_when(
        term %in% names(var_labels) ~ var_labels[term],
        TRUE ~ term
      ),
      p_formatted = case_when(
        p.value < 0.001 ~ "p < 0.001",
        p.value < 0.01 ~ sprintf("p = %.3f", p.value),
        TRUE ~ sprintf("p = %.2f", p.value)
      ),
      label_with_p = paste0(clean_term, "\n", p_formatted),
      label_with_p = fct_reorder(label_with_p, estimate)
    )
  
  p <- ggplot(plot_data, aes(x = estimate, y = label_with_p)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50", size = 0.8) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.3, color = "black", size = 0.6) +
    geom_point(color = "#2166ac", size = 3.5, shape = 18) +
    scale_x_continuous(
      trans = "log10",
      breaks = c(0.1, 0.2, 0.5, 1, 2, 5, 10),
      labels = c("0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0"),
      limits = c(0.1, max(plot_data$conf.high) * 1.5)
    ) +
    labs(
      title = title_text,
      subtitle = "Hazard Ratios with 95% Confidence Intervals",
      x = "Hazard Ratio (log scale)",
      y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "gray90", size = 0.5, linetype = "dotted"),
      panel.border = element_rect(color = "black", fill = NA, size = 0.5),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold", margin = margin(b = 5)),
      plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray40", margin = margin(b = 15)),
      axis.title.x = element_text(size = 11, face = "bold", margin = margin(t = 10)),
      axis.text.y = element_text(size = 10, hjust = 0, margin = margin(r = 5)),
      axis.text.x = element_text(size = 10),
      plot.margin = margin(20, 60, 20, 20),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )
  
  return(p)
}

tidy_dfs <- tidy(cox_dfs, exponentiate = TRUE, conf.int = TRUE)
plot_dfs_enhanced <- make_publication_forest_plot(tidy_dfs, "Disease-Free Survival: Multivariable Cox Regression")

tidy_os <- tidy(cox_os, exponentiate = TRUE, conf.int = TRUE)
plot_os_enhanced <- make_publication_forest_plot(tidy_os, "Overall Survival: Multivariable Cox Regression")

ggsave("forest_dfs_publication.pdf", plot = plot_dfs_enhanced, width = 12, height = 6, dpi = 300, device = "pdf")
ggsave("forest_dfs_publication.png", plot = plot_dfs_enhanced, width = 12, height = 8, dpi = 300, device = "png")
ggsave("forest_os_publication.pdf", plot = plot_os_enhanced, width = 12, height = 6, dpi = 300, device = "pdf")
ggsave("forest_os_publication.png", plot = plot_os_enhanced, width = 12, height = 8, dpi = 300, device = "png")

print(plot_dfs_enhanced)
print(plot_os_enhanced)

create_summary_table <- function(tidy_df, outcome_name) {
  tidy_df %>%
    filter(term != "(Intercept)") %>%
    select(term, estimate, conf.low, conf.high, p.value) %>%
    mutate(
      HR_CI = sprintf("%.2f (%.2f-%.2f)", estimate, conf.low, conf.high),
      p_value = case_when(
        p.value < 0.001 ~ "<0.001",
        TRUE ~ sprintf("%.3f", p.value)
      ),
      outcome = outcome_name
    ) %>%
    select(outcome, term, HR_CI, p_value)
}

dfs_table <- create_summary_table(tidy_dfs, "Disease-Free Survival")
os_table <- create_summary_table(tidy_os, "Overall Survival")

cat("\n=== Disease-Free Survival Results ===\n")
print(dfs_table)
cat("\n=== Overall Survival Results ===\n")
print(os_table)

zph_dfs <- cox.zph(cox_dfs)
print(zph_dfs)

zph_os <- cox.zph(cox_os)
print(zph_os)
