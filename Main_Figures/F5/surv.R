library(dplyr)
library(survival)
library(survminer)
library(ggplot2)
library(gridExtra)

setwd('/Users/ravibandaru/Downloads')
d <- read.csv("./survival_data.csv")

df_patient <- data.frame(
  response      = ifelse(d$REPINST_Predictions == "Responder", 1, 2),
  ihc           = ifelse(d$PDL1.IHC == ">20", 1,
                         ifelse(d$PDL1.IHC == "1-19", 2, 3)),
  tf            = ifelse(d$Tumor.Fraction == "Low Tumor Fraction", 1, 2),
  event_dfs     = as.integer(ifelse(d$E_Relapse,1,0)),
  time_to_event_dfs = as.integer(d$Relapse.Months..Adjuvant. * 30.5),
  event_os      = as.integer(ifelse(d$E_Survival,1,0)),
  time_to_event_os = as.integer(d$Survival.Months..Adjuvant. * 30.5)
)


predictor_settings <- list(
  response = list(
    legend = c("Responder", "Non‑Responder"),
    palette = c("#0173B2", "#CA9161"),
    title_dfs = "Disease‑Free Survival\n(Predicted Response)",
    title_os  = "Overall Survival\n(Predicted Response)"
  ),
  ihc = list(
    legend = c(">20", "1–19", "0"),
    palette = c("#15B01A", "#FFD700", "#DC143C"),
    title_dfs = "Disease‑Free Survival\n(PD‑L1 IHC)",
    title_os  = "Overall Survival\n(PD‑L1 IHC)"
  ),
  tf = list(
    legend = c("Low Tumor Fraction", "High Tumor Fraction"),
    palette = c("#15B01A", "#DC143C"),
    title_dfs = "Disease‑Free Survival\n(Tumor Fraction)",
    title_os  = "Overall Survival\n(Tumor Fraction)"
  )
)

dfs_surv <- Surv(df_patient$time_to_event_dfs, df_patient$event_dfs)
os_surv  <- Surv(df_patient$time_to_event_os,  df_patient$event_os)

dfs_plots <- list()
os_plots  <- list()

for (pred in names(predictor_settings)) {
  setting <- predictor_settings[[pred]]
  group <- factor(df_patient[[pred]], labels = setting$legend)
  
  km_dfs <- survfit(dfs_surv ~ group)
  km_os  <- survfit(os_surv  ~ group)
  
  p1 <- ggsurvplot(
    km_dfs, data = df_patient, legend.labs = setting$legend,
    palette = setting$palette, title = setting$title_dfs,
    pval = TRUE, ylab = "Event Probability", xlab = "Time (Days)",
    risk.table = FALSE, conf.int = FALSE
  )$plot
  
  p2 <- ggsurvplot(
    km_os, data = df_patient, legend.labs = setting$legend,
    palette = setting$palette, title = setting$title_os,
    pval = TRUE, ylab = "Event Probability", xlab = "Time (Days)",
    risk.table = FALSE, conf.int = FALSE
  )$plot
  
  dfs_plots[[length(dfs_plots) + 1]] <- p1
  os_plots[[length(os_plots) + 1]] <- p2
}

pdf("KM_combined.pdf", width = 18, height = 10)
grid.arrange(
  grobs = c(dfs_plots, os_plots),
  nrow = 2, ncol = 3
)
dev.off()
