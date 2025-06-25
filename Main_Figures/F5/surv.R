library(dplyr)
library(survival)
library(ggplot2)
library(gridExtra)
library(scales)
library(grid)

setwd('/Users/ravibandaru/Downloads')
d <- read.csv("./SURVIVAL_DATA.csv")

df_patient <- data.frame(
  response = ifelse(d$REPINST_Predictions == "Responder", 1, 2),
  ihc = ifelse(d$PDL1.IHC == ">20", 1,
               ifelse(d$PDL1.IHC == "1-19", 2, 3)),
  tf = ifelse(d$Tumor.Fraction == "Low Tumor Fraction", 1, 2),
  stratification= factor(ifelse(d$Stratification == "Intermediate", 1, 2), labels = c("Intermediate", "High")),
  event_dfs_new = as.integer(ifelse(d$E_Relapse, 1, 0)),
  time_dfs_new = as.numeric(d$Relapse.Months..Adjuvant.),
  event_os_new = as.integer(ifelse(d$E_Survival, 1, 0)),
  time_os_new = as.numeric(d$Survival.Months..Adjuvant.),
  event_dfs_old = as.integer(ifelse(d$OLD.E_Relapse, 1, 0)),
  time_dfs_old = as.numeric(d$OLD.Relapse.Months..Adjuvant.),
  event_os_old = as.integer(ifelse(d$OLD.E_Survival, 1, 0)),
  time_os_old = as.numeric(d$OLD.Survival.Months..Adjuvant.)
)

predictor_settings <- list(
  response = list(
    legend = c("Responder (R)", "Non-Responder (NR)"),
    risk_labels = c("R", "NR"),
    palette = c("#0173B2", "#CA9161"),
    title_dfs = "Disease-Free Survival (Predicted Response)",
    title_os = "Overall Survival (Predicted Response)",
    filename = "Response"
  ),
  stratification = list(
    legend = c("Intermediate (I)", "High (HI)"),
    risk_labels = c("I", "HI"),
    palette = c("#15B01A", "#DC143C"),
    title_dfs = "Disease-Free Survival (Risk Stratification)",
    title_os = "Overall Survival (Risk Stratification)",
    filename = "Stratification"
  ),
  ihc = list(
    legend = c(">20", "1-19", "0"),
    risk_labels = c(">20", "1-19", "0"),
    palette = c("#15B01A", "#FFD700", "#DC143C"),
    title_dfs = "Disease-Free Survival (PD-L1 IHC)",
    title_os = "Overall Survival (PD-L1 IHC)",
    filename = "PDL1_IHC"
  ),
  tf = list(
    legend = c("Low Tumor Fraction", "High Tumor Fraction"),
    risk_labels = c("Low", "High"),
    palette = c("#15B01A", "#DC143C"),
    title_dfs = "Disease-Free Survival (Tumor Fraction)",
    title_os = "Overall Survival (Tumor Fraction)",
    filename = "Tumor_Fraction"
  )
)

create_km_plot <- function(surv_obj, group, colors, labels, risk_labels, title, max_time, time_points) {
  km_fit <- survfit(surv_obj ~ group)
  
  surv_data <- data.frame(
    time = km_fit$time,
    surv = km_fit$surv,
    group = rep(names(km_fit$strata), km_fit$strata)
  )
  
  surv_data$group <- gsub("group=", "", surv_data$group)
  
  unique_groups <- unique(surv_data$group)
  start_points <- data.frame(
    time = rep(0, length(unique_groups)),
    surv = rep(1, length(unique_groups)),
    group = unique_groups
  )
  
  surv_data <- rbind(start_points, surv_data)
  surv_data <- surv_data[order(surv_data$group, surv_data$time), ]
  
  surv_data <- surv_data[surv_data$time <= max_time, ]
  
  pval <- survdiff(surv_obj ~ group)$pvalue
  pval_text <- if (pval < 0.001) {
    "P < 0.001"
  } else if (pval < 0.01) {
    paste("P =", sprintf("%.3f", pval))
  } else {
    paste("P =", sprintf("%.3f", pval))
  }
  
  group_levels <- levels(group)
  names(colors) <- group_levels
  
  p_main <- ggplot(surv_data, aes(x = time, y = surv, color = group)) +
    geom_step(linewidth = 0.8) +
    scale_color_manual(values = colors, labels = labels, breaks = group_levels) +
    scale_y_continuous(limits = c(0, 1), labels = percent_format(), 
                       breaks = seq(0, 1, 0.2)) +
    scale_x_continuous(limits = c(0, max_time), expand = c(0.02, 0),
                       breaks = time_points, labels = as.character(time_points)) +
    labs(
      title = title,
      x = "Time (Months)",
      y = "Survival Probability",
      color = ""
    ) +
    theme_classic() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 10, face = "bold", margin = margin(b = 10)),
      axis.title.x = element_text(size = 10, face = "bold"),
      axis.title.y = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9, color = "black"),
      legend.text = element_text(size = 9),
      legend.position = "bottom",
      legend.margin = margin(t = -5),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      plot.margin = margin(t = 10, r = 10, b = 5, l = 10)
    ) +
    guides(color = guide_legend(override.aes = list(size = 1.5)))
  
  p_main <- p_main + 
    annotate("rect", xmin = max_time * 0.75, xmax = max_time * 1.05,
             ymin = 0.9, ymax = 1, fill = "white", color = "black", linewidth = 0.5) +
    annotate("text", x = max_time * 0.9, y = 0.95, 
             label = pval_text, size = 3.5, fontface = "bold")
  
  
  risk_table_data <- NULL
  
  for (i in 1:length(levels(group))) {
    group_name <- levels(group)[i]
    group_indices <- which(group == group_name)
    
    if (length(group_indices) > 0) {
      group_surv <- Surv(surv_obj[group_indices, 1], surv_obj[group_indices, 2])
      group_fit <- survfit(group_surv ~ 1)
      
      n_risk <- summary(group_fit, times = time_points, extend = TRUE)$n.risk
      
      risk_df <- data.frame(
        time = time_points,
        n_risk = n_risk,
        group = group_name,
        risk_label = risk_labels[i],
        group_num = i,
        stringsAsFactors = FALSE
      )
      
      risk_table_data <- rbind(risk_table_data, risk_df)
    }
  }
  
  risk_colors <- colors
  names(risk_colors) <- levels(group)
  
  p_risk <- ggplot(risk_table_data, aes(x = time, y = reorder(risk_label, -group_num))) +
    geom_text(aes(label = n_risk, color = group), size = 3, fontface = "bold") +
    scale_color_manual(values = risk_colors) +
    scale_x_continuous(limits = c(0, max_time), expand = c(0.02, 0),
                       breaks = time_points, labels = as.character(time_points)) +
    labs(x = "", y = "") +
    theme_classic() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(size = 8, color = "black", angle = 0),
      axis.text.y = element_text(size = 8, color = "black", face = "bold"),
      axis.line.x = element_blank(),
      axis.line.y = element_blank(),
      axis.ticks.x = element_line(color = "black", linewidth = 0.5),
      axis.ticks.y = element_blank(),
      plot.margin = margin(t = 0, r = 10, b = 5, l = 10),
      panel.background = element_rect(fill = "white", color = NA)
    ) +
    geom_hline(yintercept = seq(1.5, length(risk_labels) - 0.5, 1), 
               color = "black", linewidth = 0.3)
  
  risk_label <- textGrob("Number at Risk", x = 0, hjust = 0, 
                         gp = gpar(fontsize = 9, fontface = "bold"))
  
  combined_plot <- grid.arrange(
    p_main, 
    risk_label,
    p_risk, 
    ncol = 1, 
    heights = c(4, 0.15, 1)
  )
  
  return(combined_plot)
}

dfs_surv <- Surv(df_patient$time_dfs_new, df_patient$event_dfs_new)
os_surv <- Surv(df_patient$time_os_new, df_patient$event_os_new)

for (pred in names(predictor_settings)) {
  setting <- predictor_settings[[pred]]
  group <- factor(df_patient[[pred]], labels = setting$legend)
  
  pdf(paste0("KM_DFS_", setting$filename, "_NEW.pdf"), width = 4.5, height = 4.5)
  p1 <- create_km_plot(dfs_surv, group, setting$palette, setting$legend, setting$risk_labels, 
                       setting$title_dfs, max_time = 72, time_points = c(0, 12, 24, 36, 48, 60, 72))
  dev.off()
  
  pdf(paste0("KM_OS_", setting$filename, "_NEW.pdf"), width = 4.5, height = 4.5)
  p2 <- create_km_plot(os_surv, group, setting$palette, setting$legend, setting$risk_labels, 
                       setting$title_os, max_time = 72, time_points = c(0, 12, 24, 36, 48, 60, 72))
  dev.off()
}

dfs_surv <- Surv(df_patient$time_dfs_old, df_patient$event_dfs_old)
os_surv <- Surv(df_patient$time_os_old, df_patient$event_os_old)

for (pred in names(predictor_settings)) {
  setting <- predictor_settings[[pred]]
  group <- factor(df_patient[[pred]], labels = setting$legend)
  
  pdf(paste0("KM_DFS_", setting$filename, "_OLD.pdf"), width = 4.5, height = 4.5)
  p1 <- create_km_plot(dfs_surv, group, setting$palette, setting$legend, setting$risk_labels, 
                       setting$title_dfs, max_time = 60, time_points = c(0, 12, 24, 36, 48, 60))
  dev.off()
  
  pdf(paste0("KM_OS_", setting$filename, "_OLD.pdf"), width = 4.5, height = 4.5)
  p2 <- create_km_plot(os_surv, group, setting$palette, setting$legend, setting$risk_labels, 
                       setting$title_os, max_time = 60, time_points = c(0, 12, 24, 36, 48, 60))
  dev.off()
}