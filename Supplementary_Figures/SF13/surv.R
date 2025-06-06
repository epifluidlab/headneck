library(dplyr)
library(survival)
library(survminer)
library(ggplot2)
library(survcomp)
library(gridExtra)

setwd('/Users/ravibandaru/Downloads')
d <- read.csv("./survival_data.csv")

df_patient <- data.frame(
  id                         = d[[1]],
  age                        = d$Age,
  gender                     = ifelse(d$Gender=="Male",1,2),
  ethnicity                  = as.integer(as.factor(d$Ethnicity)),
  diagnosis                  = as.integer(as.factor(d$Diagnosis)),
  hpv                        = ifelse(d$HPV=="Unknown",1,2),
  smoking                    = ifelse(d$Smoking=="No",1,2),
  alcohol                    = ifelse(d$Alcohol=="No",1,2),
  stratification             = ifelse(d$Stratification=="Intermediate",1,2),
  response                   = ifelse(d$Predicted.Treatment.Response=="Responder",1,2),
  ihc                        = ifelse(d$PDL1.IHC==">20",1,ifelse(d$PDL1.IHC=="1-19",2,3)),
  tf                         = ifelse(d$Tumor.Fraction=="Low Tumor Fraction",1,2),
  event_dfs                  = as.integer(ifelse(d$E_Relapse,1,0)),
  time_to_event_dfs          = as.integer(d$Relapse.Months*30),
  event_os                   = as.integer(ifelse(d$E_Survival,1,0)),
  time_to_event_os           = as.integer(d$Survival.Months*30),
  responder_stratification   = ifelse(
    d$Predicted.Treatment.Response.with.Stratification=="Responder_Intermediate",1,
    ifelse(d$Predicted.Treatment.Response.with.Stratification=="Responder_High",2,
           ifelse(d$Predicted.Treatment.Response.with.Stratification=="Non-Responder_Intermediate",3,
                  ifelse(d$Predicted.Treatment.Response.with.Stratification=="Non-Responder_High",4,NA)))),
  ihc_stratification         = ifelse(
    d$IHC.with.Stratification==">20_Intermediate",1,
    ifelse(d$IHC.with.Stratification==">20_High",2,
           ifelse(d$IHC.with.Stratification=="1-19_Intermediate",3,
                  ifelse(d$IHC.with.Stratification=="1-19_High",4,
                         ifelse(d$IHC.with.Stratification=="0_Intermediate",5,
                                ifelse(d$IHC.with.Stratification=="0_High",6,NA)))))),
  tf_stratification          = ifelse(
    d$Tumor.Fraction.with.Stratification=="Low Tumor Fraction_Intermediate",1,
    ifelse(d$Tumor.Fraction.with.Stratification=="Low Tumor Fraction_High",2,
           ifelse(d$Tumor.Fraction.with.Stratification=="High Tumor Fraction_Intermediate",3,
                  ifelse(d$Tumor.Fraction.with.Stratification=="High Tumor Fraction_High",4,NA))))
)

predictor_settings <- list(
  responder_stratification = list(
    legend  = c("Responder (Intermediate)","Responder (High)","Non-Responder (Intermediate)","Non-Responder (High)"),
    palette = c("#98FB98","#006400","#FA8072","#B22222"),
    title   = "Disease-Free Survival\n(Predicted Response)"
  ),
  ihc_stratification = list(
    legend  = c(">20 (Intermediate)",">20 (High)","1–19 (Intermediate)","1–19 (High)","0 (Intermediate)","0 (High)"),
    palette = c("#98FB98","#006400","#FFD700","#B8860B","#FA8072","#B22222"),
    title   = "Disease-Free Survival\n(PD-L1 IHC)"
  ),
  tf_stratification = list(
    legend  = c("Low Tumor Fraction (Intermediate)","Low Tumor Fraction (High)","High Tumor Fraction (Intermediate)","High Tumor Fraction (High)"),
    palette = c("#98FB98","#006400","#FA8072","#B22222"),
    title   = "Disease-Free Survival\n(Tumor Fraction)"
  )
)

dfs_plots <- list()
os_plots  <- list()


for(pred in names(predictor_settings)) {
  settings  <- predictor_settings[[pred]]
  surv_dfs  <- survfit(as.formula(paste0("Surv(time_to_event_dfs, event_dfs) ~ ", pred)), data = df_patient)
  surv_os   <- survfit(as.formula(paste0("Surv(time_to_event_os,  event_os)  ~ ", pred)), data = df_patient)
  
  # common theme tweak: no grid lines
  no_grid <- theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  # Disease-free plot
  p_dfs <- ggsurvplot(
    surv_dfs, 
    data        = df_patient,
    palette     = settings$palette,
    legend.title= "",
    legend.labs = settings$legend,
    risk.table  = FALSE,
    ggtheme     = no_grid
  )$plot +
    labs(title = settings$title) +
    guides(
      colour = guide_legend(nrow = 2, byrow = TRUE)
    )
  
  # Overall survival plot (change title only)
  os_title <- sub("Disease-Free", "Overall", settings$title)
  p_os <- ggsurvplot(
    surv_os,
    data        = df_patient,
    palette     = settings$palette,
    legend.title= "",
    legend.labs = settings$legend,
    risk.table  = FALSE,
    ggtheme     = no_grid
  )$plot +
    labs(title = os_title) +
    guides(
      colour = guide_legend(nrow = 2, byrow = TRUE)
    )
  
  dfs_plots[[pred]] <- p_dfs
  os_plots[[pred]]  <- p_os
}

pdf("SF13.pdf",width=15,height=8)
grid.arrange(
  dfs_plots[[1]], dfs_plots[[2]], dfs_plots[[3]],
  os_plots[[1]],  os_plots[[2]],  os_plots[[3]],
  ncol=3, nrow=2
)
dev.off()

for(pred in names(predictor_settings)){
  lvls <- sort(unique(df_patient[[pred]]))
  pairs <- combn(lvls,2,simplify=FALSE)
  cat("\nDFS p-values for",pred,":\n")
  for(pr in pairs){
    sub <- df_patient[df_patient[[pred]] %in% pr,]
    sub$grp <- factor(sub[[pred]],levels=pr,labels=predictor_settings[[pred]]$legend[match(pr,lvls)])
    lr <- survdiff(Surv(time_to_event_dfs,event_dfs)~grp,data=sub)
    p  <- 1 - pchisq(lr$chisq, length(lr$n)-1)
    cat(levels(sub$grp)[1],"vs",levels(sub$grp)[2],": p=",signif(p,3),"\n")
  }
  cat("\nOS p-values for",pred,":\n")
  for(pr in pairs){
    sub <- df_patient[df_patient[[pred]] %in% pr,]
    sub$grp <- factor(sub[[pred]],levels=pr,labels=predictor_settings[[pred]]$legend[match(pr,lvls)])
    lr <- survdiff(Surv(time_to_event_os,event_os)~grp,data=sub)
    p  <- 1 - pchisq(lr$chisq, length(lr$n)-1)
    cat(levels(sub$grp)[1],"vs",levels(sub$grp)[2],": p=",signif(p,3),"\n")
  }
}