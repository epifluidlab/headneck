# Comprehensive differential analysis and volcano plotting with q-value FDR

# ── Libraries ──
library(limma)
library(ggplot2)
library(qvalue)

# ── 0) Set working directory and load data ──
setwd("/home/xfj2191/epifluidlab/projects/cfdna_wgs_headneck_manuscript_2025/differential_bins")

# 0.1) Load fragmentation data
frag_data <- read.table(
  "/projects/b1198/epifluidlab/ravi/0425/headneck/Main_Figures/F3/combined_corrected_10.bed",
  header=TRUE, stringsAsFactors=FALSE
)
rownames(frag_data) <- paste(frag_data[,1], frag_data[,2], frag_data[,3], sep=":")
frag_mat <- as.matrix(frag_data[, 4:ncol(frag_data)])
rownames(frag_mat) <- rownames(frag_data)

# 0.2) Load metadata
metadata <- read.csv(
  "/projects/b1198/epifluidlab/ravi/0425/headneck/Main_Figures/F3/combined_covariates_10.csv",
  header=TRUE, stringsAsFactors=FALSE
)

# ── 1) Prepare metadata factors ──
metadata$PatientID   <- factor(metadata$OGID)
metadata$Group       <- factor(metadata$Treatment.Response)
metadata$TimeNumeric <- factor(with(metadata,
  ifelse(Type.of.Visit=="Screen", 0,
  ifelse(Type.of.Visit=="Day 0", 1, 2)))
)
metadata$Time        <- factor(metadata$Type.of.Visit,
                               levels=c("Screen","Day 0","Adj Wk 1"))
metadata$Gender      <- factor(metadata$Gender)
metadata$Race        <- factor(metadata$Race)
metadata$Ethnicity   <- factor(metadata$Ethnicity)
metadata$Diagnosis   <- factor(metadata$Diagnosis)
metadata$Alcohol     <- factor(metadata$Alcohol)
metadata$Smoking     <- factor(metadata$Smoking)


# ── 3) Build design matrix and estimate within-patient correlation ──
design <- model.matrix(~ Group * TimeNumeric, data=metadata)
colnames(design) <- make.names(colnames(design))

corfit <- duplicateCorrelation(
  frag_mat,
  design,
  block=metadata$PatientID
)

# ── 4) Fit linear model with weights and blocking ──
w <- arrayWeights(frag_mat, design)
fit <- lmFit(
  frag_mat,
  design,
  weights     = w,
  block       = metadata$PatientID,
  correlation = corfit$consensus.correlation
)
fit <- eBayes(fit)

# ── 5) Define and fit contrasts for each timepoint ──
cm <- makeContrasts(
  diffScreen = GroupResponder,
  diffDay0   = GroupResponder + GroupResponder.TimeNumeric1,
  diffAdjW1  = GroupResponder + GroupResponder.TimeNumeric2,
  levels     = design
)
fit2 <- contrasts.fit(fit, cm)
fit2 <- eBayes(fit2)

# ── 6) Extract time-point results WITHOUT BH (raw p-values) ──
res_scr <- topTable(fit2, coef="diffScreen", number=Inf, adjust.method="none")
res_d0  <- topTable(fit2, coef="diffDay0",   number=Inf, adjust.method="none")
res_aw1 <- topTable(fit2, coef="diffAdjW1",  number=Inf, adjust.method="none")

# ── 7) Compute empirical mean differences for each timepoint ──
mean_diff <- function(mat, meta, timept) {
  isR <- meta$Group=="Responder"     & meta$Time==timept
  isN <- meta$Group=="Non-Responder" & meta$Time==timept
  rowMeans(mat[, isR, drop=FALSE]) - rowMeans(mat[, isN, drop=FALSE])
}
md_scr <- mean_diff(frag_mat, metadata, "Screen")
md_d0  <- mean_diff(frag_mat, metadata, "Day 0")
md_aw1 <- mean_diff(frag_mat, metadata, "Adj Wk 1")

# ── 8) Attach meanDiff and compute q-values ──
attach_qval <- function(df, md_vec) {
  df$meanDiff <- md_vec[rownames(df)]
  df$qval     <- qvalue(df$P.Value)$qvalues
  df
}
vol1 <- attach_qval(res_scr, md_scr)
vol2 <- attach_qval(res_d0,  md_d0)
vol3 <- attach_qval(res_aw1, md_aw1)

# ── 9) Global comparison ignoring time ──
designG <- model.matrix(~ Group + Age + Race + Gender + Ethnicity + Diagnosis + Smoking + Alcohol, data=metadata)
corG    <- duplicateCorrelation(
  frag_mat,
  designG,
  block=metadata$PatientID
)
w <- arrayWeights(frag_mat, designG)
fitG <- lmFit(
  frag_mat,
  designG,
  weights     = w,
  block       = metadata$PatientID,
  correlation = corG$consensus.correlation
)
fitG <- eBayes(fitG)
res_glob <- topTable(fitG, coef="GroupResponder", number=Inf, adjust.method="none")
md_glob  <- rowMeans(frag_mat[, metadata$Group=="Responder"]) -
            rowMeans(frag_mat[, metadata$Group=="Non-Responder"])  
vol4 <- attach_qval(res_glob, md_glob)

# ── 10) Prepare volcano data frames (using q-values) ──
prep_vol <- function(df) {
  df$logQ   <- -log10(df$qval)
  df$signif <- "ns"
  df$signif[df$qval < 0.1 & df$meanDiff >  0] <- "pos"
  df$signif[df$qval < 0.1 & df$meanDiff <  0] <- "neg"
  df$signif <- factor(df$signif, levels=c("pos","neg","ns"))
  df
}
vol1 <- prep_vol(vol1)
vol2 <- prep_vol(vol2)
vol3 <- prep_vol(vol3)
vol4 <- prep_vol(vol4)

# ── 11) Volcano‐plotting function and plotting ──
plot_vol <- function(vol, title) {
  ggplot(vol, aes(x=meanDiff, y=logQ, color=signif)) +
    geom_point(alpha=0.6, size=1) +
    scale_color_manual(values=c(pos="red", neg="blue", ns="grey")) +
    geom_hline(yintercept=-log10(0.1), linetype="dashed") +
    labs(title=title,
         x="Mean difference (Responder − Non-Responder)",
         y="-Log10 q-value") +
    theme_minimal() + theme(legend.position="none")
}
p1 <- plot_vol(vol1, "Screen")
p2 <- plot_vol(vol2, "Day 0")
p3 <- plot_vol(vol3, "Adj Wk1")
p4 <- plot_vol(vol4, "Global (all timepoints)")

# ── 12) Save to PDF ──
#pdf("F3A.pdf", width=5, height=5)
print(p1)
print(p2)
print(p3)
print(p4)
dev.off()
write.csv(vol4, file = "F3A.csv", row.names = FALSE)

# ── 13) Write BED files for global significant regions at multiple q-value cutoffs ──
threshold <- 0.1
regs <- rownames(vol4)[vol4$qval < threshold]
parts <- strsplit(regs, ":", fixed = TRUE)

bed_df <- data.frame(
  chr   = vapply(parts, `[`, 1, FUN.VALUE = character(1)),
  start = as.integer(vapply(parts, `[`, 2, FUN.VALUE = character(1))),
  end   = as.integer(vapply(parts, `[`, 3, FUN.VALUE = character(1)))
)

write.table(
  bed_df,
  file      = "F3A.Differential_rMDS.bed",
  sep       = "\t",
  quote     = FALSE,
  row.names = FALSE,
  col.names = FALSE
)

dim(vol4[vol4$qval<0.1,])
vol4["chr12:52000000:52500000",]
vol4["chr12:52500000:53000000",]


