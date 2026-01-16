library(mixOmics)
BLD_DATA <- "/Users/anojat/Documents/psycourse/bld/data/integrated_analysis"

lipid_data <- read.csv(
  file.path(BLD_DATA, "integrated_analysis_lipid_df_train.csv"),
            check.names=FALSE)
prs_data <- read.csv(
  file.path(BLD_DATA, "integrated_analysis_prs_df_train.csv"),
            check.names=FALSE)

outcome_data <- read.csv(
  file.path(BLD_DATA, "integrated_analysis_outcome_df_train.csv"),
            check.names=FALSE)

rownames(lipid_data) <- lipid_data[[1]]
lipid_data[[1]] <- NULL

rownames(prs_data) <- prs_data[[1]]
prs_data[[1]] <- NULL

rownames(outcome_data) <- outcome_data[[1]]
outcome_data[[1]] <- NULL

X <- list(
  lipids = as.matrix(lipid_data),
  prs = as.matrix(prs_data))

sapply(X, dim)

Y = outcome_data
Y$true_label <- as.factor(Y$true_label)
Y_bin <- Y$true_label
summary(Y)


# -----------------------------
# (Optional but recommended for sanity) Explore pairwise integration first
#    sPLS between PRS and lipids (unsupervised wrt outcome)
# -----------------------------
expl_keepX <- 10   # PRS (17 total)
expl_keepY <- 50   # lipids (361 total)

spls_expl <- spls(X$prs, X$lipids, ncomp = 1, keepX = expl_keepX, keepY = expl_keepY)
cor(spls_expl$variates$X, spls_expl$variates$Y)

# ----------------------------
# Here's the thing with the interpretability/shared axis vs. prediction thing.
# I choose 1 (in a range from [0,1,1]) because I want to optimise cross-corr

design <- matrix(c(0, 1,
                   1, 0),
                 nrow = 2, byrow = TRUE)
colnames(design) <- rownames(design) <- names(X)

ncomp <- 1

# --- here i guess the modeling part starts, not reviewed yet
