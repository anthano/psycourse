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
summary(Y_bin)


# -----------------------------
# (Optional but recommended for sanity) Explore pairwise integration first
#    sPLS between PRS and lipids (unsupervised wrt outcome)
# -----------------------------
expl_keepX <- 10   # PRS (17 total)
expl_keepY <- 50   # lipids (361 total)

spls_expl <- spls(X$prs, X$lipids, ncomp = 1, keepX = expl_keepX, keepY = expl_keepY)
cor(spls_expl$variates$X, spls_expl$variates$Y)

# it does compare to the cca I did (~0.3)

# ----------------------------
# Here's the thing with the interpretability/shared axis vs. prediction thing.
# I choose 1 (in a range from [0,1,1]) because I want to optimise cross-corr

design <- matrix(c(0, 1,
                   1, 0),
                 nrow = 2, byrow = TRUE)
colnames(design) <- rownames(design) <- names(X)

ncomp <- 1

# --- here i guess the modeling part starts, not reviewed yet

basic.diablo.psycourse <- block.plsda(X,Y_bin, ncomp = 1, design=design)
perf.diablo.psycourse <- perf(basic.diablo.psycourse, validation= 'Mfold', folds = 5, nrepeat=5)
plot(perf.diablo.psycourse)

# ---
# --- tune sparsity for DIABLO (block.splsda) ---
set.seed(1)

ncomp <- 1  # keep small for your setting; consider 2 only if stable

test_keepX <- list(
  lipids = c(10, 20, 40, 60, 80, 120),  # selected lipid species
  prs    = c(3, 5, 8, 10, 12, 15)        # selected PRS (out of 17)
)

tune_res <- tune.block.splsda(
  X = X,
  Y = Y_bin,
  ncomp = ncomp,
  test.keepX = test_keepX,
  design = design,
  validation = "Mfold",
  folds = 5,
  nrepeat = 10,
  dist = "mahalanobis.dist"  # use the best-performing distance from your plot
)

tune_res$choice.keepX

# --- fit final DIABLO model with chosen keepX ---
final_keepX <- tune_res$choice.keepX

diablo_fit <- block.splsda(
  X = X,
  Y = Y_bin,
  ncomp = ncomp,
  keepX = final_keepX,
  design = design
)

# --- performance (balanced error rate etc.) ---
set.seed(1)
perf_res <- perf(
  diablo_fit,
  validation = "Mfold",
  folds = 5,
  nrepeat = 10,
  dist = "mahalanobis.dist"
)

perf_res$error.rate
plot(perf_res)

# --- integration strength + interpretation ---
# correlation between block variates (shared axis)
cor(diablo_fit$variates$prs[,1], diablo_fit$variates$lipids[,1])

# selected features
sv_lipids <- selectVar(diablo_fit, block = "lipids", comp = 1)
selected_lipids <- sv_lipids$lipids$name
selected_lipids

sv_prs <- selectVar(diablo_fit, block = "prs", comp = 1)
selected_prs <- sv_prs$prs$name
selected_prs

prs_weights <- sv_prs$prs$value
lipid_weights <- sv_lipids$lipids$value



# sample scores plot + cross-block feature correlation heatmap
plotIndiv(diablo_fit, ind.names = FALSE, legend = TRUE, title = "DIABLO comp1")
cimDiablo(diablo_fit, comp = 1, legend.position = "right")
