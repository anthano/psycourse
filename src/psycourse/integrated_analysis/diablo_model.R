library(mixOmics)
BLD_DATA <- "/Users/anojat/Documents/psycourse/bld/data"
lipid_data <- read.csv(
  file.path(BLD_DATA, "integrated_analysis_lipid_data.csv")
)
prs_data <- read.csv(
  file.path(BLD_DATA, "integrated_analysis_prs_data.csv")
)
outcome_data <- read.csv(
  file.path(BLD_DATA, "integrated_analysis_outcome_data.csv")
)

data <- list(lipid_data,
             prs_data)

lapply(data, dim)

Y = outcome_data

## NOTE: Do train-test split + set index or sth
