## ============================================================================
## Requirements (install once):
# BiocManager::install(c("SingleCellExperiment", "scran", "scater",
#                         "DropletUtils", "zellkonverter", "scry"))
# install.packages(c("Matrix", "purrr", "Rcpp", "RcppArmadillo",
#                     "clue", "mclust", "RANN", "ggplot2", "gridExtra", "Seurat"))
## ============================================================================

suppressMessages({
  # Core data types & manipulation
  library(SingleCellExperiment)
  library(Matrix)
  library(dplyr)
  
  # Modeling, graph analytics & math
  library(Seurat)
  library(scry)        # Fixes: nullResiduals
  library(igraph)      # Fixes: graph.adjacency
  library(dbscan)      # Used internally by spatialMNN for kNN
  library(RANN)
  library(mclust)
  library(clue)
  
  # Visualization
  library(ggplot2)
  library(gridExtra)
})

## ----------------------------------------------------------------------
## Loaders
##
## Python's AnnData -> R's SingleCellExperiment (SCE). Ground-truth labels
## go in colData(sce)$annotation; spatial coords go in reducedDim(sce, "spatial").
## Each loader returns list(sce = <SCE>, n_clusters = <int>).
## ----------------------------------------------------------------------

# load_DLPFC <- function(file) {
#   # NOTE: DropletUtils::read10xVisium is the closest R analogue of
#   # sc.read_visium(). Column-naming conventions differ slightly between
#   # scanpy and DropletUtils (e.g. barcode prefixing with sample name) --
#   # double check colnames(sce) line up with the barcodes in your
#   # manual_annotations CSV before trusting the merge below.
#   base <- file.path("..", "..", "app", "data",
#                     "Human dorsolateral prefrontal cortex (DLPFC)")
#   
#   spe <- SpatialExperiment::read10xVisium(
#     samples   = file.path(base, file),
#     sample_id = file,
#     type      = "sparse",
#     data      = "filtered",
#     images    = "lowres",
#     load      = TRUE
#   )
#   
#   ground_truth <- read.csv(
#     file.path(base, "manual_annotations", paste0(file, "_true.csv")),
#     row.names = 1
#   )
#   
#   # Match ground truth to spots by barcode. Adjust the barcode-cleaning
#   # regex if read10xVisium has prefixed colnames with the sample name.
#   barcodes <- sub("^[^_]+_", "", colnames(sce))
#   colData(sce)$annotation <- ground_truth[barcodes, "label"]
#   
#   n_clusters <- length(unique(na.omit(colData(sce)$annotation)))
#   list(sce = sce, n_clusters = n_clusters)
# }


load_DLPFC <- function(file) {
  base <- file.path("app", "data",
                    "Human dorsolateral prefrontal cortex (DLPFC)")
  sample_dir <- file.path(base, file)
  matrix_h5 <- file.path(sample_dir, paste0(file, "_filtered_feature_bc_matrix.h5"))
  
  sce <- DropletUtils::read10xCounts(matrix_h5, type = "HDF5", col.names = TRUE)
  colnames(sce) <- colData(sce)$Barcode
  
  positions_file <- file.path(sample_dir, "spatial", "tissue_positions_list.csv")
  has_header <- grepl("barcode", readLines(positions_file, n = 1), ignore.case = TRUE)
  positions <- read.csv(positions_file, header = has_header, row.names = 1)
  if (!has_header) {
    colnames(positions) <- c("in_tissue", "array_row", "array_col",
                             "pxl_row_in_fullres", "pxl_col_in_fullres")
  }
  positions <- positions[colnames(sce), , drop = FALSE]
  
  colData(sce)$in_tissue <- positions$in_tissue
  colData(sce)$row <- positions$array_row
  colData(sce)$col <- positions$array_col
  reducedDim(sce, "spatial") <-
    as.matrix(positions[, c("pxl_col_in_fullres", "pxl_row_in_fullres")])
  
  ground_truth <- read.csv(
    file.path(base, "manual_annotations", paste0(file, "_true.csv")),
    row.names = 1
  )
  
  # Match ground truth to spots by barcode. Adjust the regex if your
  # barcodes carry a sample-name prefix or a "-1" suffix mismatch against
  # the manual_annotations row names.
  barcodes <- sub("^[^_]+_", "", colnames(sce))
  colData(sce)$annotation <- ground_truth[barcodes, "label"]
  
  n_clusters <- length(unique(na.omit(colData(sce)$annotation)))
  list(sce = sce, n_clusters = n_clusters)
}


load_HER2 <- function(file) {
  base <- file.path("app", "data", "HER2+ breast cancer")
  
  # Counts (ST-cnts): spots x genes, same orientation as the pandas read_csv
  counts <- read.delim(
    file.path(base, "ST-cnts", paste0(file, ".tsv.gz")),
    row.names = 1, check.names = FALSE
  )
  
  # Spots (ST-spotfiles)
  spots <- read.delim(file.path(base, "ST-spotfiles", paste0(file, "_selection.tsv")))
  spots$key <- paste0(spots$x, "x", spots$y)
  rownames(spots) <- spots$key
  
  keys <- rownames(counts)
  selected_keys <- spots$key[spots$selected == 1]
  keep <- keys %in% selected_keys
  counts <- counts[keep, , drop = FALSE]
  keys <- keys[keep]
  
  col_data <- DataFrame(
    key      = keys,
    array_x  = spots[keys, "x"],
    array_y  = spots[keys, "y"],
    pixel_x  = spots[keys, "pixel_x"],
    pixel_y  = spots[keys, "pixel_y"],
    new_x    = spots[keys, "new_x"],
    new_y    = spots[keys, "new_y"]
  )
  col_data$float_key <- paste0(round(col_data$new_x, 3), "x", round(col_data$new_y, 3))
  
  # Pathologist ground truth (ST-pat)
  labels <- read.delim(
    file.path(base, "ST-pat", "lbl", paste0(file, "_labeled_coordinates.tsv")),
    row.names = 1
  )
  labels <- labels[!is.na(labels$x) & !is.na(labels$y), ]
  labels$float_key <- paste0(round(labels$x, 3), "x", round(labels$y, 3))
  rownames(labels) <- labels$float_key
  
  col_data$annotation <- labels[col_data$float_key, "label"]
  
  sce <- SingleCellExperiment(
    assays  = list(counts = t(as.matrix(counts))),  # genes x spots
    colData = col_data
  )
  # array_x/array_y form an integer ST rectangular grid -> feed find_neighbors()
  # via platform = "ST" using these as row/col.
  colData(sce)$col <- col_data$array_x
  colData(sce)$row <- col_data$array_y
  reducedDim(sce, "spatial") <- cbind(x = col_data$array_x, y = col_data$array_y)
  
  n_clusters <- length(unique(na.omit(col_data$annotation)))
  list(sce = sce, n_clusters = n_clusters)
}


load_MOSTA <- function(file) {
  # zellkonverter::readH5AD is the closest R analogue of sc.read_h5ad();
  # it maps adata.obsm['spatial'] -> reducedDim(sce, "spatial") and
  # adata.obs -> colData(sce) automatically.
  base <- file.path("app", "data", "Stereo-seq MOSTA")
  sce <- zellkonverter::readH5AD(file.path(base, file, paste0(file, ".MOSTA.h5ad")))

  # This h5ad carries two assays: adata.X (log-normalized, non-integer) mapped
  # to "X", and the raw integer counts mapped to "count" (singular). Two callers
  # want the raw layer under different names: run_BISON() hardcodes an assay
  # literally named "counts" (via logNormCounts()), while the SpaRTaCo benchmark
  # passes assay_name = "count" straight through to run_spartaco(). So expose
  # BOTH -- keep the native "count" and add a "counts" alias pointing at the same
  # raw matrix. (We don't use rename_counts_assay() here: with two assays and
  # neither named "counts" it aborts as ambiguous, and it can't know that
  # "count", not "X", is the raw layer.)
  an <- SummarizedExperiment::assayNames(sce)
  if (!("counts" %in% an) && ("count" %in% an)) {
    SummarizedExperiment::assay(sce, "counts") <- SummarizedExperiment::assay(sce, "count")
  }

  # MOSTA bins are on an integer grid; derive row/col from the spatial coords
  # for find_neighbors()'s lattice construction. If your bins aren't exact
  # integers, round first.
  spatial <- reducedDim(sce, "spatial")
  colData(sce)$col <- round(spatial[, 1])
  colData(sce)$row <- round(spatial[, 2])
  
  n_clusters <- length(unique(na.omit(colData(sce)$annotation)))
  list(sce = sce, n_clusters = n_clusters)
}


preprocess_data <- function(sce, file) {
  if ("in_tissue" %in% colnames(colData(sce))) {
    sce <- sce[, colData(sce)$in_tissue == 1]
  }
  rownames(sce) <- make.unique(rownames(sce))
  colData(sce)$slide_id <- file
  sce
}


## ----------------------------------------------------------------------
## Generic spatial-scale helpers (kept for parity / reuse elsewhere in your
## pipeline, e.g. if you still run STMSGAL-style models alongside BISON).
## Not used by run_BISON() -- BISON's neighbor graph is a discrete grid
## adjacency, not a physical-distance radius, so no cross-platform rescaling
## is needed for it.
## ----------------------------------------------------------------------

get_median_nn_distance <- function(coords, k = 6) {
  nn <- RANN::nn2(coords, k = k + 1)
  median(nn$nn.dists[, k + 1])
}

rescale_spatial_coords_to_reference <- function(sce, reference_median_nn_dist,
                                                spatial_name = "spatial", k = 6) {
  coords <- reducedDim(sce, spatial_name)
  current_median <- get_median_nn_distance(coords, k = k)
  scale_factor <- reference_median_nn_dist / current_median
  reducedDim(sce, spatial_name) <- coords * scale_factor
  
  message(sprintf(
    "Rescaled spatial coords by %.4fx (median NN dist: %.2f -> %.2f)",
    scale_factor, current_median, reference_median_nn_dist
  ))
  sce
}


## ----------------------------------------------------------------------
## BISON helper: size factor / gene factor
##
## ASSUMPTION FLAG: the function that produced sce$sizefactor /
## rowData(sce)$genefactor in MOB_sce_filter.RData was not among the files
## you gave me. This implements a standard multiplicative offset
## decomposition for the Poisson model lambda_ij = s_i * g_j * mu_{rho_j,k_i}:
##   sizefactor_i = (library size of spot i) / (mean library size)
##   genefactor_j = (mean count of gene j across spots) / (mean of those means)
## Check this against the BISON paper/repo if exact reproduction of published
## numbers matters -- e.g. they may use scran::computeSumFactors() instead of
## a raw library-size ratio.
## ----------------------------------------------------------------------

compute_size_gene_factors <- function(count_mat) {
  lib_size <- colSums(count_mat)
  sizefactor <- lib_size / mean(lib_size)
  
  gene_mean <- rowMeans(count_mat)
  genefactor <- gene_mean / mean(gene_mean)
  
  list(sizefactor = sizefactor, genefactor = genefactor)
}

## ----------------------------------------------------------------------
## spatialMNN helpers
## ----------------------------------------------------------------------

rename_counts_assay <- function(sce, assay_name = "counts") {
  present_assays <- SummarizedExperiment::assayNames(sce)
  
  if (!(assay_name %in% present_assays)) {
    if (length(present_assays) == 1) {
      # Print the old assay name before renaming it
      message("Renaming assay '", present_assays[1], "' to '", assay_name, "'.")
      SummarizedExperiment::assayNames(sce)[1] <- assay_name
    } else {
      stop("Ambiguous: sce has multiple assays and none is named '", assay_name,
           "'. Assays present: ", paste(present_assays, collapse = ", "))
    }
  } else {
    # Optional: Let the user know the requested assay was already present
    message("Assay '", assay_name, "' already present. Available assays: ", 
            paste(present_assays, collapse = ", "))
  }
  sce
}

sce2Seurat <- function(sce, annotation_col = "annotation", assay_name = "counts") {
  stopifnot(assay_name %in% SummarizedExperiment::assayNames(sce))
  
  counts_mat <- as.matrix(SummarizedExperiment::assay(sce, assay_name))
  coords <- SingleCellExperiment::reducedDim(sce, "spatial")
  
  meta_df <- data.frame(
    coord_x = as.numeric(coords[, 1]),
    coord_y = as.numeric(coords[, 2]),
    row.names = colnames(sce)
  )
  
  if (!is.null(annotation_col) &&
      annotation_col %in% colnames(SummarizedExperiment::colData(sce))) {
    meta_df$layer <- SummarizedExperiment::colData(sce)[[annotation_col]]
  }
  
  SeuratObject::CreateSeuratObject(counts = counts_mat, meta.data = meta_df)
}

## ----------------------------------------------------------------------
## run_BISON(): BISON's analogue of run_STMSGAL() / run_hyperGCN() etc.
##
## platform: "Visium" (hex grid, 6 neighbors, e.g. DLPFC) or "ST"
##   (rectangular grid, 4 neighbors, e.g. HER2+, MOSTA bins) -- passed
##   straight through to find_neighbors()/find_neighbor_index() from utils.R.
##
## K: number of spot clusters (spatial domains) -- pass n_clusters (ground
##    truth count), same role as `level` in the Python run_* functions.
## R_: number of gene clusters ("patterns"). NOTE the trailing underscore --
##    BISON() in main.R actually ignores its own `R` argument and reads a
##    *global* variable `L` instead (this is present in the source file you
##    gave me, not something introduced here). We set that global right
##    before calling BISON() so R_ actually takes effect.
## ----------------------------------------------------------------------

run_BISON <- function(sce, level, platform = c("Visium", "ST"),
                      n_top_genes = 1000, K = level, R_ = level,
                      f = 1, n_iters = 1000, seed = 1,
                      export_path = NULL) {
  platform <- match.arg(platform)
  
  ## --- HVG selection (mirrors the tutorial's preprocessing block) ---
  sce_log <- scater::logNormCounts(sce)
  dec <- scran::modelGeneVar(sce_log, assay.type = "logcounts")
  top <- scran::getTopHVGs(dec, n = n_top_genes)
  sce_hvg <- sce[top, ]
  
  ## --- Spatial adjacency graph ---
  Adj <- find_neighbors(sce_hvg, platform = platform, coordinate = "lattice")
  neighbors <- find_neighbor_index(Adj, platform = platform)
  
  ## --- Count matrix + size/gene factors ---
  count_mat <- as.matrix(assay(sce_hvg, "counts"))
  P <- nrow(count_mat)
  N <- ncol(count_mat)
  
  factors <- compute_size_gene_factors(count_mat)
  s_mat <- matrix(rep(factors$sizefactor, each = P), P, N)
  g_mat <- matrix(rep(factors$genefactor, N), P, N)
  sg <- s_mat * g_mat
  
  ## --- BISON() reads gene-cluster count L from the global environment, not
  ## from a function argument (see BISON()'s body in main.R: `L_init = L`).
  ## Set it here right before the call so R_ takes effect as intended.
  assign("L", R_, envir = .GlobalEnv)
  
  result <- BISON(
    count_mat = count_mat, sg = sg, neighbors = neighbors,
    K = K, R = R_, f = f, n_iters = n_iters, seed = seed
  )
  
  predicted_key <- paste0("bison_spot_label_", level)
  colData(sce)[[predicted_key]] <- factor(result$pred_spot_label)
  
  if (!is.null(export_path)) {
    save_annotations(sce, predicted_key, export_path)
  }
  
  list(sce = sce, predicted_key = predicted_key, bison_result = result)
}


run_spatialMNN <- function(sce,
                           level,
                           sample_name,
                           assay_name,
                           annotation_col = "annotation",
                           top_pcs = 8,
                           cor_threshold = 0.6,
                           nn = 6,
                           cl_resolution = 10,
                           cl_min = 5,
                           find_HVG = TRUE,
                           hvg = 2000,
                           cor_met = "PC",
                           edge_smoothing = TRUE,
                           use_glmpca = TRUE,
                           verbose = TRUE,
                           export_path = NULL) {
  
  seu <- sce2Seurat(sce, annotation_col = annotation_col, assay_name = assay_name)
  seu@meta.data[["orig.ident"]] <- sample_name
  seurat_ls <- setNames(list(seu), sample_name)
  
  seurat_ls <- stage_1(seurat_ls,
                       cor_threshold = cor_threshold,
                       nn = nn,
                       cl_resolution = cl_resolution,
                       top_pcs = top_pcs,
                       cl_min = cl_min,
                       find_HVG = find_HVG,
                       hvg = hvg,
                       cor_met = cor_met,
                       edge_smoothing = edge_smoothing,
                       use_glmpca = use_glmpca,
                       verbose = verbose)
  
  rtn_ls <- stage_2(seurat_ls, cl_key = "merged_cluster",
                    rtn_seurat = T, nn_2 = 10, method = "MNN",
                    top_pcs = 8, use_glmpca = T, rare_ct = "m", resolution = 1)
  seurat_ls <- assign_label(seurat_ls, rtn_ls$cl_df, "MNN", 0.6, cl_key = "merged_cluster")
  
  predicted_key <- paste0("spatialMNN_spot_label_", level)
  SummarizedExperiment::colData(sce)[[predicted_key]] <-
    factor(seurat_ls[[sample_name]]@meta.data[["merged_cluster"]])
  
  if (!is.null(export_path)) save_annotations(sce, predicted_key, export_path)
  
  list(sce = sce, predicted_key = predicted_key, seurat_obj = seurat_ls[[sample_name]])
}


run_spartaco <- function(sce,
                         level,
                         sample_name,
                         assay_name,
                         annotation_col = "annotation",
                         K = 10,
                         R = NULL,
                         find_HVG = TRUE,
                         hvg = 2000,
                         max.iter = 1000,
                         conv.criterion = list(epsilon = 0.01, iterations = 5),
                         verbose = TRUE,
                         export_path = NULL) {
  
  if (is.null(R)) R <- level   # R = spot clusters -> matches n_clusters
  
  counts <- as.matrix(SummarizedExperiment::assay(sce, assay_name))
  
  if (find_HVG) {
    dev  <- scry::devianceFeatureSelection(counts)
    keep <- order(dev, decreasing = TRUE)[seq_len(min(hvg, length(dev)))]
    counts <- counts[keep, , drop = FALSE]
  }
  
  # spartaco needs non-degenerate rows/cols
  counts <- counts[rowSums(counts) > 0, colSums(counts) > 0, drop = FALSE]
  
  coordinates <- reducedDim(sce, "spatial")
  coordinates <- coordinates[colnames(counts), , drop = FALSE]
  
  # Forward `verbose` to spartaco(). Its progress display (an mclapply2 fifo +
  # a forked progress monitor + a txtProgressBar) is what corrupts under nested
  # forking, so callers running many slides concurrently should pass verbose =
  # FALSE. Serial callers keep the default TRUE and are unaffected.
  fit <- spartaco::spartaco(
    data = counts,
    coordinates = coordinates,
    K = K,
    R = R,
    max.iter = max.iter,
    conv.criterion = conv.criterion,
    verbose = verbose
  )
  
  predicted_key <- paste0("spartaco_spot_label_", level)
  spot_labels <- setNames(factor(fit$Ds), colnames(counts))
  
  # spots dropped during zero-filtering get NA -- fine, calculate_ARI/F1 already
  # drop NAs via their `valid` mask
  colData(sce)[[predicted_key]] <- spot_labels[colnames(sce)]
  
  if (!is.null(export_path)) save_annotations(sce, predicted_key, export_path)
  
  list(sce = sce, predicted_key = predicted_key, spartaco_result = fit)
}  



## Visualization
plot_annotations <- function(sce, model, level, predicted, tissue, slide,
                             export_path, spatial_name = "spatial") {
  suppressMessages(library(ggplot2))
  
  coords <- as.data.frame(reducedDim(sce, spatial_name))
  colnames(coords)[1:2] <- c("x", "y")
  coords$annotation <- factor(colData(sce)$annotation)
  coords$predicted <- factor(colData(sce)[[predicted]])
  
  p1 <- ggplot(coords, aes(x, y, color = annotation)) +
    geom_point(size = 0.6) +
    coord_fixed() +
    theme_minimal() +
    ggtitle(sprintf("Ground truth (tissue=%s, slide=%s, n=%d)", tissue, slide, level))
  
  p2 <- ggplot(coords, aes(x, y, color = predicted)) +
    geom_point(size = 0.6) +
    coord_fixed() +
    theme_minimal() +
    ggtitle(sprintf("Predicted (model=%s)", model))
  
  if (!dir.exists(export_path)) {
    dir.create(dirname(export_path), recursive = TRUE, showWarnings = FALSE)
  }
  combined <- gridExtra::arrangeGrob(p1, p2, ncol = 2)
  ggsave(export_path, combined, width = 12, height = 6)
}


save_annotations <- function(sce, predicted, export_path, spatial_name = "spatial") {
  coords <- reducedDim(sce, spatial_name)
  df <- data.frame(
    annotation = colData(sce)$annotation,
    pred       = colData(sce)[[predicted]],
    x = coords[, 1],
    y = coords[, 2]
  )
  colnames(df)[2] <- predicted
  
  if (!dir.exists(export_path)) {
    dir.create(dirname(export_path), recursive = TRUE, showWarnings = FALSE)
  }
  
  write.csv(df, export_path, row.names = TRUE)
}


## Stats: ARI, F1 (macro and weighted)
calculate_ARI <- function(sce, predicted) {
  annotation <- colData(sce)$annotation
  pred <- colData(sce)[[predicted]]
  valid <- !is.na(annotation) & !is.na(pred)
  mclust::adjustedRandIndex(annotation[valid], pred[valid])
}


match_clusters_to_labels <- function(true_labels, pred_labels) {
  true_labels <- as.character(true_labels)
  pred_labels <- as.character(pred_labels)
  
  true_classes <- unique(true_labels)
  pred_classes <- unique(pred_labels)
  
  cost_matrix <- matrix(0, nrow = length(pred_classes), ncol = length(true_classes))
  for (i in seq_along(pred_classes)) {
    for (j in seq_along(true_classes)) {
      overlap <- sum(pred_labels == pred_classes[i] & true_labels == true_classes[j])
      cost_matrix[i, j] <- -overlap
    }
  }
  
  # clue::solve_LSAP requires a square matrix and minimizes cost -- pad with
  # zeros (no overlap = worst case for a real assignment, harmless as filler)
  n <- max(nrow(cost_matrix), ncol(cost_matrix))
  padded <- matrix(0, n, n)
  padded[seq_len(nrow(cost_matrix)), seq_len(ncol(cost_matrix))] <- cost_matrix
  # solve_LSAP requires non-negative costs
  padded <- padded - min(padded)
  
  assignment <- clue::solve_LSAP(padded)
  
  mapping <- setNames(rep(NA_character_, length(pred_classes)), pred_classes)
  for (i in seq_along(pred_classes)) {
    j <- assignment[i]
    if (j <= length(true_classes)) {
      mapping[pred_classes[i]] <- true_classes[j]
    }
  }
  # Any predicted cluster not covered by the assignment gets an explicit
  # "unmatched" label instead of silently becoming NA
  mapping[is.na(mapping)] <- "unmatched"
  
  unname(mapping[pred_labels])
}


.f1_multiclass <- function(true, pred, weighted = FALSE) {
  classes <- union(unique(true), unique(pred))
  f1s <- numeric(length(classes))
  support <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    cls <- classes[i]
    tp <- sum(pred == cls & true == cls)
    fp <- sum(pred == cls & true != cls)
    fn <- sum(pred != cls & true == cls)
    precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
    recall <- if (tp + fn == 0) 0 else tp / (tp + fn)
    f1s[i] <- if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
    support[i] <- sum(true == cls)
  }
  
  if (weighted) sum(f1s * support) / sum(support) else mean(f1s)
}


calculate_F1 <- function(sce, predicted) {
  annotation <- as.character(colData(sce)$annotation)
  pred <- as.character(colData(sce)[[predicted]])
  valid <- !is.na(annotation) & !is.na(pred)
  annotation <- annotation[valid]
  pred <- pred[valid]
  
  pred_mapped <- match_clusters_to_labels(annotation, pred)
  
  list(
    f1_macro    = .f1_multiclass(annotation, pred_mapped, weighted = FALSE),
    f1_weighted = .f1_multiclass(annotation, pred_mapped, weighted = TRUE)
  )
}


calculate_stats <- function(sce, model, predicted, tissue, slide, export_path = "stats.csv") {
  ari <- calculate_ARI(sce, predicted)
  f1 <- calculate_F1(sce, predicted)
  
  cat(sprintf(
    "Dataset: %s, %s | ARI: %.2f, F1 macro: %.2f, F1 weighted: %.2f\n",
    tissue, slide, ari, f1$f1_macro, f1$f1_weighted
  ))
  
  new_row <- data.frame(
    Model = model, Dataset = tissue, Slide = slide,
    ARI = ari, F1_Weighted = f1$f1_weighted, F1_Score = f1$f1_macro
  )
  
  if (file.exists(export_path)) {
    existing <- read.csv(export_path)
    updated <- rbind(existing, new_row)
    write.csv(updated, export_path, row.names = FALSE)
  } else {
    write.csv(new_row, export_path, row.names = FALSE)
  }
}