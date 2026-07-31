import sys
import os

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from PIL import Image

from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from pyensembl import EnsemblRelease

# import models
import novae
sys.path.append(os.path.abspath("../../models/STMSGAL"))
import STMSGAL

sys.path.append(os.path.abspath("../../models/HyperGCN"))
from hpLapGCN import hpLapGCN

sys.path.append(os.path.abspath("../../models/SCOIGET"))
# from scoiget.preprocess_utils import generate_adata, preprocess_adata, quality_control, plot_adata_visualizations
from scoiget.cnv_utils import add_genomic_locations, gene_binning_from_adata, perform_clustering, find_normal_cluster, compute_pseudo_copy
# from scoiget.cnv_utils import perform_clustering_gpu, find_normal_cluster_gpu, auto_corr_gpu, kmeans_gpu, silhouette_score_gpu
from scoiget.graph_utils import get_x_bin_data, get_x_bin_data_torch, construct_spatial_knn_graph, compute_edge_weights_and_probabilities
# from scoiget.scoiget_model import SCOIGET, Encoder, CNEncoder, Decoder
from scoiget.train_utils import prepare_data, train_scoiget
from scoiget.cluster_utils import clustering


# loader functions
def load_DLPFC(file):
    adata = sc.read_visium(
        path="../../app/data/Human dorsolateral prefrontal cortex (DLPFC)/" + file,
        count_file=file + "_filtered_feature_bc_matrix.h5",
        load_images=True,
    )
    ground_truth_annotations = pd.read_csv("../../app/data/Human dorsolateral prefrontal cortex (DLPFC)/manual_annotations/" + file + "_true.csv", index_col=0, header=0)
    adata.obs['annotation'] = ground_truth_annotations['label']

    n_clusters = adata.obs['annotation'].nunique()

    return adata, n_clusters

def load_HER2(file):
    # Get counts (ST-cnts)
    counts = pd.read_csv(f"../../app/data/HER2+ breast cancer/ST-cnts/{file}.tsv.gz", sep="\t", index_col=0)
    adata = ad.AnnData(counts)
    adata.obs["key"] = adata.obs_names

    # Get spots (ST-spotfiles)
    spots = pd.read_csv(f"../../app/data/HER2+ breast cancer/ST-spotfiles/{file}_selection.tsv", sep="\t")
    spots["key"] = spots['x'].astype(str) + 'x' + spots['y'].astype(str)
    spots = spots.set_index("key")

    # Keep only spots flagged as under tissue
    selected_keys = spots.index[spots["selected"] == 1]
    adata = adata[adata.obs["key"].isin(selected_keys)].copy()

    # Attach pixel coordinates matched to the spotfile's image (for H&E overlay)
    adata.obs["array_x"] = spots.loc[adata.obs["key"], "x"].values
    adata.obs["array_y"] = spots.loc[adata.obs["key"], "y"].values
    adata.obs["pixel_x"] = spots.loc[adata.obs["key"], "pixel_x"].values
    adata.obs["pixel_y"] = spots.loc[adata.obs["key"], "pixel_y"].values

    # Use PIXEL coordinates for obsm["spatial"] so points align with the H&E image
    adata.obsm["spatial"] = adata.obs[["pixel_x", "pixel_y"]].values

    # Get pathologist ground-truth annotation (ST-pat)
    adata.obs["new_x"] = spots.loc[adata.obs["key"], "new_x"].values
    adata.obs["new_y"] = spots.loc[adata.obs["key"], "new_y"].values

    adata.obs["float_key"] = adata.obs["new_x"].round(3).astype(str) + "x" + adata.obs["new_y"].round(3).astype(str)

    labels = pd.read_csv(f"../../app/data/HER2+ breast cancer/ST-pat/lbl/{file}_labeled_coordinates.tsv", sep="\t", index_col='Row.names')
    labels = labels.dropna(subset=["x", "y"])
    labels["float_key"] = labels["x"].round(3).astype(str) + "x" + labels["y"].round(3).astype(str)
    labels = labels.set_index("float_key")

    adata.obs["annotation"] = labels["label"].reindex(adata.obs["float_key"]).values

    n_clusters = adata.obs['annotation'].nunique()

    # --- Attach H&E image + scalefactors so sc.pl.spatial can use img_key ---
    image_path = f"../../app/data/HER2+ breast cancer/images/HE/{file}.jpg"
    img = np.array(Image.open(image_path))

    # Estimate spot diameter in pixel space: median distance to nearest
    # neighboring spot (a standard trick when no scalefactors file is provided)
    coords = adata.obsm["spatial"]
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)  # k=2: nearest neighbor excluding self
    median_spot_spacing = np.median(dists[:, 1])

    library_id = file
    adata.uns["spatial"] = {
        library_id: {
            "images": {"hires": img},
            "scalefactors": {
                "tissue_hires_scalef": 1.0,   # image is already at full pixel resolution
                "spot_diameter_fullres": median_spot_spacing,
            },
            "metadata": {},
        }
    }

    return adata, n_clusters

def load_MOSTA(file):
    adata = sc.read_h5ad(filename=f"../../app/data/Stereo-seq MOSTA/{file}/{file}.MOSTA.h5ad")
    n_clusters = adata.obs['annotation'].nunique()

    return adata, n_clusters

# STMSGAL helpers
def get_median_nn_distance(coords, k=6):
    """Median distance to the k-th nearest neighbor -- a robust proxy for
    'typical spot spacing' regardless of the dataset's native coordinate units."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)  # +1 to skip self
    distances, _ = nbrs.kneighbors(coords)
    return np.median(distances[:, -1])

def rescale_spatial_coords_to_reference(adata, reference_median_nn_dist, k=6):
    """
    Rescales adata.obsm['spatial'] so its median k-NN spacing matches a
    reference value (e.g. DLPFC's, which the default rad_cutoff=300 was
    tuned against). This makes a fixed rad_cutoff behave consistently
    across platforms with wildly different native coordinate units.
    """
    coords = adata.obsm['spatial'].astype(float)
    current_median = get_median_nn_distance(coords, k=k)

    scale_factor = reference_median_nn_dist / current_median
    adata.obsm['spatial'] = coords * scale_factor

    print(f"Rescaled spatial coords by {scale_factor:.4f}x "
          f"(median NN dist: {current_median:.2f} -> {reference_median_nn_dist:.2f})")

    return adata

# SCOIGET helpers
def ensure_gene_ids(adata, species, release=98):
    """
    Ensures adata.var has a 'gene_ids' column (Ensembl gene IDs), required by
    add_genomic_locations(). 10x Visium data (DLPFC) already has this from
    Space Ranger; other loaders (HER2, MOSTA) need it constructed here by
    mapping gene symbols (adata.var_names) to Ensembl IDs via pyensembl.
    """
    if 'gene_ids' in adata.var.columns:
        return adata

    data = EnsemblRelease(release, species=species)

    def symbol_to_ensembl_id(symbol):
        try:
            ids = data.gene_ids_of_gene_name(symbol)
            return ids[0] if ids else None  # take first match if multiple (rare synonym collisions)
        except ValueError:
            return None  # symbol not found in this Ensembl release

    gene_ids = [symbol_to_ensembl_id(sym) for sym in adata.var_names]
    adata.var['gene_ids'] = gene_ids

    n_missing = sum(g is None for g in gene_ids)
    if n_missing > 0:
        print(f"Warning: {n_missing}/{adata.n_vars} gene symbols could not be mapped to an Ensembl ID")

    return adata

def prepare_scoiget_graph(adata, species, bin_size=10, n_neighbors=5):
    """
    Builds the graph-based inputs SCOIGET's run_SCOIGET() expects:
    obsm['feat'], obsm['graph_neigh'], obsm['edge_probabilities'].
    Call after preprocess_data() and before run_SCOIGET().
    """
    adata = ensure_gene_ids(adata, species)
    adata = add_genomic_locations(adata, species)
    
    print('----Get bin data and feature----')
    adata, _ = gene_binning_from_adata(adata, bin_size)

    adata = get_x_bin_data_torch(adata, bin_size=bin_size)
    adata_binned = adata.uns['binned_data']
    X_bin = adata_binned.obsm['X_bin']
    adata.obsm['feat'] = X_bin

    print('----Construct spatial neighbor graph----')
    construct_spatial_knn_graph(adata, n_neighbors=n_neighbors)

    if 'graph_neigh' not in adata.obsm:
        raise ValueError("graph_neigh not found in adata.obsm. Check `construct_spatial_knn_graph` function.")
    print(f"Number of edges in `graph_neigh`: {adata.obsm['graph_neigh'].count_nonzero()}")

    print('----Compute edge weights and probabilities----')
    compute_edge_weights_and_probabilities(adata, use_norm_x=False)

    edge_prob_key = 'edge_probabilities'
    if edge_prob_key not in adata.obsm:
        raise ValueError(f"{edge_prob_key} not found in adata.obsm. Check `compute_edge_weights_and_probabilities` function.")

    edge_count = adata.obsm[edge_prob_key].count_nonzero()
    graph_edge_count = adata.obsm['graph_neigh'].count_nonzero()
    print(f"Number of edges in `edge_probabilities`: {edge_count}")
    print(f"Number of edges in `graph_neigh`: {graph_edge_count}")
    if edge_count != graph_edge_count:
        raise ValueError("Mismatch between the number of edges in `edge_probabilities` and `graph_neigh`.")

    return adata

# general helpers
def preprocess_data(adata, file):
    if 'in_tissue' in adata.obs.columns.tolist():
        adata = adata[adata.obs['in_tissue'] == 1].copy()
    adata.var_names_make_unique()
    adata.obs['slide_id'] = file

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata

# def res_search_fixed_clus(adata, fixed_clus_count, low=0.01, high=2.5, guess=1, increment=0.02):
#     # original linear search implementation
#     for res in sorted(list(np.arange(low, high, increment)), reverse=True):
#         sc.tl.leiden(adata, random_state=0, resolution=res)
#         count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
#         if count_unique_leiden == fixed_clus_count:
#             break
#     return res

def res_search_fixed_clus(adata, fixed_clus_count, low=0.01, high=2.5, guess=1, increment=0.01, max_iter=50):
    # binary search implementation
    res = guess
    for _ in range(max_iter):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())

        if count_unique_leiden == fixed_clus_count:
            return res

        if count_unique_leiden < fixed_clus_count:
            low = res
        else:
            high = res

        if high - low < increment:
            break

        res = (low + high) / 2

    return res


# run model functions
def run_hyperGCN(adata, level,
                 n_epochs=300,
                  k=50, feat_hidden1=20, feat_hidden2=11,
                  gcn_hidden1=20, gcn_hidden2=11, p_drop=0.2,
                  using_dec=True, using_mask=False, feat_w=10, clu=0.1,
                  gcn_lr=0.01, gcn_decay=0.01, dec_interval=20, dec_tol=0.0,
                  cell_feat_dim=300, eval_graph_n=20, export_path=None):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    class Params:
        pass
    params = Params()
    params.device = device
    params.k = k
    params.epochs = n_epochs
    params.feat_hidden1 = feat_hidden1
    params.feat_hidden2 = feat_hidden2
    params.gcn_hidden1 = gcn_hidden1
    params.gcn_hidden2 = gcn_hidden2
    params.p_drop = p_drop
    params.using_dec = using_dec
    params.using_mask = using_mask
    params.feat_w = feat_w
    params.clu = clu
    params.gcn_lr = gcn_lr
    params.gcn_decay = gcn_decay
    params.dec_interval = dec_interval
    params.dec_tol = dec_tol
    params.dec_cluster_n = level

    # Preprocess expression: PCA-reduce to cell_feat_dim
    # (adapted from adata_preprocess)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_scaled = sc.pp.scale(X, zero_center=True, max_value=10, copy=True)
    n_components = min(cell_feat_dim, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    adata_X = pca.fit_transform(X_scaled)

    params.cell_feat_dim = adata_X.shape[1]

    # Build spatial graph
    spatial_co = adata.obsm['spatial']
    adj = kneighbors_graph(spatial_co, k, mode="connectivity",
                            metric="euclidean", include_self=True, n_jobs=-1)
    adj_hp = torch.tensor(adj.toarray().astype(np.float32))
    graph_dict = {"spatial": spatial_co, "adj_norm": adj_hp}

    # Train
    sedr_net = hpLapGCN(adata_X, graph_dict, params)
    if params.using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    sedr_feat, _, _, _ = sedr_net.process()

    # Wrap result back into an AnnData-compatible representation
    adata.obsm['HyperGCN'] = sedr_feat

    sc.pp.neighbors(adata, use_rep='HyperGCN', n_neighbors=eval_graph_n)
    sc.tl.umap(adata)

    eval_resolution = res_search_fixed_clus(adata, level)
    sc.tl.leiden(adata, resolution=eval_resolution)

    # Rename predicted column
    predicted_key = f"hypergcn_leiden_{level}"
    adata.obs[predicted_key] = adata.obs['leiden']

    if export_path is not None:
        save_annotations(adata, predicted_key, export_path)

    return adata, predicted_key

def run_STMSGAL(adata, level, 
                rad_cutoff=300, alpha=0.7, pre_resolution=0.2, n_epochs=100, dsc_alpha=0.35,
                cost_ssc_coef=0.1, n_top_genes=1000, export_path=None):

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)

    #Constructing the spatial network
    STMSGAL.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    STMSGAL.Stats_Spatial_Net(adata)

    # Train
    adata, pred_dsc = STMSGAL.train_STMSGAL(
        adata, alpha=alpha, pre_resolution=pre_resolution,
        n_epochs=n_epochs, save_attention=True, save_loss=False,
        n_cluster=level, cost_ssc_coef=cost_ssc_coef
    )

    sc.pp.neighbors(adata, use_rep='STMSGAL')
    sc.tl.umap(adata)

    eval_resolution = res_search_fixed_clus(adata, level)
    sc.tl.leiden(adata, resolution=eval_resolution)

    # Rename predicted column
    predicted_key = f"stmsgal_leiden_{level}"
    adata.obs[predicted_key] = adata.obs['leiden']

    if export_path is not None:
        save_annotations(adata, predicted_key, export_path)

    return adata, predicted_key

def run_SCOIGET(adata, file, level,
                 intermediate_dim=128, latent_dim=32, max_cp=15,
                 kl_weights_stage1=0.1, kl_weights_stage2=0.5,
                 epochs_stage1=100, epochs_stage2=100, lr=0.001,
                 lambda_smooth_stage1=0.1, lambda_smooth_stage2=1,
                 dropout=0.2, hmm_states=3, gnn_heads=8,
                 output_dir="./output_dir", export_path=None):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Stage 1: train on raw features ---
    data = prepare_data(adata, use_norm_x=False)

    model_stage1 = train_scoiget(
        data,
        original_dim=data.x.shape[1],
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        max_cp=max_cp,
        kl_weights=kl_weights_stage1,
        epochs=epochs_stage1,
        lr=lr,
        lambda_smooth=lambda_smooth_stage1,
        use_mini_batch=False,
        dropout=dropout,
        hmm_states=hmm_states,
        gnn_heads=gnn_heads,
        device=device,
        save_path=output_dir
    )
    torch.save(model_stage1.state_dict(), os.path.join(output_dir, f"{file}_model_1st.pth"))

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    model_stage1 = model_stage1.to(device)

    with torch.no_grad():
        z_mean, z_var, latent_z = model_stage1.z_encoder(data.x, data.edge_index)
        reconstructed_features = model_stage1.decoder(latent_z)
        pseudo_copy_number, _ = model_stage1.encoder([data.x, reconstructed_features], data.edge_index)

    pseudo_copy_number = pseudo_copy_number.detach().cpu().numpy()
    scaling_factor = pseudo_copy_number.mean()
    adata.obsm['norm_x'] = pseudo_copy_number / scaling_factor

    # --- Rebuild graph using pseudo copy-number signal ---
    compute_edge_weights_and_probabilities(adata, use_norm_x=True)
    data_norm = prepare_data(adata, use_norm_x=True)

    # --- Stage 2: retrain on refined graph ---
    scoiget_model = train_scoiget(
        data_norm,
        original_dim=data_norm.x.shape[1],
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        max_cp=max_cp,
        kl_weights=kl_weights_stage2,
        epochs=epochs_stage2,
        lr=lr,
        lambda_smooth=lambda_smooth_stage2,
        use_mini_batch=False,
        validation_split=0.2,
        dropout=dropout,
        hmm_states=hmm_states,
        gnn_heads=gnn_heads,
        device=device,
    )
    model_path = os.path.join(output_dir, f"{file}_model_2nd.pth")
    torch.save(scoiget_model.state_dict(), model_path)

    scoiget_model.eval()
    data_norm.x = data_norm.x.to(device)
    data_norm.edge_index = data_norm.edge_index.to(device)
    data_norm.edge_attr = data_norm.edge_attr.to(device)
    scoiget_model = scoiget_model.to(device)

    with torch.no_grad():
        z_mean, z_var, latent_z = scoiget_model.z_encoder(data_norm.x, data_norm.edge_index)
        reconstructed_features = scoiget_model.decoder(latent_z)
        copy_number_profile, _ = scoiget_model.encoder([data_norm.x, reconstructed_features], data_norm.edge_index)

    adata.obsm['latent'] = latent_z.cpu().numpy()
    adata.obsm['copy_number'] = copy_number_profile.cpu().numpy()
    adata.obs['copy_number_mean'] = copy_number_profile.cpu().numpy().mean(axis=1)

    # PCA on copy number as an alternative representation
    pca = PCA(svd_solver='arpack')
    adata.obsm['X_pca'] = pca.fit_transform(adata.obsm['copy_number'])

    # --- Cluster directly targeting the ground-truth cluster count ---
    adata = clustering(
        adata,
        method='leiden',
        auto_choose=False,
        n_clusters=level,
        refinement=False,
        use_rep="copy_number"  # swap to "latent" or "X_pca" if you want to compare representations
    )

    predicted_key = f"scoiget_leiden_{level}"
    adata.obs[predicted_key] = adata.obs['leiden']

    if export_path is not None:
        save_annotations(adata, predicted_key, export_path)

    return adata, predicted_key

def run_novae(adata, level, technology=None, export_path=None):
    novae.spatial_neighbors(adata, technology=technology, slide_key="slide_id")
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var["highly_variable"]].copy()

    model = novae.Novae(adata, n_hops_local=1, n_hops_view=1, panel_subset_size=0.6)
    model.fit()

    model.compute_representations()
    model.assign_domains(level=level)

    predicted_key = f"novae_domains_{level}"
    if export_path is not None:
        save_annotations(adata, predicted_key, export_path)

    return adata, predicted_key

# visualization functions
def plot_annotations(adata, model, level, predicted, tissue, slide, export_path):
    sc.pl.embedding(adata, basis="spatial", 
                    color=["annotation", predicted], 
                    title=[f"Ground truth (tissue={tissue}, slide={slide}, n={level})",
                           f"Predicted (model={model})"],
                    save=export_path,
                           )

def save_annotations(adata, predicted, export_path):
    df = adata.obs[["annotation", predicted]].copy()
    spatial_coords = adata.obsm["spatial"]
    df["x"] = spatial_coords[:, 0]
    df["y"] = spatial_coords[:, 1]

    df.to_csv(export_path, index=True)

def get_spatial_plot_kwargs(adata):
    has_image = (
        "spatial" in adata.uns
        and isinstance(adata.uns["spatial"], dict)
        and len(adata.uns["spatial"]) > 0
    )
    if has_image:
        return {"img_key": "hires", "alpha": 0.7}
    else:
        # No H&E image / scalefactors available (e.g. HER2+ loaded from raw counts+coords)
        # Must supply spot_size manually since scanpy can't infer it.
        return {"img_key": None, "spot_size": 1.0, "alpha": 1.0}

# stats functions
def calculate_ARI(adata, predicted):
    # Drop rows where either label is missing
    valid = adata.obs['annotation'].notna() & adata.obs[predicted].notna()

    ari = adjusted_rand_score(
        adata.obs['annotation'][valid],
        adata.obs[predicted][valid]
    )

    return ari

def match_clusters_to_labels(true_labels, pred_labels):
    true_labels = pd.Series(true_labels).astype(str)
    pred_labels = pd.Series(pred_labels).astype(str)

    true_classes = true_labels.unique()
    pred_classes = pred_labels.unique()

    cost_matrix = np.zeros((len(pred_classes), len(true_classes)))
    for i, p in enumerate(pred_classes):
        for j, t in enumerate(true_classes):
            overlap = ((pred_labels == p) & (true_labels == t)).sum()
            cost_matrix[i, j] = -overlap

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {pred_classes[r]: true_classes[c] for r, c in zip(row_ind, col_ind)}

    # Any predicted cluster not covered by the assignment gets an explicit
    # "unmatched" label instead of silently becoming NaN
    for p in pred_classes:
        if p not in mapping:
            mapping[p] = "unmatched"

    pred_mapped = pred_labels.map(mapping)
    return pred_mapped

def calculate_F1(adata, predicted):
    valid = adata.obs['annotation'].notna() & adata.obs[predicted].notna()

    pred_mapped = match_clusters_to_labels(
        adata.obs['annotation'][valid],
        adata.obs[predicted][valid]
    )

    f1_macro = f1_score(adata.obs['annotation'][valid], pred_mapped, average='macro')
    f1_weighted = f1_score(adata.obs['annotation'][valid], pred_mapped, average='weighted')

    return f1_macro, f1_weighted

def calculate_stats(adata, model, predicted, tissue, slide, run_time, export_path="stats.csv"):
    ari = calculate_ARI(adata, predicted=predicted)
    f1_macro, f1_weighted = calculate_F1(adata, predicted=predicted)
    # print(f"Dataset: {tissue}, {slide} | ARI: {ari:.2f}")
    print(f"Dataset: {tissue}, {slide} | ARI: {ari:.2f}, F1 macro: {f1_macro:.2f}, F1 weighted: {f1_weighted:.2f}")

    new_data = {
        'Model': [model],
        'Dataset': [tissue],
        'Slide': [slide],
        'ARI': [ari],
        'F1_Weighted': [f1_weighted],
        'F1_Score': [f1_macro],
        'Run time': [run_time],
    }
    new_df = pd.DataFrame(new_data)
    
    if os.path.exists(export_path):
        existing_df = pd.read_csv(export_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(export_path, index=False)
    else:
        new_df.to_csv(export_path, index=False)
