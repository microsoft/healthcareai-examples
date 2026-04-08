"""Helper utilities for the GigaTIME virtual phenotyping notebook.

This module centralizes sample loading, prediction post-processing,
cell-level localization, count comparison, and visualization helpers.
"""

from pathlib import Path
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image
from skimage.measure import label, regionprops

CHANNEL_NAMES = [
    "DAPI",
    "TRITC",
    "Cy5",
    "PD-1",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
]

NUCLEAR_CHANNELS = {"DAPI", "TRITC", "Cy5", "Ki67"}
EXCLUDED_CHANNELS = {"TRITC", "Cy5"}
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHANNEL_THRESHOLDS = {"Ki67": 0.2}
DEFAULT_VISUALIZATION_MARKERS = ["DAPI", "CD3", "CD8", "CD68", "PD-L1", "CK", "Ki67"]

BIOLOGICAL_MARKER_NOTES = {
    "DAPI": "Nuclear stain surrogate used as the localization anchor.",
    "CD3": "Pan-T-cell associated marker.",
    "CD4": "Helper T-cell associated marker.",
    "CD8": "Cytotoxic T-cell associated marker.",
    "CD68": "Macrophage-associated marker.",
    "PD-1": "Immune checkpoint receptor-associated marker.",
    "PD-L1": "Immune checkpoint ligand-associated marker.",
    "CK": "Epithelial and tumor-associated marker context.",
    "Ki67": "Proliferation-associated nuclear marker.",
}

SAMPLE_PAIR_NAMES = [
    "10008_36140_556_556",
    "11120_23908_556_556",
    "10564_25576_556_556",
    "0_16680_556_556",
    "10008_37808_556_556",
    "10008_35584_556_556",
    "10008_38920_556_556",
    "10008_6116_556_556",
    "10008_21684_556_556",
    "10008_21128_556_556",
    "11120_22796_556_556",
    "10564_26688_556_556",
    "0_30024_556_556",
    "10008_24464_556_556",
    "10008_18904_556_556",
    "10564_3336_556_556",
    "10564_33916_556_556",
    "0_27800_556_556",
    "10564_36696_556_556",
    "0_18348_556_556",
    "10564_7228_556_556",
    "10564_12232_556_556",
    "10008_36696_556_556",
    "11120_17236_556_556",
    "0_7228_556_556",
    "0_26132_556_556",
    "10564_6116_556_556",
    "10564_38920_556_556",
    "0_16124_556_556",
    "10564_6672_556_556",
    "0_10564_556_556",
    "10008_16680_556_556",
    "0_6116_556_556",
    "10008_25576_556_556",
    "0_15568_556_556",
    "10008_29468_556_556",
    "10008_34472_556_556",
    "10008_27244_556_556",
    "10008_35028_556_556",
    "10564_29468_556_556",
    "11120_18904_556_556",
    "10008_3336_556_556",
    "0_4448_556_556",
    "0_556_556_556",
    "0_6672_556_556",
    "0_2780_556_556",
    "10008_6672_556_556",
    "0_24464_556_556",
    "0_37252_556_556",
    "0_5560_556_556",
]


def load_metadata(sample_root):
    """Load sample metadata from helper constants and attach the shared data directory."""

    sample_root = Path(sample_root)
    data_root = sample_root / "data"

    if not data_root.exists():
        raise FileNotFoundError(f"Sample data directory not found: {data_root}")

    metadata_df = pd.DataFrame({"pair_name": SAMPLE_PAIR_NAMES})
    metadata_df["dir_name"] = [data_root] * len(metadata_df)
    return metadata_df


def load_indexed_image_files(metadata_df):
    """Return metadata rows as indexed H&E image paths."""

    return [
        (idx, str(Path(row["dir_name"]) / f"{row['pair_name']}_he.png"))
        for idx, row in metadata_df.iterrows()
    ]


def load_saved_predictions(metadata_df):
    """Load previously saved prediction arrays keyed by dataframe index."""

    prediction_by_index = {}
    for idx, row in metadata_df.iterrows():
        pred_path = Path(row["dir_name"]) / f"{row['pair_name']}_pred.npy"
        if pred_path.exists():
            prediction_by_index[idx] = np.load(pred_path)
    return prediction_by_index


def unpack_and_load_mask(mask_path):
    with gzip.open(mask_path, "rb") as file_handle:
        data = pickle.load(file_handle)

    packed_mask = data["comet_array_binary"]
    mask_shape = data["original_shape"]
    last_dim = data["original_last_dim"]
    unpacked_mask = np.unpackbits(packed_mask, axis=-1)
    data["comet_array_binary"] = unpacked_mask[..., :last_dim].reshape(mask_shape)
    return data


def load_he_image(image_path):
    return np.asarray(Image.open(image_path).convert("RGB"))


def ensure_channel_first_mask(mask, channel_count=None):
    channel_count = channel_count or len(CHANNEL_NAMES)
    mask = np.asarray(mask, dtype="float32")
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, received shape {mask.shape}")

    if mask.shape[0] == channel_count:
        return mask
    if mask.shape[-1] == channel_count:
        return mask.transpose(2, 0, 1)

    raise ValueError(
        f"Unable to infer channel dimension for mask shape {mask.shape} and {channel_count} channels"
    )


def resize_channel_first_mask(mask, target_shape, resample=Image.BILINEAR):
    mask = ensure_channel_first_mask(mask, len(CHANNEL_NAMES))
    target_height, target_width = target_shape
    resized_channels = []

    for channel in mask:
        channel_image = Image.fromarray(channel.astype("float32"))
        resized_channels.append(
            np.asarray(
                channel_image.resize((target_width, target_height), resample=resample),
                dtype="float32",
            )
        )

    return np.stack(resized_channels, axis=0)


def binarize_prediction_mask(probability_mask, threshold=DEFAULT_THRESHOLD):
    probability_mask = ensure_channel_first_mask(probability_mask, len(CHANNEL_NAMES))
    return (probability_mask >= threshold).astype("float32")


def localize_marker_in_single_cell(
    binary_mask,
    labels_dapi,
    labels_dapi_expanded,
    channel_names=CHANNEL_NAMES,
    nuclear_channels=NUCLEAR_CHANNELS,
    default_threshold=DEFAULT_THRESHOLD,
    channel_thresholds=None,
):
    """Convert marker masks into cell-aware localized masks."""

    binary_mask = ensure_channel_first_mask(binary_mask, len(channel_names)).transpose(
        1, 2, 0
    )
    thresholds = DEFAULT_CHANNEL_THRESHOLDS.copy()
    if channel_thresholds:
        thresholds.update(channel_thresholds)

    cell_masks = np.stack(
        [
            labels_dapi if channel in nuclear_channels else labels_dapi_expanded
            for channel in channel_names
        ],
        axis=-1,
    )
    masked = binary_mask.copy()
    masked[cell_masks == 0] = 0

    labeled_regions = (
        ("nuclei", regionprops(label(labels_dapi))),
        ("cell", regionprops(label(labels_dapi_expanded))),
    )
    refined_mask = np.zeros_like(masked)

    for region_type, regions in labeled_regions:
        for region in regions:
            region_mask = region.convex_image
            min_row, min_col, max_row, max_col = region.bbox
            mask_bbox = masked[min_row:max_row, min_col:max_col, :]
            region_area = region_mask.sum()
            if region_area == 0:
                continue

            region_ratios = mask_bbox[region_mask].sum(axis=0) / region_area
            for channel_idx, (channel_name, region_ratio) in enumerate(
                zip(channel_names, region_ratios)
            ):
                is_nuclear_channel = channel_name in nuclear_channels
                is_valid_channel = (
                    is_nuclear_channel
                    if region_type == "nuclei"
                    else not is_nuclear_channel
                )
                threshold = thresholds.get(channel_name, default_threshold)
                if is_valid_channel and region_ratio > threshold:
                    refined_mask[min_row:max_row, min_col:max_col, channel_idx][
                        region_mask
                    ] = 1

    return refined_mask.astype("float32").transpose(2, 0, 1)


def count_labeled_objects(label_mask):
    unique_labels = np.unique(np.asarray(label_mask))
    return int(np.count_nonzero(unique_labels))


def count_localized_cells_per_marker(
    localized_mask,
    labels_dapi,
    labels_dapi_expanded,
    channel_names=CHANNEL_NAMES,
    nuclear_channels=NUCLEAR_CHANNELS,
):
    localized_mask = ensure_channel_first_mask(localized_mask, len(channel_names))
    nuclei_regions = regionprops(label(labels_dapi))
    cell_regions = regionprops(label(labels_dapi_expanded))
    marker_counts = {}

    for channel_idx, channel_name in enumerate(channel_names):
        regions = nuclei_regions if channel_name in nuclear_channels else cell_regions
        positive_count = 0
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            region_mask = region.image
            region_view = localized_mask[channel_idx, min_row:max_row, min_col:max_col]
            if np.any(region_view[region_mask] > 0):
                positive_count += 1
        marker_counts[channel_name] = positive_count

    return marker_counts


def build_sample_record(
    pair_name,
    row,
    prediction_probability_mask=None,
    channel_names=CHANNEL_NAMES,
    nuclear_channels=NUCLEAR_CHANNELS,
    prediction_threshold=DEFAULT_THRESHOLD,
    channel_thresholds=None,
):
    """Build one unified sample record for plotting and comparison."""

    pair_dir = Path(row["dir_name"])
    image_path = pair_dir / f"{pair_name}_he.png"
    mask_path = pair_dir / f"{pair_name}_comet_binary_thres_labels.pkl.gz"
    pred_path = pair_dir / f"{pair_name}_pred.npy"

    if not image_path.exists():
        raise FileNotFoundError(f"H&E image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")

    he_image = load_he_image(image_path)
    pkl_data = unpack_and_load_mask(mask_path)
    labels_dapi = pkl_data["labels_dapi"]
    labels_dapi_expanded = pkl_data["labels_dapi_expanded"]
    target_shape = labels_dapi.shape

    ground_truth_binary_mask = ensure_channel_first_mask(
        pkl_data["comet_array_binary"],
        len(channel_names),
    )
    ground_truth_localized_mask = localize_marker_in_single_cell(
        ground_truth_binary_mask,
        labels_dapi,
        labels_dapi_expanded,
        channel_names=channel_names,
        nuclear_channels=nuclear_channels,
        channel_thresholds=channel_thresholds,
    )
    ground_truth_marker_counts = count_localized_cells_per_marker(
        ground_truth_localized_mask,
        labels_dapi,
        labels_dapi_expanded,
        channel_names=channel_names,
        nuclear_channels=nuclear_channels,
    )
    ground_truth_marker_counts["DAPI"] = count_labeled_objects(labels_dapi)

    if prediction_probability_mask is not None:
        prediction_probability_mask = resize_channel_first_mask(
            prediction_probability_mask,
            target_shape=target_shape,
            resample=Image.BILINEAR,
        )
        prediction_binary_mask = binarize_prediction_mask(
            prediction_probability_mask,
            threshold=prediction_threshold,
        )
        prediction_localized_mask = localize_marker_in_single_cell(
            prediction_binary_mask,
            labels_dapi,
            labels_dapi_expanded,
            channel_names=channel_names,
            nuclear_channels=nuclear_channels,
            channel_thresholds=channel_thresholds,
        )
        prediction_marker_counts = count_localized_cells_per_marker(
            prediction_localized_mask,
            labels_dapi,
            labels_dapi_expanded,
            channel_names=channel_names,
            nuclear_channels=nuclear_channels,
        )
    else:
        prediction_binary_mask = None
        prediction_localized_mask = None
        prediction_marker_counts = None

    return {
        "pair_name": pair_name,
        "metadata": row.to_dict(),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "prediction_path": str(pred_path),
        "he_image": he_image,
        "labels_dapi": labels_dapi,
        "labels_dapi_expanded": labels_dapi_expanded,
        "ground_truth_binary_mask": ground_truth_binary_mask,
        "prediction_probability_mask": prediction_probability_mask,
        "prediction_binary_mask": prediction_binary_mask,
        "ground_truth_localized_mask": ground_truth_localized_mask,
        "prediction_localized_mask": prediction_localized_mask,
        "ground_truth_marker_counts": ground_truth_marker_counts,
        "prediction_marker_counts": prediction_marker_counts,
    }


def load_sample_records(
    metadata_df,
    prediction_map=None,
    channel_names=CHANNEL_NAMES,
    nuclear_channels=NUCLEAR_CHANNELS,
    prediction_threshold=DEFAULT_THRESHOLD,
    channel_thresholds=None,
):
    """Build sample records for all rows in the metadata table."""

    prediction_map = {} if prediction_map is None else prediction_map
    sample_records = {}

    for idx, row in metadata_df.iterrows():
        pair_name = row["pair_name"]
        sample_records[pair_name] = build_sample_record(
            pair_name=pair_name,
            row=row,
            prediction_probability_mask=prediction_map.get(idx),
            channel_names=channel_names,
            nuclear_channels=nuclear_channels,
            prediction_threshold=prediction_threshold,
            channel_thresholds=channel_thresholds,
        )

    return sample_records


def get_sample_record(sample, sample_records, metadata_df):
    """Resolve a sample index or pair name to its sample record."""

    if isinstance(sample, int):
        pair_name = metadata_df.iloc[sample]["pair_name"]
    else:
        pair_name = str(sample).strip()
        if pair_name not in set(metadata_df["pair_name"]):
            raise KeyError(f"Sample not found: {pair_name}")

    if pair_name not in sample_records:
        raise KeyError(f"Sample record missing for {pair_name}")

    return sample_records[pair_name]


def get_analysis_channels(channel_names=CHANNEL_NAMES):
    return [
        (idx, channel_name, channel_name.split(" - ")[0].strip())
        for idx, channel_name in enumerate(channel_names)
        if channel_name not in EXCLUDED_CHANNELS
    ]


def resolve_marker_index(marker, channel_names=CHANNEL_NAMES):
    analysis_channels = get_analysis_channels(channel_names)
    if marker is None:
        marker_idx, marker_name, _ = analysis_channels[0]
        return marker_idx, marker_name

    if isinstance(marker, int):
        return marker, channel_names[marker]

    marker_name = str(marker).strip()
    normalized_names = {
        channel_name: idx for idx, channel_name in enumerate(channel_names)
    }
    normalized_names.update(
        {
            channel_name.split(" - ")[0].strip(): idx
            for idx, channel_name in enumerate(channel_names)
        }
    )
    if marker_name not in normalized_names:
        raise KeyError(f"Marker not found: {marker_name}")

    marker_idx = normalized_names[marker_name]
    return marker_idx, channel_names[marker_idx]


def describe_markers(markers):
    """Return a small table with high-level marker interpretations."""

    if isinstance(markers, (str, int)):
        markers = [markers]

    resolved_markers = [resolve_marker_index(marker) for marker in markers]
    rows = []
    for _, marker_name in resolved_markers:
        rows.append(
            {
                "marker": marker_name,
                "interpretation": BIOLOGICAL_MARKER_NOTES.get(
                    marker_name,
                    "High-level exploratory marker context is not defined for this channel.",
                ),
            }
        )

    return pd.DataFrame(rows)


def build_marker_count_comparison_df(
    ground_truth_counts, prediction_counts, channel_names=CHANNEL_NAMES
):
    analysis_channel_names = [
        channel_name for _, channel_name, _ in get_analysis_channels(channel_names)
    ]
    comparison_df = pd.DataFrame(
        {
            "marker": analysis_channel_names,
            "ground_truth_count": [
                ground_truth_counts[channel_name]
                for channel_name in analysis_channel_names
            ],
            "prediction_count": [
                prediction_counts[channel_name]
                for channel_name in analysis_channel_names
            ],
        }
    )
    comparison_df["count_delta"] = (
        comparison_df["prediction_count"] - comparison_df["ground_truth_count"]
    )
    return comparison_df


def get_marker_count_comparison(
    sample, sample_records, metadata_df, channel_names=CHANNEL_NAMES
):
    """Compare localized marker counts for one sample."""

    sample_record = get_sample_record(sample, sample_records, metadata_df)
    prediction_counts = sample_record["prediction_marker_counts"]
    ground_truth_counts = sample_record["ground_truth_marker_counts"]
    if prediction_counts is None or ground_truth_counts is None:
        raise ValueError(
            "Prediction and ground-truth counts are both required for comparison"
        )

    return build_marker_count_comparison_df(
        ground_truth_counts=ground_truth_counts,
        prediction_counts=prediction_counts,
        channel_names=channel_names,
    )


def get_total_marker_count_comparison(sample_records, channel_names=CHANNEL_NAMES):
    """Aggregate localized marker counts across all comparable samples."""

    analysis_channel_names = [
        channel_name for _, channel_name, _ in get_analysis_channels(channel_names)
    ]
    total_ground_truth_counts = {
        channel_name: 0 for channel_name in analysis_channel_names
    }
    total_prediction_counts = {
        channel_name: 0 for channel_name in analysis_channel_names
    }
    compared_samples = 0

    for sample_record in sample_records.values():
        prediction_counts = sample_record["prediction_marker_counts"]
        ground_truth_counts = sample_record["ground_truth_marker_counts"]
        if prediction_counts is None or ground_truth_counts is None:
            continue

        compared_samples += 1
        for channel_name in analysis_channel_names:
            total_ground_truth_counts[channel_name] += ground_truth_counts[channel_name]
            total_prediction_counts[channel_name] += prediction_counts[channel_name]

    comparison_df = build_marker_count_comparison_df(
        ground_truth_counts=total_ground_truth_counts,
        prediction_counts=total_prediction_counts,
        channel_names=channel_names,
    )
    comparison_df.attrs["samples_compared"] = compared_samples
    return comparison_df


def plot_virtual_mif_for_markers(
    sample,
    markers,
    sample_records,
    metadata_df,
    channel_names=CHANNEL_NAMES,
    cmap="magma",
):
    """Plot the H&E patch beside selected virtual mIF channels."""

    sample_record = get_sample_record(sample, sample_records, metadata_df)
    prediction_probability_mask = sample_record["prediction_probability_mask"]
    if prediction_probability_mask is None:
        raise ValueError(
            "Prediction probabilities are required for virtual mIF plotting"
        )

    if isinstance(markers, (str, int)):
        markers = [markers]

    resolved_markers = [
        resolve_marker_index(marker, channel_names) for marker in markers
    ]
    n_cols = len(resolved_markers) + 1
    fig, axes = plt.subplots(
        1, n_cols, figsize=(4 * n_cols, 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    axes[0].imshow(sample_record["he_image"])
    axes[0].set_title(f"H&E\n{sample_record['pair_name']}")
    axes[0].axis("off")

    for plot_idx, (marker_idx, marker_name) in enumerate(resolved_markers, start=1):
        axes[plot_idx].imshow(prediction_probability_mask[marker_idx], cmap=cmap)
        axes[plot_idx].set_title(f"Virtual mIF\n{marker_name}")
        axes[plot_idx].axis("off")

    plt.show()


def build_red_overlay(mask_2d, alpha=0.45):
    binary_mask = np.asarray(mask_2d) > 0
    overlay = np.zeros(binary_mask.shape + (4,), dtype="float32")
    overlay[..., 0] = binary_mask.astype("float32")
    overlay[..., 3] = binary_mask.astype("float32") * alpha
    return overlay


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def get_tissue_bbox(he_image, white_threshold=235, padding=24):
    """Estimate a padded tissue bounding box from an H&E image."""

    tissue_mask = np.any(he_image < white_threshold, axis=-1)
    if not tissue_mask.any():
        return 0, he_image.shape[0], 0, he_image.shape[1]

    row_indices, col_indices = np.where(tissue_mask)
    row_start = max(int(row_indices.min()) - padding, 0)
    row_end = min(int(row_indices.max()) + padding + 1, he_image.shape[0])
    col_start = max(int(col_indices.min()) - padding, 0)
    col_end = min(int(col_indices.max()) + padding + 1, he_image.shape[1])
    return row_start, row_end, col_start, col_end


def add_context_overlay(
    axis,
    he_image,
    localized_mask,
    marker_entries,
    bbox,
    channel_names=CHANNEL_NAMES,
    alpha=0.45,
):
    row_start, row_end, col_start, col_end = bbox
    axis.imshow(he_image[row_start:row_end, col_start:col_end])
    legend_handles = []

    for marker_name, color_hex, legend_label in marker_entries:
        marker_idx, _ = resolve_marker_index(marker_name, channel_names)
        marker_mask = (
            localized_mask[marker_idx, row_start:row_end, col_start:col_end] > 0
        )
        overlay = np.zeros(marker_mask.shape + (4,), dtype="float32")
        overlay[..., :3] = hex_to_rgb01(color_hex)
        overlay[..., 3] = marker_mask.astype("float32") * alpha
        axis.imshow(overlay)
        legend_handles.append(
            Patch(facecolor=color_hex, edgecolor="none", label=legend_label)
        )

    axis.axis("off")
    return legend_handles


def render_context_theme(
    theme,
    context_record,
    localized_mask,
    bbox,
    pair_name,
    channel_names=CHANNEL_NAMES,
):
    """Render one themed H&E context figure with marker overlays."""

    panel_count = len(theme["panels"])
    total_columns = panel_count + 1
    layout = theme.get("layout")

    if layout == "single-row" or panel_count != 3:
        fig, axes = plt.subplots(1, total_columns, figsize=(4.6 * total_columns, 5.2))
        axes = np.atleast_1d(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10.8))
        axes = axes.flatten()

    fig.subplots_adjust(
        top=0.78, left=0.04, right=0.99, bottom=0.05, wspace=0.06, hspace=0.16
    )

    row_start, row_end, col_start, col_end = bbox
    axes[0].imshow(context_record["he_image"][row_start:row_end, col_start:col_end])
    axes[0].set_title("H&E overview", fontsize=13, fontweight="bold", loc="left", pad=8)
    axes[0].axis("off")

    for axis, panel in zip(axes[1:], theme["panels"]):
        legend_handles = add_context_overlay(
            axis=axis,
            he_image=context_record["he_image"],
            localized_mask=localized_mask,
            marker_entries=panel["markers"],
            bbox=bbox,
            channel_names=channel_names,
        )
        axis.set_title(
            panel["title"],
            fontsize=13,
            fontweight="bold",
            loc="left",
            pad=8,
        )
        axis.legend(
            handles=legend_handles,
            loc="lower left",
            frameon=True,
            framealpha=0.9,
            fontsize=8,
            handlelength=1.2,
            borderpad=0.3,
            labelspacing=0.3,
        )

    for axis in axes[len(theme["panels"]) + 1 :]:
        axis.axis("off")

    fig.suptitle(
        theme["heading"],
        fontsize=18,
        fontweight="bold",
        x=0.04,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.04,
        0.945,
        theme["description"],
        fontsize=10,
        ha="left",
        va="top",
    )
    fig.text(
        0.04,
        0.912,
        f"Sample: {pair_name}",
        fontsize=9,
        color="#555555",
        ha="left",
        va="top",
    )

    plt.show()


def plot_overlay_comparison(
    sample,
    sample_records,
    metadata_df,
    channel_names=CHANNEL_NAMES,
    markers=None,
    prediction_mask_key="prediction_localized_mask",
    ground_truth_mask_key="ground_truth_localized_mask",
    alpha=0.45,
):
    """Overlay predicted and reference localized masks on the H&E patch."""

    sample_record = get_sample_record(sample, sample_records, metadata_df)
    prediction_mask = sample_record[prediction_mask_key]
    ground_truth_mask = sample_record[ground_truth_mask_key]
    if prediction_mask is None or ground_truth_mask is None:
        raise ValueError(
            "Prediction and ground-truth masks are both required for overlay comparison"
        )

    if markers is None:
        markers = DEFAULT_VISUALIZATION_MARKERS
    if isinstance(markers, (str, int)):
        markers = [markers]

    resolved_markers = [
        resolve_marker_index(marker, channel_names) for marker in markers
    ]
    comparison_df = get_marker_count_comparison(
        sample=sample,
        sample_records=sample_records,
        metadata_df=metadata_df,
        channel_names=channel_names,
    ).set_index("marker")
    he_image = sample_record["he_image"]

    fig, axes = plt.subplots(
        len(resolved_markers),
        3,
        figsize=(10, max(4, 3.2 * len(resolved_markers))),
        gridspec_kw={"wspace": 0.02, "hspace": 0.24},
    )
    axes = np.atleast_2d(axes)

    for row_idx, (marker_idx, marker_name) in enumerate(resolved_markers):
        ground_truth_count = int(comparison_df.loc[marker_name, "ground_truth_count"])
        prediction_count = int(comparison_df.loc[marker_name, "prediction_count"])

        if marker_name == "DAPI":
            ground_truth_overlay = build_red_overlay(
                sample_record["labels_dapi"], alpha=alpha
            )
        else:
            ground_truth_overlay = build_red_overlay(
                ground_truth_mask[marker_idx], alpha=alpha
            )
        prediction_overlay = build_red_overlay(prediction_mask[marker_idx], alpha=alpha)

        he_axis = axes[row_idx, 0]
        ground_truth_axis = axes[row_idx, 1]
        prediction_axis = axes[row_idx, 2]

        if row_idx == 0:
            he_axis.imshow(he_image)
            he_axis.set_title(f"H&E patch\n{sample_record['pair_name']}")
        he_axis.axis("off")

        ground_truth_axis.imshow(he_image)
        ground_truth_axis.imshow(ground_truth_overlay)
        ground_truth_axis.set_title(
            f"Ground truth\n{marker_name} | n={ground_truth_count}"
        )
        ground_truth_axis.set_ylabel(marker_name, rotation=0, labelpad=25, va="center")
        ground_truth_axis.axis("off")

        prediction_axis.imshow(he_image)
        prediction_axis.imshow(prediction_overlay)
        prediction_axis.set_title(f"Prediction\n{marker_name} | n={prediction_count}")
        prediction_axis.axis("off")

    plt.show()


__all__ = [
    "BIOLOGICAL_MARKER_NOTES",
    "CHANNEL_NAMES",
    "DEFAULT_VISUALIZATION_MARKERS",
    "EXCLUDED_CHANNELS",
    "NUCLEAR_CHANNELS",
    "describe_markers",
    "get_marker_count_comparison",
    "get_tissue_bbox",
    "get_total_marker_count_comparison",
    "load_indexed_image_files",
    "load_metadata",
    "load_sample_records",
    "load_saved_predictions",
    "plot_overlay_comparison",
    "plot_virtual_mif_for_markers",
    "render_context_theme",
]
