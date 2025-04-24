import random
from collections import defaultdict
from typing import Dict, List, Tuple

import supervisely as sly


def create_dataset_mapping(
    src_ds_tree: Dict[sly.DatasetInfo, Dict],
    dst_ds_tree: Dict[sly.DatasetInfo, Dict],
) -> Tuple[Dict[int, int], List[Tuple[int, sly.DatasetInfo]]]:
    """
    Create a mapping between source and destination datasets to handle nested datasets.

    Args:
        src_ds_tree: The source project dataset tree
        dst_ds_tree: The destination project dataset tree
        src_datasets: Flat list of source datasets
        dst_datasets: Flat list of destination datasets

    Returns:
        A tuple containing:
        - src_to_dst_map: Dict mapping source dataset IDs to destination dataset IDs (or None if not exists)
        - ds_to_create: List of tuples (parent_dst_id, src_ds_info) for datasets that need to be created
    """
    # Maps source dataset IDs to destination dataset IDs (or None if it doesn't exist)
    src_to_dst_map = {}

    # List of (parent_dst_id, src_ds_info) pairs for datasets that need to be created
    ds_to_create = []

    # Helper function to build mapping recursively
    def process_datasets(src_tree, dst_tree, parent_dst_id=None):
        for src_ds_info, src_children in src_tree.items():
            # Try to find matching dataset in destination by name
            dst_ds_info = None
            dst_children = {}

            for dst_info, dst_child in dst_tree.items():
                if dst_info.name == src_ds_info.name:
                    dst_ds_info = dst_info
                    dst_children = dst_child
                    break

            if dst_ds_info:
                # Dataset exists in destination
                src_to_dst_map[src_ds_info.id] = dst_ds_info.id
                # Process children recursively
                process_datasets(src_children, dst_children, dst_ds_info.id)
            else:
                # Dataset doesn't exist in destination, needs to be created
                src_to_dst_map[src_ds_info.id] = None
                ds_to_create.append(src_ds_info.id)
                # Process children recursively with None as the parent ID
                # (they'll be created after their parent)
                process_datasets(src_children, {}, None)

    # Start the recursive mapping
    process_datasets(src_ds_tree, dst_ds_tree)

    return src_to_dst_map, ds_to_create


def get_diffs(
    api: sly.Api,
    src_datasets: List[sly.DatasetInfo],
    src_tree: Dict[sly.DatasetInfo, Dict],
    dst_tree: Dict[sly.DatasetInfo, Dict],
) -> Dict[int, List[sly.ImageInfo]]:
    """
    Get the images that are different between source and destination datasets.

    Args:
        api: sly.API instance
        src_datasets: List of source datasets (flat list)
        dst_datasets: List of destination datasets (flat list)
        src_tree: Source dataset tree
        dst_tree: Destination dataset tree

    Returns:
        A dictionary mapping source dataset IDs to lists of different images.
    """
    src_to_dst_map, ds_to_create = create_dataset_mapping(src_tree, dst_tree)

    diff_images = defaultdict(list)
    for src_ds in src_datasets:
        dst_ds = src_to_dst_map.get(src_ds.id)
        src_imgs = api.image.get_list(src_ds.id, force_metadata_for_links=False)
        if dst_ds is None:
            diff_images[src_ds.id].extend(src_imgs)
        else:
            dst_imgs = api.image.get_list(dst_ds, force_metadata_for_links=False)
            src_imgs_dict = {img.name: img for img in src_imgs}
            dst_imgs_dict = {img.name: img for img in dst_imgs}
            for img_name, img in src_imgs_dict.items():
                if img_name not in dst_imgs_dict:
                    diff_images[src_ds.id].append(img)

    return diff_images, src_to_dst_map, ds_to_create


def prepare_sample(
    diffs: Dict[int, List[sly.ImageInfo]], sample_size: int
) -> Dict[int, List[sly.ImageInfo]]:
    """
    Prepare a sample of images from the differences and sample size.
    Args:
        diffs (dict): Dictionary of differences between source and destination datasets.
        sample_size (int): Number of images to sample.
    Returns:
        dict: Dictionary of sampled images.
    """
    # Calculate the total number of differences
    total_diffs = sum(len(imgs) for imgs in diffs.values())

    # If the sample size is greater than the total differences, return all images
    if sample_size >= total_diffs:
        return diffs

    # Calculate the sample size for each dataset
    samples_per_dataset = {}
    remaining = sample_size
    for ds_id, imgs in diffs.items():
        # Calculate proportional size and round down
        ds_sample = int((len(imgs) / total_diffs) * sample_size)
        samples_per_dataset[ds_id] = ds_sample
        remaining -= ds_sample

    # Distribute any remaining samples randomly
    if remaining > 0:
        datasets_with_space = [
            ds_id for ds_id, imgs in diffs.items() if len(imgs) > samples_per_dataset[ds_id]
        ]
        while remaining > 0 and datasets_with_space:
            ds_id = random.choice(datasets_with_space)
            if len(diffs[ds_id]) > samples_per_dataset[ds_id]:
                samples_per_dataset[ds_id] += 1
                remaining -= 1
            else:
                datasets_with_space.remove(ds_id)

    # Prepare the sampled images
    sampled_images = {}
    for ds_id, sample_count in samples_per_dataset.items():
        if sample_count > 0:
            sampled_images[ds_id] = random.sample(diffs[ds_id], sample_count)
    return sampled_images


def copy_or_move_images(
    api: sly.Api,
    dst_project_id: int,
    src_to_dst_map: Dict[int, int],
    sampled_images: Dict[int, List[sly.ImageInfo]],
    ds_to_create: List[int],
    src_datasets: List[sly.DatasetInfo],
    move: bool = False,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Copy or move images from the source project to the destination project.

    Args:
        api (supervisely.api.Api): API instance.
        dst_project_id (int): ID of the destination project.
        src_to_dst_map (dict): Mapping of source dataset IDs to destination dataset IDs.
        sampled_images (dict): Dictionary of sampled images.
        ds_to_create (list): List of datasets to create in the destination project.
        src_datasets (list): List of source datasets.
        move (bool): Whether to move images instead of copying.

    Returns:
        dict: Dictionary with source and destination dataset IDs.
    """
    # Prepare children-parent relationships for source and destination datasets
    src_id_to_info = {ds.id: ds for ds in src_datasets}
    src_child_to_parents = {ds.id: [] for ds in src_datasets}
    for ds in src_datasets:
        current = ds
        while parent_id := current.parent_id:
            src_child_to_parents[ds.id].append(parent_id)
            current = src_id_to_info[parent_id]

    added = {}
    src = {}
    for src_ds_id, src_imgs in sampled_images.items():
        if len(src_imgs) > 0:
            dst_ds_id = src_to_dst_map.get(src_ds_id)
            if dst_ds_id is None and src_ds_id in ds_to_create:
                # Create new dataset in destination project
                src_parent_ids = src_child_to_parents[src_ds_id]
                dst_parent_id = None
                for parent_id in src_parent_ids:
                    src_ds = api.dataset.get_info_by_id(parent_id)
                    dst_ds = api.dataset.create(
                        dst_project_id, src_ds.name, parent_id=dst_parent_id
                    )
                    dst_parent_id = dst_ds.id
                    src_to_dst_map[parent_id] = dst_parent_id

                # Create new dataset in destination project
                src_ds = api.dataset.get_info_by_id(src_ds_id)
                dst_ds = api.dataset.create(dst_project_id, src_ds.name, parent_id=dst_parent_id)
                dst_ds_id = dst_ds.id
                src_to_dst_map[src_ds_id] = dst_ds_id

            new_imgs = api.image.copy_batch_optimized(
                src_dataset_id=src_ds_id,
                src_image_infos=src_imgs,
                dst_dataset_id=dst_ds_id,
                with_annotations=True,
                save_source_date=False,
            )

            if move:
                api.image.remove_batch([i.id for i in src_imgs], batch_size=200)

            src[src_ds_id] = [i.id for i in src_imgs]
            added[dst_ds_id] = [i.id for i in new_imgs]
            sly.logger.info(f"Copied {len(new_imgs)} images to dataset {dst_ds_id}")
    return src, added
