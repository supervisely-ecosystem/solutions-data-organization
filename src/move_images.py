import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import supervisely as sly
from fastapi.requests import Request
from fastapi.routing import APIRouter

import src.sly_functions as f
import src.sly_globals as g

move_images_router = APIRouter()


@move_images_router.post("/move_labeled_data")
async def move_images(request: Request):
    try:
        state = request.state.state
        src_project_id = state["src_project_id"]
        dst_project_id = state["dst_project_id"]
        image_ids = state["image_ids"]
        src_collection_id = state.get("src_collection_id")
        split_settings = state.get("split_settings")
        sly.logger.info(
            f"Move images: {src_project_id} (collection ID:{src_collection_id} -> {dst_project_id}, split_settings: {split_settings}, image_ids: {image_ids}"
        )
        return {
            "data": run_move_images(
                src_project_id,
                dst_project_id,
                image_ids,
                src_collection_id,
                split_settings,
            )
        }
    except Exception as e:
        sly.logger.error(f"Error during random sample: {e}")
        return {"error": str(e)}


def run_move_images(
    src_project_id: int,
    dst_project_id: int,
    image_ids: List[int],
    src_collection_id: Optional[int] = None,
    split_settings: Optional[Dict[str, int]] = None,
):
    """
    Function to move images from the source project to the destination project.
    After moving, the images are deleted from the source collection and project.

    If split_settings is provided, the images are split into train and val sets.
    For each set, a new collection will be created in the destination project.
    The split_settings dictionary should contain the following keys:
        - "mode": "random" or "datasets'
        - "train":
            - if mode is "random": percentage of images to be used for training
            - if mode is "datasets": list of dataset IDs or names to be used for training
        - "val":
            - if mode is "random": percentage of images to be used for validation
            - if mode is "datasets": list of dataset IDs or names to be used for validation
    Args:
        src_project_id (int): ID of the source project.
        dst_project_id (int): ID of the destination project.
        image_ids (list): List of image IDs to be moved.
        src_collection_id (int, optional): ID of the source collection. Defaults to None.
        split_settings (dict, optional): Dictionary with split settings. Defaults to None.
    Returns:
        Tuple of number of added images, preview URLs, and counts.
        - added (int): Number of images added to the destination project.
        - preview_urls (list): List of preview URLs for the added images.
        - counts (list): List of counts for the added images.
        Tuple[int, List[str], List[int]]
    """
    if len(image_ids) == 0:
        sly.logger.warning("No images to move")
        return {
            "num_added": 0,
            "preview_urls": None,
            "counts": None,
            "train_collection": None,
            "val_collection": None,
        }

    num_added = 0
    preview_urls, counts = None, None
    train_collection, val_collection = None, None

    # Get source and destination projects from the API
    src_datasets = g.api.dataset.get_list(src_project_id, recursive=True)
    src_ds_tree = g.api.dataset.get_tree(src_project_id)
    dst_ds_tree = g.api.dataset.get_tree(dst_project_id)

    # Create a mapping with different between source and destination datasets
    src_to_dst_map, ds_to_create = f.create_dataset_mapping(src_ds_tree, dst_ds_tree)

    f.merge_update_metas(g.api, src_project_id, dst_project_id)

    img_infos = g.api.image.get_info_by_id_batch(image_ids, force_metadata_for_links=False)
    items = defaultdict(list)
    for img_info in img_infos:
        items[img_info.dataset_id].append(img_info)

    # Move the images to the destination project
    _, added = f.copy_or_move_images(
        g.api,
        dst_project_id,
        src_to_dst_map,
        items,
        ds_to_create,
        src_datasets,
        move=True,
    )
    num_added = sum([len(i) for i in added.values()])
    sly.logger.info(f"Copied {num_added} images to the destination project")

    # Remove the images from the source collection
    if src_collection_id is not None:
        g.api.entities_collection.remove_items(src_collection_id, image_ids)

    # add the images to the destination collections (train/val)
    if split_settings is not None:
        mode = split_settings.get("mode")
        sly.logger.info(f"Splitting images into train and val sets: {mode}")
        if mode == "random":
            train_percent = split_settings.get("train", 0)
            val_percent = split_settings.get("val", 0)
            train_count = int(num_added / 100 * train_percent)
            val_count = int(num_added / 100 * val_percent)
            new_img_ids = []
            for img_ids in added.values():
                new_img_ids.extend(img_ids)
            random.shuffle(new_img_ids)
            train_ids = new_img_ids[:train_count]
            val_ids = new_img_ids[train_count : train_count + val_count]
            sly.logger.info(
                f"Splitting {num_added} images into train ({len(train_ids)}) and val ({len(val_ids)})"
            )

            if len(train_ids) > 0 and len(val_ids) > 0:
                existing_train, existing_val, split_idx = f.get_splits_details(
                    g.api, dst_project_id
                )
                train_collection = g.api.entities_collection.create(
                    dst_project_id, f"train_{split_idx + 1}"
                )
                val_collection = g.api.entities_collection.create(
                    dst_project_id, f"val_{split_idx + 1}"
                )
                sly.logger.info(
                    f"Created collections '{train_collection.name}' and '{val_collection.name}' for new train and val sets"
                )
                g.api.entities_collection.add_items(train_collection.id, train_ids)
                g.api.entities_collection.add_items(val_collection.id, val_ids)

                sly.logger.info(
                    f"Added {len(train_ids)} images to train collection '{train_collection.name}' "
                    f"and {len(val_ids)} images to val collection '{val_collection.name}'"
                )
                random_train = random.choice(train_ids)
                random_val = random.choice(val_ids)

                counts = [existing_train + len(train_ids), existing_val + len(val_ids)]
                preview_urls = [
                    g.api.image.get_info_by_id(random_train).preview_url,
                    g.api.image.get_info_by_id(random_val).preview_url,
                ]

    return {
        "num_added": num_added,
        "preview_urls": preview_urls,
        "counts": counts,
        "train_collection": train_collection,
        "val_collection": val_collection,
    }
