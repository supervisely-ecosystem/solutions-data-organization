import supervisely as sly
from fastapi.requests import Request
from fastapi.routing import APIRouter

import src.sly_functions as f
import src.sly_globals as g

sampling_router = APIRouter()


@sampling_router.post("/random_sample")
async def random_sample(request: Request):
    req = await request.json()
    try:
        state = req["state"]
        src_project_id = state["src_project_id"]
        dst_project_id = state["dst_project_id"]
        sample_size = state["sample_size"]
        return {"data": run_random_sample(src_project_id, dst_project_id, sample_size)}
    except Exception as e:
        sly.logger.error(f"Error during random sample: {e}")
        return {"error": str(e)}


def run_random_sample(src_project_id, dst_project_id, sample_size: int):
    """
    Function to create a random sample of images
    from the source project and copy them to the destination project.

    Args:
        src_project_id (int): ID of the source project.
        dst_project_id (int): ID of the destination project.
        sample_size (int): Number of images to sample.
    """
    if sample_size == 0:
        sly.logger.warning("Sample size is 0")
        return {"src": None, "dst": None}

    # Get source and destination projects from the API
    src_datasets = g.api.dataset.get_list(src_project_id, recursive=True)
    src_ds_tree = g.api.dataset.get_tree(src_project_id)
    dst_ds_tree = g.api.dataset.get_tree(dst_project_id)

    # Create a mapping with different between source and destination datasets
    diffs = f.get_diffs(g.api, src_datasets, src_ds_tree, dst_ds_tree)
    diff_images, src_to_dst_map, ds_to_create = diffs

    # If there is no difference between the datasets, return None
    if not diff_images:
        sly.logger.warning("No new items to copy to the labeling project")
        return {"src": None, "dst": None}

    # Prepare the sample
    sampled_images = f.prepare_sample(diff_images, sample_size)

    # # Copy the sampled images to the destination project
    src, added = f.copy_or_move_images(
        g.api, dst_project_id, src_to_dst_map, sampled_images, ds_to_create, src_datasets
    )

    return {"src": src, "dst": added}


if __name__ == "__main__":
    # Example usage
    src_project_id = 1312
    dst_project_id = 1764
    sample_size = 4

    result = run_random_sample(src_project_id, dst_project_id, sample_size)
    print(result)
