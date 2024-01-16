import os
import json
import nrrd
import hydra
import pickle
import trimesh
import pyrender
import numpy as np
from tqdm import tqdm
from PIL import Image
from functools import partial
from tqdm.contrib.concurrent import process_map

IMAGE_SIZE = (224, 224)

def create_model_id_caption_mapping(caption_file_path, id_word_file_path, output_json_path, ignored_models):
    with open(caption_file_path, "rb") as f:
        embeddings_dict = pickle.load(f)
    caption_tuples = embeddings_dict["caption_tuples"]
    with open(id_word_file_path, "r") as f:
        primitives = json.load(f)
    samples = []
    category_model_id_dict = {}
    for inds, category, nrrd_name in tqdm(caption_tuples):
        text = []
        for ind in inds:
            if ind == 0:  # 0: pad
                break
            text.append(primitives["idx_to_word"][str(ind)])
        model_id = nrrd_name.split(".")[0]
        # TODO
        if f"{category}/{model_id}" in ignored_models:
            continue
        samples.append(
            {"model_id": model_id, "category": category, "caption": " ".join(text).replace("\n", ""), "tokens": inds.tolist()}
        )
        if (category, model_id) not in category_model_id_dict:
            category_model_id_dict[(category, model_id)] = True
    with open(output_json_path, "w") as f:
        json.dump(samples, f, indent=2)
    return tuple(category_model_id_dict.keys())


def render_one_obj(category_model_id, obj_model_root_path, output_root_path, num_views):
    output_dir_path = os.path.join(output_root_path, category_model_id[0], category_model_id[1])
    obj_model_path = os.path.join(
        obj_model_root_path, category_model_id[0], category_model_id[1], "models", "model_normalized.obj"
    )

    os.makedirs(output_dir_path, exist_ok=True)

    renderer = pyrender.OffscreenRenderer(viewport_width=IMAGE_SIZE[0], viewport_height=IMAGE_SIZE[1])

    trimesh_obj = trimesh.load(obj_model_path, force="scene")

    scene = pyrender.Scene.from_trimesh_scene(trimesh_obj)

    scene.ambient_light = np.full(shape=3, fill_value=0.1)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=1.0)
    camera_node = scene.add(camera)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_node = scene.add(light)
    scene.set_pose(light_node, trimesh.transformations.rotation_matrix(np.pi / 2, [-1, 0, 0]))

    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

    for i, angle in enumerate(angles):
        output_img_path = os.path.join(output_dir_path, f"{i}.jpg")

        camera_pose = trimesh.scene.cameras.look_at(
            points=np.expand_dims(trimesh_obj.centroid, axis=0), fov=np.pi / 3, distance=0.85,
            rotation=trimesh.transformations.rotation_matrix(
                angle=angle, direction=[0, 1, 0]
            ) @ trimesh.transformations.rotation_matrix(
                angle=np.pi / 5, direction=[-1, 0, 0]
            )
        )
        scene.set_pose(camera_node, camera_pose)
        color = renderer.render(scene, flags=pyrender.RenderFlags.NONE)[0]
        img = Image.fromarray(color)
        img.save(output_img_path)


def pack_npz(category_model_id, data_root_path, img_root_path, output_root_path, num_views):
    category = category_model_id[0]
    model_id = category_model_id[1]

    os.makedirs(os.path.join(output_root_path, category), exist_ok=True)

    voxel_dict = {}

    for voxel_size in (32, 64, 128):
        voxel_dict[f"voxel{voxel_size}"] = nrrd.read(
            os.path.join(data_root_path, f"nrrd_256_filter_div_{voxel_size}_solid", model_id, f"{model_id}.nrrd")
        )[0]

    multi_imgs = np.empty(shape=(num_views, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.uint8)
    for i in range(num_views):
        img_path = os.path.join(img_root_path, category, model_id, f"{i}.jpg")
        multi_imgs[i] = np.transpose(a=np.asarray(Image.open(img_path)), axes=(2, 0, 1))

    np.savez_compressed(
        os.path.join(output_root_path, category, f"{model_id}.npz"), voxel32=voxel_dict["voxel32"],
        voxel64=voxel_dict["voxel64"], voxel128=voxel_dict["voxel128"], images=multi_imgs
    )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):

    os.environ['PYOPENGL_PLATFORM'] = "egl"

    print(f"Using {cfg.cpu_workers} cpu workers")

    for split in ("train", "val", "test"):

        print(f"==> Processing {split} split ...")

        print("Create object and caption mappings ...")
        os.makedirs(os.path.dirname(getattr(cfg.data, f"{split}_lang_data_path")), exist_ok=True)

        if cfg.data.dataset == "Text2ShapeChairTable":
            category_model_id_list = create_model_id_caption_mapping(
                os.path.join(cfg.data.dataset_path, f"processed_captions_{split}.p"),
                os.path.join(cfg.data.dataset_path, "shapenet.json"),
                getattr(cfg.data, f"{split}_lang_data_path"),
                cfg.data.ignored_models
            )
        else:
            category_model_id_dict = {}
            with open(getattr(cfg.data, f"{split}_lang_data_path"), "r") as f:
                data = json.load(f)
            for item in data:
                if (item["category"], item["model_id"]) not in category_model_id_dict:
                    category_model_id_dict[(item["category"], item["model_id"])] = True
            category_model_id_list = tuple(category_model_id_dict.keys())

        print("Render multi-view images ...")
        output_root_path = os.path.join(cfg.data.dataset_path, "preprocessed", "multiview_imgs")
        process_map(
            partial(
                render_one_obj, obj_model_root_path=os.path.join(os.path.dirname(cfg.data.dataset_path), "ShapeNetCore.v2"),
                output_root_path=output_root_path, num_views=cfg.data.num_views
            ), category_model_id_list, chunksize=1, max_workers=cfg.cpu_workers
        )

        print("Pack data to .npz files ...")
        process_map(
            partial(
                pack_npz, data_root_path=os.path.join(cfg.data.dataset_path),
                img_root_path=output_root_path,
                output_root_path=cfg.data.exp_data_root_path, num_views=cfg.data.num_views
            ), category_model_id_list, chunksize=1, max_workers=cfg.cpu_workers
        )


if __name__ == '__main__':
    main()
