import glob
import os
from pathlib import Path

# Resolve paths from this file location instead of current working directory.
# This makes imports robust when launching from JiT-opt or other folders.
_PKG_ROOT = Path(__file__).resolve().parent.parent  # .../unified_video_action/unified_video_action
_BDDL_ROOT = _PKG_ROOT / "env" / "libero" / "bddl_files"


def _bddl_path(task_name: str, filename: str) -> str:
    return str(_BDDL_ROOT / task_name / filename)

bddl_file_name_dict = {}

task_names = [
    "libero_goal",
    "libero_10",
    "libero_90",
    "libero_object",
    "libero_spatial",
]
for task_name in task_names:
    for file in glob.glob(
        str(_BDDL_ROOT / task_name / "*.bddl")
    ):
        if task_name == "libero_10" or task_name == "libero_90":
            bddl_file_name_dict[
                "chiliocosm/bddl_files/%s/%s" % ("libero_100", file.split("/")[-1])
            ] = file
        else:
            bddl_file_name_dict[
                "chiliocosm/bddl_files/%s/%s" % (task_name, file.split("/")[-1])
            ] = file

bddl_file_name_dict_correct = {
    "chiliocosm/bddl_files/libero_goal/open_the_middle_layer_of_the_drawer.bddl": _bddl_path("libero_goal", "open_the_middle_drawer_of_the_cabinet.bddl"),
    "chiliocosm/bddl_files/libero_goal/open_the_top_layer_of_the_drawer_and_put_the_bowl_inside.bddl": _bddl_path("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside.bddl"),
    "chiliocosm/bddl_files/libero_goal/put_the_cream_cheese_on_the_bowl.bddl": _bddl_path("libero_goal", "put_the_cream_cheese_in_the_bowl.bddl"),
    "chiliocosm/bddl_files/libero_goal/put_the_bowl_on_the_top_of_the_drawer.bddl": _bddl_path("libero_goal", "put_the_bowl_on_top_of_the_cabinet.bddl"),
    "chiliocosm/bddl_files/libero_goal/put_the_wine_bottle_on_the_top_of_the_drawer.bddl": _bddl_path("libero_goal", "put_the_wine_bottle_on_top_of_the_cabinet.bddl"),
    "chiliocosm/bddl_files/libero_100_debug/STUDY_TABLETOP_SCENE1_pick_up_the_book_and_place_it_in_the_back_of_the_caddy.bddl": _bddl_path("libero_10", "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl"),
    "chiliocosm/bddl_files/libero_100/KITCHEN_SCENE2_put_the_black_bowl_in_the_middle_on_the_plate.bddl": _bddl_path("libero_90", "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_100/KITCHEN_SCENE2_stack_the_black_bowl_in_the_middle_on_the_black_bowl_at_the_front.bddl": _bddl_path("libero_90", "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.bddl"),
    "chiliocosm/bddl_files/libero_100_debug/KITCHEN_TABLETOP_SCENE9_put_the_frypan_into_the_bottom_layer_of_the_cabinet.bddl": _bddl_path("libero_90", "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf.bddl"),
    "chiliocosm/bddl_files/libero_100/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_of_the_caddy.bddl": _bddl_path("libero_90", "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl"),
    "chiliocosm/bddl_files/libero_100/STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_compartment_of_the_caddy.bddl": _bddl_path("libero_90", "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy.bddl"),
    "chiliocosm/bddl_files/libero_100/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_compartment_of_the_caddy.bddl": _bddl_path("libero_90", "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_alphabet_soup_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_bbq_sauce_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_butter_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_butter_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_chocolate_pudding_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_cream_cheese_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_ketchup_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_ketchup_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_milk_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_milk_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_orange_juice_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_orange_juice_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_salad_dressing_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_salad_dressing_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_object/pick_the_tomato_sauce_and_place_it_in_the_basket.bddl": _bddl_path("libero_object", "pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_from_table_center_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_in_the_top_layer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_next_to_the_cookies_box_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_next_to_the_plate_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_on_the_cookies_box_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl"),
    "chiliocosm/bddl_files/libero_spatial/pick_the_akita_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl": _bddl_path("libero_spatial", "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl"),
}

for k, v in bddl_file_name_dict_correct.items():
    if v in bddl_file_name_dict.values():
        bddl_file_name_dict[k] = v
    else:
        # Keep runtime robust across slightly different LIBERO asset snapshots.
        # Missing correction entries should not crash the whole evaluation pipeline.
        pass
