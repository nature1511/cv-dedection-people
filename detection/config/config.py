class Configs:
    model_number_classes = 81
    random_seed = 0
    path_data = "data"
    path_new_data = "new_data"
    path_weight_model = "weight"
    path_new_data = "new_data"
    path_to_save_state_model = "weight"
    device = "cpu"
    batch_size = 64
    use_padding_in_image_transform = False
    use_head = False
    SSDTransformer_max_num = 1000
    decode_single_max_num = 200
    decode_result = {"criteria": 0.5, "max_output": 200, "pic_threshold": 0.25}
    use_pick_best_in_eval = False
    show_image = False
    save_image = True
