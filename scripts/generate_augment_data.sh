# generate sub-instructions
python dataset_process/make_subinstr.py --source data/R2R_train.json --target data/R2R_train_subinstr.json
python dataset_process/make_subinstr.py --source data/prevalent/prevalent_aug.json --target data/prevalent/prevalent_aug_subinstr.json

# generate positive instructions
python dataset_process/gen_aug_sub_instr.py --source data/R2R_train_subinstr.json --target data/R2R_train_subinstr_aug_pos.json --dest_dir data
python dataset_process/gen_aug_sub_instr.py --source data/prevalent/prevalent_aug_subinstr.json --target data/prevalent/prevalent_aug_subinstr_aug_pos.json --dest_dir data

# generate sub-optimal trajectories
python dataset_process/gen_subop_traj_search.py --source data/R2R_train.json --target data/R2R_train_bfs_paths.json
python dataset_process/gen_subop_traj_search_err.py --source data/R2R_train.json --target data/R2R_train_bfs_paths_err.json
python dataset_process/gen_subop_traj_search.py --source data/prevalent/prevalent_aug.json --target data/prevalent/prevalent_aug_bfs_paths.json
python dataset_process/gen_subop_traj_search_err.py --source data/prevalent/prevalent_aug.json --target data/prevalent/prevalent_aug_bfs_paths_err.json
