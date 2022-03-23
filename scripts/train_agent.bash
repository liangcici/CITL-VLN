name=VLNBERT-train-Prevalent

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug_diss_subinstr.json
      --aug_path data/prevalent/prevalent_aug_bfs_paths.json
      --aug_path2 data/prevalent/prevalent_aug_bfs_paths_err.json
      --aug_train_path data/R2R_train_bfs_paths.json
      --aug_train_path2 data/R2R_train_bfs_paths_err.json
      --aug_path_type len
      --pos_thr 1.2
      --neg_thr 1.4
      --aug_path_num 16
      --con_path_loss_weight 0.1
      --con_path_loss_type circle
      --circle_queue
      --nce_k 240
      --circle_mining
      --aug_lang data/prevalent/prevalent_aug_sub_instrs_pos.json
      --aug_train_lang data/R2R_train_subinstr_aug_pos.json
      --aug_lang_num 13
      --lang_loss_weight 0.01
      --lang_local_loss_weight 0.01

      --test_only 0

      --train auglistener

      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python r2r_src/train.py $flag --name $name
