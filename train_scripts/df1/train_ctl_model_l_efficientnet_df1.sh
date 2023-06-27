python train_ctl_model.py \
--config_file="configs/320_efficientnet.yml" \
MODEL.NAME 'efficientnet-b4' \
GPU_IDS [0] \
DATASETS.NAMES 'df1' \
DATASETS.JSON_TRAIN_PATH 'dataset/subsets_train_reid_cropped_320_320.json' \
DATASETS.ROOT_DIR 'dataset/subsets_320_320_images' \
SOLVER.IMS_PER_BATCH 24 \
TEST.IMS_PER_BATCH 24 \
SOLVER.BASE_LR 1e-4 \
SOLVER.MAX_EPOCHS 50 \
OUTPUT_DIR './logs2/df1/320_efficientnet_b4_LGPR_GGPR' \
DATALOADER.USE_RESAMPLING False \
MODEL.KEEP_CAMID_CENTROIDS False \
INPUT.USE_LGT False \
INPUT.USE_LGPR True \
INPUT.USE_GGPR True \
INPUT.USE_FUSE_RGB False \
MODEL.USE_CENTROIDS True \
EXPERIMENT_NAME 'efficientnet'
