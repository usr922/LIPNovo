CUDA_VISIBLE_DEVICES=0 \
python tests/imputation_denovo.py \
--mode train \
--data_path ./nine_species \
--config_path configs/config.yaml
