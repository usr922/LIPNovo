CUDA_VISIBLE_DEVICES=0 \
python tests/imputation_denovo.py \
--mode denovo \
--data_path ./nine_species \
--ckpt_path ./saved_models/nine_species.ckpt \
--denovo_output_path results/nine_species.csv \
--config_path configs/config.yaml
