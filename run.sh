train_mode=$1
# wget https://raw.githubusercontent.com/wengong-jin/nips17-rexgen/master/USPTO/data.zip
pip install transformers x-transformers accelerate deepspeed wandb rdkit torchvision absl-py scikit-image
WANDB_API_KEY=af0e67b1480739c41dbc92f73b22db0bde21e7c7 accelerate-launch --config_file config/default_config.yaml train_${train_mode}.py
# python3 predict.py --image_path saved_ckpt/output.png --text "a sentence" --graph_mode --ckpt_path saved_ckpt/2000.pt
# python3 predict.py --image_path saved_ckpt/output.png --text "a sentence" --ckpt_path saved_ckpt/1000.pt