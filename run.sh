train_mode=$1
export OMP_NUM_THREADS=16
# wget https://raw.githubusercontent.com/wengong-jin/nips17-rexgen/master/USPTO/data.zip
# 不知道为啥google cloud上用到了torch_xla
# pip uninstall torch_xla
pip install transformers x-transformers accelerate deepspeed wandb rdkit torchvision absl-py scikit-image
WANDB_API_KEY=af0e67b1480739c41dbc92f73b22db0bde21e7c7 accelerate launch --config_file config/default_config.yaml train_${train_mode}.py
# WANDB_API_KEY=af0e67b1480739c41dbc92f73b22db0bde21e7c7 python3 train_${train_mode}.py
# python3 predict.py --image_path saved_ckpt/output.png --text "a sentence" --graph_mode --ckpt_path saved_ckpt/2000.pt
# python3 predict.py --image_path saved_ckpt/output.png --text "a sentence" --ckpt_path saved_ckpt/1000.pt