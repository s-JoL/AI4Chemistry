# rm -rf wandb launch_logs
# tar -cvf tmp.tar.gz config/ data/ model/ utils/ .gitignore AGI.ipynb run.sh train_graph.py train_smiles.py
# ARNOLD_TRIAL_ID
train_mode=$1
total_gpu=`expr $ARNOLD_WORKER_GPU \* $ARNOLD_NUM`
pip install transformers x-transformers accelerate deepspeed wandb rdkit torchvision absl-py scikit-image
WANDB_API_KEY=af0e67b1480739c41dbc92f73b22db0bde21e7c7 accelerate-launch --config_file config/default_config.yaml train_${train_mode}.py
# WANDB_API_KEY=af0e67b1480739c41dbc92f73b22db0bde21e7c7 accelerate-launch --config_file config/default_config.yaml --main_process_ip $ARNOLD_WORKER_0_HOST --main_process_port $ARNOLD_WORKER_0_PORT --num_processes $total_gpu --num_machines $ARNOLD_NUM --machine_rank $ARNOLD_ID train_${train_mode}.py
# hadoop fs -put saved_ckpt hdfs://haruna/byte_search/aweme_relevance/roformer/chem/$ARNOLD_TRIAL_ID
# python3 predict.py --image_path saved_ckpt/output.png --text "a sentence" --graph_mode --ckpt_path saved_ckpt/2000.pt
# python3 predict.py --image_path saved_ckpt/output.png --text "a sentence" --ckpt_path saved_ckpt/1000.pt