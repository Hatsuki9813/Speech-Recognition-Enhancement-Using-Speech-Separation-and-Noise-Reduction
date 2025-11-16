network=MossFormer2_SS_16K
config=config/inference/${network}.yaml

CUDA_VISIBLE_DEVICES=1 python3 -u inference.py \
  --config $config \
  --checkpoint-dir checkpoints/$network \
  --network ${network} \
  --use-cuda 0
