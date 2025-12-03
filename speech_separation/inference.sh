network=MossFormer2_SS_16K
config=config/inference/${network}.yaml

python3 -u inference.py \
  --config $config \
  --checkpoint-dir checkpoints/$network \
  --input-path "data/60594d0b-5eb0-4a17-8b75-bb683fd98043.wav" \
  --output-dir "data" \
  --network ${network} \
  --use-cuda 0
