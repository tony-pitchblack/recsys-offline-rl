## Download YooChoose/Retailrocket datasets (`SA2C_code.zip`)

```bash
source .venv/bin/activate
uv pip install -r requirements.txt

mkdir -p data
gdown "https://drive.google.com/uc?id=185KB520pBLgwmiuEe7JO78kUwUL_F45t" -O "data/SA2C_code.zip"
unzip -o "data/SA2C_code.zip" -d "data/"
rm -f "data/SA2C_code.zip"

# normalize nested zip layout if needed
if [ -d "data/SA2C_code/SA2C_code/Kaggle" ]; then mv "data/SA2C_code/SA2C_code/"* "data/SA2C_code/"; rmdir "data/SA2C_code/SA2C_code"; fi
```

## Run Torch (local .venv + uv) — SASRec only

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Start MLflow server (tmux)

- Requires `.env` with `MLFLOW_HOST` + `MLFLOW_PORT` (example: `MLFLOW_HOST=0.0.0.0`, `MLFLOW_PORT=5000`).

```bash
source .venv/bin/activate
uv pip install -r requirements.txt

bash scripts/start_mlflow_tmux.sh
```

## Sampled softmax (CE) with 1/4 vocab

- Add `ce_n_negatives: 0.25` to your rectools YAML config (top-level). This sets the number of sampled negatives to \(\lfloor 0.25 * item\_num \rfloor\).
- Implementation detail: for fractional `ce_n_negatives`, training uses a matmul-based “shared negatives” construction (no per-row `[N,C,H]` embedding gather).
- `ce_n_negatives: null` means full softmax (scores full vocab).
- `num_val_negative_samples` (optional) controls eval candidate pool size for `bert4rec_loo` only.

## Plot test results (clicks + purchase ndcg@10)

- Writes plots under `results/plots/{dataset_name}/test_results.png`.

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python scripts/plot_test_results.py  --max-metric-value 0.5

# examples
# (omit --dataset to plot all datasets found under logs/)
CUDA_VISIBLE_DEVICES=0 python scripts/plot_test_results.py --dataset retailrocket --max-metric-value 0.5
CUDA_VISIBLE_DEVICES=0 python scripts/plot_test_results.py --max-metric-value 1.0 0.6
```

## retailrocket
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/default.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/default_optimal_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/baseline.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/sampled_loss.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/sampled_loss_pointwise_critic.yml
CUDA_VISIBLE_DEVICES=4 python train.py --config conf/retailrocket/crr.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/default_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/default_from_pretrained.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/default_from_pretrained_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/sampled_loss_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/sampled_loss_from_pretrained.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/sampled_loss_from_pretrained_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/sampled_loss_pointwise_critic_auto_warmup.yml

# purchase-only
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/default_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/ndcg_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/default_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/baseline_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/default_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/ndcg_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/baseline.yml --smoke-cpu --max_steps 64
```

## yoochoose
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_optimal_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/baseline.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/sampled_loss.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/sampled_loss_pointwise_critic.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/sampled_loss_pointwise_critic_mlp.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/crr.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_from_pretrained.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_from_pretrained_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/sampled_loss_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/sampled_loss_from_pretrained.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/sampled_loss_from_pretrained_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/sampled_loss_pointwise_critic_auto_warmup.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/sampled_loss_pointwise_critic_mlp_auto_warmup.yml

# purchase-only
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/ndcg_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_optimal_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/baseline_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/default_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/ndcg_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/baseline.yml --smoke-cpu --max_steps 64
```

## ml_1m
```bash
source .venv/bin/activate
uv pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=0 python train.py --config conf/ml_1m/baseline_multi.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/ml_1m/default_multi.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/ml_1m/baseline_multi.yml --smoke-cpu --max_steps 64
```

## Optuna gridsearch (rectools)

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/baseline_gridsearch.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/yoochoose/baseline_gridsearch_purchase_only.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/baseline_gridsearch.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/retailrocket/baseline_gridsearch_purchase_only.yml
```

## Run SA2C scripts using installed envs
```bash
conda activate sa2c_code_tf
cd data/SA2C_code/Kaggle && CUDA_VISIBLE_DEVICES=0 python SA2C.py --data data
cd ../RC15 && CUDA_VISIBLE_DEVICES=0 python SA2C.py --data data
```