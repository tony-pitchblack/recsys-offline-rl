# RecSys Offline RL for e-commerce

This repo demonstrates Offline RL usage for Recommender System in e-commerce domain to optimize ranking metrics:
- Seq2Seq backbone SASRec based on [RecTools](https://github.com/MobileTeleSystems/RecTools);
- Actor-Critic RL architecture & training pipeline based on [**"Supervised Advantage Actor-Critic for Recommender Systems"**](https://arxiv.org/abs/2111.03474) paper.

The model is evaluated on following e-commerce datasets (click & purchase interactions):

- [**YooChoose**](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015);
- [**RetailRocket**](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Additional ablations:
- Train & evaluate the model on purchase-only data;
- Train & evaluate the model with *ranking NDCG-based reward function*

*Scalar reward:*
```text
reward = 1, if action is a click
reward = 5, if action is a purchase
```

*NDCG-based reward:*
```text
reward_ndcg = 1 / log2(rank + 1)
rank = 1 + number of candidate items with logit > target_item_logit
```
where `target_item_logit` is the model logit for the observed item.

## Offline RL pipeline overview
First, a replay buffer is constructed from user's logged data.

Then the model is trained in two stages:

- (a) **Warmup stage** (SNQN algorithm): pretrain critic jointly with actor via a reward function;

- (b) **Finetuning stage** (SA2C algorithm): regularize actor's policy with critic's Q-value estimates.

*The data is split by users into train/val/test in 80:10:10 ratio; metrics are reported for test split*

![SQN and SA2C](results/misc/snqn_sa2c.png)

## Results

Main observations:
1. Switching the backbone to RecTools yields larger improvements than RL finetuning in the evaluated setups.
2. The effect of RL finetuning is not consistent across backbones and datasets.
3. RL finetuning provides smaller gains in the purchase-only setting than in the clicks-and-purchases setting.
4. NDCG-based reward function in purchse-only setting does not significantly improve over regular scalar reward function

*We report `NDCG@10` metric separately for clicks and purchases. Note that in `clicks & puchases` setting the model is still trained on both clicks and purchases data despite the metrics being reported separately.*

### Clicks & purchases

#### RetailRocket

##### Clicks NDCG@10 ([experiments barplot](#retailrocket-clicks-purchases-barplot))

| Implementation | SASRec | SASRec + SA2C |
| --- | --- | --- |
| Authors (tensorflow) | `0.211` | `0.242` (`+0.031`) |
| Ours (torch) | `0.220` | `0.238` (`+0.018`) |
| Ours (torch + rectools) | **0.322** | **0.326 (+0.004)** |

##### Purchases NDCG@10 ([experiments barplot](#retailrocket-clicks-purchases-barplot))

| Implementation | SASRec | SASRec + SA2C |
| --- | --- | --- |
| Authors (tensorflow) | `0.451` | `0.525` (`+0.074`) |
| Ours (torch) | `0.436` | `0.518` (`+0.082`) |
| Ours (torch + rectools) | **0.635** | **0.703 (+0.068)** |

#### YooChoose

##### Clicks NDCG@10 ([experiments barplot](#yoochoose-clicks-purchases-barplot))

| Implementation | SASRec | SASRec + SA2C |
| --- | --- | --- |
| Authors (tensorflow) | `0.252` | `0.272` (`+0.020`) |
| Ours (torch) | `0.266` | `0.236` (`-0.030`) |
| Ours (torch + rectools) | **0.278** | **0.279 (+0.001)** |

##### Purchases NDCG@10 ([experiments barplot](#yoochoose-clicks-purchases-barplot))

| Implementation | SASRec | SASRec + SA2C |
| --- | --- | --- |
| Authors (tensorflow) | `0.333` | **0.373 (+0.040)** |
| Ours (torch) | `0.325` | `0.334` (`+0.009`) |
| Ours (torch + rectools) | **0.355** | `0.371` (`+0.016`) |

#### Experiments barplots

Implementations:
- `torch` = our torch reimplementation with author's SASRec architecture
- `rectools` = our torch reimplementation with the `rectools` SASRec architecture

Experiment (config) types:
- `default` = default SASRec-SA2C setup
- `baseline` = baseline SASRec setup
- `*auto_warmup` = warmup phase with early stopping on `val/ndcg@10`
- no `*auto_warmup` = hardcoded warmup epochs
- `sampled_loss` = sampled softmax for actor and sampled next-state Q-values for critic

<a id="retailrocket-clicks-purchases-barplot"></a>
##### RetailRocket (clicks & purchases)

![RetailRocket](results/plots/retailrocket/test_results.png)

<a id="yoochoose-clicks-purchases-barplot"></a>
##### YooChoose (clicks & purchases)

![YooChoose](results/plots/yoochoose/test_results.png)

### Purchase-only

In the purchase-only setting, we report `NDCG@10` only for purchases.

*For author's implementation we reuse results from clicks & purchases setting for reference*

#### RetailRocket

##### Purchases NDCG@10 ([experiments barplot](#retailrocket-purchase-only-barplot))

| Implementation | SASRec | SASRec + SA2C |
| --- | --- | --- |
| Authors (tensorflow) | `0.451` | `0.525` (`+0.074`) |
| Ours (torch + rectools) | **0.680** | **0.694 (+0.014)** |

#### YooChoose

##### Purchases NDCG@10 ([experiments barplot](#yoochoose-purchase-only-barplot))

| Implementation | SASRec | SASRec + SA2C |
| --- | --- | --- |
| Authors (tensorflow) | `0.333` | `0.373` (`+0.040`) |
| Ours (torch + rectools) | **0.416** | **0.429 (+0.013)** |

#### Experiments barplots

<a id="retailrocket-purchase-only-barplot"></a>
##### RetailRocket (purchase-only)

![RetailRocket (purchase-only)](results/plots/retailrocket/test_results_purchase_only.png)

<a id="yoochoose-purchase-only-barplot"></a>
##### YooChoose (purchase-only)

![YooChoose (purchase-only)](results/plots/yoochoose/test_results_purchase_only.png)


## Notes

For full list of commands to reproduce experiments refer to `COMMANDS.md`

Plots were generated by
```bash
python scripts/plot_test_results.py --max-metric-value 1.0 0.6 0.3
```
