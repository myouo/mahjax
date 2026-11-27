# 性能优化与 Kaggle T4×2 训练指南

本文记录 Sovereign-Mahjong V6 在 `mahjong_rl` 中的性能体检、已落地的优化，以及
在 Kaggle（2× Tesla T4）上训练/评测时需要注意的事项。

> 结论先行：MahJax 论文里的 **1M~2M steps/s** 是 8×A100 上**纯 mahjax 环境**的数字。
> 本项目训练/评测跑的是完整的 V6 栈（BeliefNet + Analytical Trinity + Oracle + Policy
> + PPO/R-NaD 更新），在单张 6GB 笔记本 GPU 上稳态约 **500 sps** 是正常的，不要直接对标论文数字。

---

## 1. 体检环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 3060 Laptop, 6 GB |
| JAX | 0.12.1 (GPU) |
| 设备数 | 1 |
| 模型 | `ModelConfig()` 默认（4×128） |
| 评测协议 | 固定墙 Duplicate 1v3，`round_mode="half"`, `round_end="confirm"` |

---

## 2. 三个常见 JAX 性能陷阱的体检结果

### 陷阱 A：Batch 不足

| 路径 | 是否命中 | 说明 |
|---|---|---|
| 1v3 `evaluate_duplicate_1v3` | **命中** | 原 `chunk_walls=4` 只有 `4 walls × 4 seats = 16` 条并行 lane |
| 自对战 `train_rl_v6` | 不命中（本机） | `num_envs=32` 时已 compute-bound：32→64 吞吐不变 |
| `run_local_rl` 周期评测 | **命中** | 原 `chunk_walls=4`；已改为 16 |

训练侧实测（单张 3060 6GB，稳态 update 2）：

| `num_envs` | 每 update transitions | 稳态耗时 | 吞吐 |
|---|---|---|---|
| 32 | 8,192 | 16.5 s | 497 sps |
| 64 | 16,384 | 32.8 s | 499 sps |

> 结论：在单张 3060 上加大 batch 不会提速；它只是把每个 update 变长。
> **换到 2×T4（每卡 batch 减半）时结论可能不同**，需要按第 5 节重新扫 batch。

### 陷阱 B：Python 层逐步循环

主路径**没有**这个问题：

- 训练 rollout 在 `lax.scan` 内；
- 1v3 评测现在整体在一个 `lax.while_loop` 内；
- host 同步只发生在每个 update / 每个 chunk 的边界。

唯一例外是交互式路径：`mahjax/ui/match.py::play_out()` 与
`mahjong_rl/eval/review_server.py` 逐 action 执行 `jax.device_get`。
它们适合浏览器交互/复盘，**不要用来批量跑几百个半庄**。

### 陷阱 C：JIT 重编译

| 场景 | 原状态 | 现在 |
|---|---|---|
| `_CompiledDuplicateVsBCEvaluator` | params 作为 jit 参数，checkpoint 间不重编译 | 保持 |
| `evaluate_duplicate_1v3` | actor 权重闭包成常量，每次调用/每条规则重编译 | 仍存在（见第 6 节） |
| 跨进程 | 无持久化编译缓存，每次约 5~13 分钟 | 已启用 `~/.cache/jax_mahjong_rl` |

同一 B=32 训练配置的首个 update 实测：**806.8 s → 148.1 s（5.4×）**。

---

## 3. 第 4 个坑：Trinity 每步 ~32 次 GRP 前向

`mahjong_rl/algo/legal_ron.py::compute_analytical_trinity` 原先每个 env step 调用
`grp_fn` 约 32 次：

- 3 对手 × 3 分数档 single-ron：9 次
- 3 组 double-ron：3 次
- 3 档 self-win：3 次
- 8 种立听配置 × {自听, 未听}：16 次
- 当前局面：1 次

每次 GRP 内部还有 50 轮 Sinkhorn。现在把全部 hypothetical table 的特征堆叠起来，
做**一次 `jax.vmap(grp_fn)`**；当前局面的矩阵由调用方通过 `curr_mat=` 复用。

隔离微基准（32 张 table，真实 GRP 权重）：

| 版本 | 单次耗时 | 吞吐 | 子图编译 |
|---|---|---|---|
| 旧 | 9.65 ms | 3,315 trinity/s | 22.5 s |
| 新 | **2.48 ms** | **12,878 trinity/s** | **9.0 s** |

---

## 4. 1v3 评测的改造

原实现：外层 vmap(walls) → 内层 vmap(seats) → 每条 lane 独立 `lax.scan`，
固定跑满 `max_steps`（即使对局早已结束），且同时计算 candidate 与 baseline 两个 actor。

现实现（`mahjong_rl/eval/duplicate_eval.py::play_duplicate_chunk`）：

1. 把 `(walls × 4 seat rotations)` **展平成一条 lane 轴**，整批送进模型；
2. 整个 chunk 放在一个 `jax.lax.while_loop` 里，所有 lane 终止后立即退出；
3. 末尾不足一批的墙 **padding 到固定形状**，保证一次调用只编译一次；
4. 事件统计（和了/放铳/立直/副露）在 device 上累加，只把最终标量拷回 host。

实测（V6 vs BC，cap 1200）：

| 配置 | 编译+首跑 | 稳态 | 吞吐 |
|---|---|---|---|
| 旧：8 lanes（32 局） | 298.6 s | 58.2 s | 0.50 games/s |
| 新：32 lanes（32 局） | 233.5 s | 19.1 s | 1.67 games/s |
| 新：64 lanes（64 局） | 239.9 s | 22.8 s | **2.81 games/s** |

6GB 卡上 64 lanes 已实测可跑；更小的卡把 `--eval-chunk-walls` 降到 8。

---

## 5. Kaggle T4×2 训练指南

### 5.1 硬件差异

| 项目 | RTX 3060 Laptop | Kaggle 2× Tesla T4 |
|---|---|---|
| 显存 | 6 GB | 16 GB / 卡 |
| 设备数 | 1 | **2** |
| 精度 | FP32 / 无 TF32 | FP32、**FP16 Tensor Core**（无 bf16、无 TF32） |
| CPU/RAM | 本机 | GPU 会话 CPU/RAM 很有限（host 侧操作要省） |
| 会话 | 无限 | 有时限（GPU 配额约 30 h/周） |

### 5.2 多卡

- `run_local_rl --multi-gpu` 默认开启；`jax.local_device_count() == 2` 时会走 `jax.pmap`。
- 约束：`--num-envs % 2 == 0`，`--minibatch-size % 2 == 0`（默认 32 / 512 都满足）。
- `num_envs` 是**全局** batch，每卡只有 `num_envs / 2` 个环境。Kaggle 每卡显存 16GB，
  建议从 **`--num-envs 64`** 起步，再试 **128**，用日志里的 `sps` 判断拐点。
- 1v3 评测 `evaluate_duplicate_1v3` 已经会 `pmap` 到所有设备
  （每次每卡 `chunk_walls` 个墙）；`run_local_rl` 的周期评测目前只用了 device 0（见第 6 节）。

### 5.3 编译缓存（Kaggle 上尤其重要）

```bash
# Kaggle 的 ~ 不跨会话保留；把缓存放到 /kaggle/working，并在会话结束时保存为 Output/Dataset
export JAX_COMPILATION_CACHE_DIR=/kaggle/working/.jax_cache
```

`mahjong_rl/jax_runtime.py` 会优先读取 `JAX_COMPILATION_CACHE_DIR`，否则用
`~/.cache/jax_mahjong_rl`。第二次同 shape 运行可省掉几分钟到十几分钟的编译。

### 5.4 一份推荐的 Kaggle 启动命令

```bash
python -m mahjong_rl.trainers.run_local_rl \
  --ckpt-dir /kaggle/input/sovereign-v6 \
  --out-dir /kaggle/working/sovereign_v6 \
  --num-envs 64 \
  --num-steps 256 \
  --eval-walls 32 \
  --eval-chunk-walls 32 \
  --checkpoint-every 25 \
  --num-updates 500
```

- `--eval-chunk-walls 32` = 每卡 32 墙 × 4 座位 = 128 lanes；16GB 卡可承受。
- 周期 checkpoint 写到 `/kaggle/working`，避免会话超时丢进度；`rl_resume_state.pkl`
  支持断点续训。
- 第一次运行仍会编译；之后同 shape 命中缓存。

### 5.5 评测方差

16 墙 = 64 局，95% CI 通常仍有 ±10 pt 量级。用于选 best checkpoint 建议至少
`--eval-walls 32`（128 局）；最终报告用 100+ 墙。

---

## 6. 已知剩余优化点

1. **`run_local_rl` 周期评测只用 device 0**：`_CompiledDuplicateVsBCEvaluator`
   目前是 `jax.jit` + `jax.device_put`。在 2×T4 上应该像 `evaluate_duplicate_1v3`
   一样 `pmap` 到两张卡，评测时间可接近减半。
2. **`evaluate_duplicate_1v3` 每条规则重新编译**：actor 权重是闭包常量。
   把 `w_rule` / 权重改为 runtime 参数后，`run_phase0` 的 3 条规则可共享一次编译。
3. **eval 同时算 candidate + baseline**：每步其实只有一方行动，理论上还能省
   1.5~2×；需要按 `cp == agent_seat` 分组，形状会变，风险较高。
4. **`encode_obs_v6` 计算两次**：candidate 与 baseline 各自编码一次，可上提共享。
5. **T4 的 FP16**：T4 的 FP16 Tensor Core 明显快于 FP32。若要上混合精度，
   需要保持 Sinkhorn / BCE / 对数域稳定在 FP32，并做 loss scaling；属于较大改动，
   先在分支上做 A/B。
6. **Kaggle host 侧开销**：Oracle replay 每个 update 都会把 `ORA_BUF_KEYS` 搬到 host
   （numpy 单线程），2~4 vCPU 上可能成为瓶颈。可考虑 replay 常驻显存、
   或降低 host 拷贝频率。
7. **`distinct_deck_ratio`**：每个 update 都 `jax.device_get(env_state.round_state.deck)`
   强制同步；可改为在 device 上算哈希/比例，或每 N 个 update 检查一次。

---

## 7. 复现

```bash
# 1) Trinity 正确性
python -m pytest tests/test_v6_legal_ron_and_trinity.py -q

# 2) 端到端流水线（Belief → GRP → BC → RL → Distill → Duplicate 1v3）
python -m pytest tests/test_v6_pipeline_smoke.py -q

# 3) 纯环境吞吐（不含 V6 栈）
python benchmarks/step_speed.py --batch 256 --steps 256

# 4) 训练吞吐：看日志里的 `xxx sps`
python -m mahjong_rl.trainers.run_local_rl --num-updates 2 --warmup-updates 0 \
  --eval-every 0 --checkpoint-every 0 --log-every 1 --no-resume
```
