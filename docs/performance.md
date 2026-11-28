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

> Notebook（`notebooks/sovereign_mahjong_v6_kaggle.ipynb`，cell 16）里的
> `CompiledDuplicate1v3Evaluator` 是独立实现：它已经共享 obs 编码并支持 pmap，
> 但仍是固定 `max_steps` 的 `lax.scan`，**没有提前退出**。它会自动享受
> Trinity 批量 GRP 的收益；若要拿到早停收益，后续可迁移到
> `play_duplicate_chunk`。

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
- 1v3 评测 `evaluate_duplicate_1v3` 和 `run_local_rl` 的周期评测
  (`_CompiledDuplicateVsBCEvaluator`) 都会 `pmap` 到所有设备；
  `--eval-chunk-walls` 的语义是**每卡**并发墙数（每卡 `4*N` 条 lane）。

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
- **建议 `--eval-walls` 是 `--eval-chunk-walls × GPU 数` 的整数倍**（例如 2 卡、
  chunk=16 时用 `--eval-walls 32`），否则最后一批需要 padding，会重复跑一部分墙。
- 周期 checkpoint 写到 `/kaggle/working`，避免会话超时丢进度；`rl_resume_state.pkl`
  支持断点续训。
- 第一次运行仍会编译；之后同 shape 命中缓存。

### 5.5 评测方差

16 墙 = 64 局，95% CI 通常仍有 ±10 pt 量级。用于选 best checkpoint 建议至少
`--eval-walls 32`（128 局）；最终报告用 100+ 墙。

### 5.6 双卡上已修的一个崩溃点

`make_v6_policy_actor` 原本在 `jax.local_device_count() > 1` 时会对传入的
policy/belief/GRP 参数做 `np.asarray(jax.device_get(x))`，以便 pmap 干净复制。
但 `run_local_rl` 的周期评测是在 `jax.jit` **内部**构建 actor 的，此时参数是 tracer，
`np.asarray(tracer)` 会抛 `TracerArrayConversionError` —— 也就是 Kaggle 双卡一开
`--eval-every` 就会在第一次评测时崩。

现在改为：检测到 tracer 时跳过 host 转换，直接使用调用方已经复制/分片好的参数。
回归测试：`tests/test_v6_multigpu_trace.py`（在单卡机器上把
`jax.local_device_count` 伪装成 2 来验证）。

### 5.7 建议的 batch 扫描

Kaggle 每卡只有 `num_envs / 2` 个环境；T4 单卡比 3060 慢，但 16 envs/卡 仍可能偏小。
用 4 个 update 扫一遍；**第 1、2 个 update 各含一次编译，只有第 3/4 个 update 是稳态**：

```bash
for B in 128 256; do
  echo "=== num_envs=$B ==="
  python -m mahjong_rl.trainers.run_local_rl \
    --out-dir /kaggle/tmp/bench_b$B \
    --num-envs $B --num-steps 256 --num-updates 4 --warmup-updates 0 \
    --eval-every 0 --checkpoint-every 0 --log-every 1 --no-resume \
    2>&1 | grep -E "\[RL |envs="
done
```

#### Kaggle 2×T4 实测（2026-11，`minibatch_size=2048`）

| `num_envs` | 每卡 envs | upd 3 稳态 | upd 3 耗时 | 备注 |
|---|---|---|---|---|
| 32 | 16 | **1,174 sps** | 7.0 s | upd 1 ≈ 252 s，upd 2 ≈ 182 s |
| 64 | 32 | **1,486 sps** | 11.0 s | upd 1 ≈ 208 s，upd 2 ≈ 186 s |
| 128 | 64 | **1,767 sps** | 18.6 s | 热缓存 upd 1/2 ≈ 72/61 s |
| 256 | 128 | **1,902 sps** | 34.5 s | 冷缓存 upd 1/2 ≈ 229/206 s |

结论：
- 128→256 只提升 **+7.6%**，边际递减已经很明显；**256 是当前 T4×2 的吞吐上限，不建议再上 512**。
- 不要用均值 `(upd2+upd3)/2` 当稳态：upd 2 含一次性编译，会把结果压低约 2×。
- 第 2 个 update 的编译来源尚未定位；可用 `JAX_LOG_COMPILES=1` 观察是哪个 pmap 图。
- `--minibatch-size` 必须能被 `2` 整除（默认 512/1024 都可以）。

#### 总步数预算与墙钟（重要）

`num_envs` 改变的是**每个 update 的数据量**。要比较“跑完同样多数据谁更快”，用
`sps`；要比较“同样 update 数谁更快”，必须把数据量一起算。

| 方案 | 总环境步数 | updates | 稳态墙钟（不含编译/评测） |
|---|---|---|---|
| B=64 / 1000 upd | 16.4M | 1000 | ~3.1 h |
| B=128 / 500 upd | 16.4M | 500 | ~2.6 h |
| **B=256 / 250 upd** | 16.4M | 250 | **~2.4 h** |
| B=256 / 1000 upd | 65.5M | 1000 | ~9.6 h（接近/超过单次 Kaggle 会话舒适上限） |

如果选择 B=256 但想保持原来的训练预算（约 16.4M 步），把所有**以 update 计**的
调度按 `64/256 = 1/4` 缩放：

```text
RL_NUM_UPDATES            1000 -> 250
RL_WARMUP_UPDATES           80 -> 20
RL_PFSP_SNAPSHOT_INTERVAL  200 -> 50
RL_EVAL_EVERY              250 -> 62
RL_VERSION_EVERY           250 -> 62
```

B=128 则按 `1/2` 缩放（500 / 40 / 100 / 125 / 125）。

#### batch 和“更新预算”要一起看

`sps` 只是吞吐；每个 update 的数据量是：

```text
transitions_per_update = num_envs × num_steps
wall_time ≈ total_transitions / sps
```

- **固定总环境步数**（推荐）：B=128 的 `sps` 比 B=64 高约 1.19×，所以同样的
  总步数会快约 1.19×。
- **固定 `num_updates`**：B 加倍 = 每个 update 的数据量和梯度 minibatch 数都加倍，
  墙钟也近似加倍。以 1000 updates 为例：
  - B=64（32/GPU）：16.4M steps，约 11.0s/update → **~3.1 h**
  - B=128（64/GPU）：32.8M steps，约 18.6s/update → **~5.2 h**
- 如果你只想把原来 B=64/1000 updates 的预算跑快，改成 B=128 时把
  `RL_NUM_UPDATES` 减半到 500，并把所有**以 update 计**的调度一起减半：
  `RL_WARMUP_UPDATES` 80→40、`RL_PFSP_SNAPSHOT_INTERVAL` 200→100、
  `RL_EVAL_EVERY` 250→125。这样总步数和训练节奏基本不变，墙钟约快 1.19×。
- 如果你愿意用更大的 batch 换更多数据/梯度步，保持 1000 updates 也可以，
  但要接受 ~5.2 h 的训练时间。

### 5.8 Kaggle 会话与磁盘

- `rl_resume_state.pkl` 包含 policy/BC/ρ-base/Oracle/Belief 权重、Adam 状态、
  PFSP pool 和 outer snapshots，会随训练变长而变大。`--checkpoint-every 25`
  在 Kaggle 上可能每次写几十~上百 MB；长跑建议 `50~100`。
- 把 `--out-dir` 指向 `/kaggle/working/...`；会话超时后用同一目录 `--resume`
  续训（默认开启）。
- `--ckpt-dir` 指向只读的 `/kaggle/input/<dataset>` 即可。

### 5.9 Kaggle 上线检查清单（按顺序执行）

- [ ] **回归**：`python -m pytest tests/test_v6_multigpu_trace.py -q`
      （确认双卡 tracer 修复生效）
- [ ] **缓存**：`export JAX_COMPILATION_CACHE_DIR=/kaggle/working/.jax_cache`
      （会话结束可把它作为 Output/Dataset 保存，下次继续复用）
- [ ] **batch sweep**：`B=128 / 256` 各跑 4 个 update，记录第 3/4 个
      update 的 `sps`（前两个含编译）；Notebook 的 **2.5 节**已内置这段扫描
- [ ] **评测参数**：`--eval-chunk-walls` 是**每卡**墙数；
      `--eval-walls` 取 `chunk_walls × GPU 数` 的整数倍，避免 padding 浪费
- [ ] **checkpoint**：`--checkpoint-every 50~100`，输出到 `/kaggle/working`
- [ ] **正式训练**：`--num-envs` 取 sweep 的拐点值（2×T4 实测 **128** 最好，再试 256）

当前 Notebook（`notebooks/sovereign_mahjong_v6_kaggle.ipynb`）已经：
- 默认拉取 `perf/jax-throughput-kaggle` 分支（合并回 main 后改回 `main`）；
- 在 2.5 节提供可直接运行的 batch sweep（解析 `sps` 并汇总）；
- T4×2 预设为 `RL_NUM_ENVS=128`（每卡 64）。

### 5.10 训练稳态的时间构成（实测）

用 `MAHJAX_PROFILE=1` 跑 3 个 update 可以打印每个阶段的耗时：

```bash
MAHJAX_PROFILE=1 python -m mahjong_rl.trainers.run_local_rl \
  --num-updates 3 --warmup-updates 0 --eval-every 0 --checkpoint-every 0 --no-resume
```

**阶段级（B=32，单卡 3060，稳态 update 2/3）：**

| 阶段 | 占比 |
|---|---|
| rollout | ~43% |
| policy update (PPO + R-NaD) | ~42% |
| oracle update | ~10–14% |
| prepare（含 BC/rho 全量前向） | ~1.5% |
| host probe / metrics / health | <1% |

**rollout 内部单步组件（B=128，单卡 3060；各组件单独 jit，非严格可加）：**

| 组件 | 耗时 |
|---|---|
| `env.step`（合法动作） | 9.6 ms |
| trinity（4 次 batched GRP） | 3.8 ms |
| policy forward（self-play；PFSP 再 ×1） | 3.5 ms |
| belief forward | 2.4 ms |
| oracle forward | 2.1 ms |
| `encode_oracle_planes` | 1.9 ms |
| `grp.apply`（EMA next） | 1.7 ms |
| `encode_obs_v6` | 1.1 ms |

注意：`env.step` 在 128 envs 时受占用率限制；官方
`benchmarks/step_speed.py --batch 256` 的 `half` 是 **6.25 ms/step（256 envs，
约 41k env-SPS）**，到 256 envs/卡时 env 的占比会下降。

结论：
- **训练的大头是模型 forward + policy update**：rollout 中 belief/policy/oracle/
  trinity/grp_ema 合计约 55–60%，env + obs 约 40%；policy update 又占 update 的 42%。
- `prepare` 虽然包含一次全量 BC 前向，但只占 ~1.5%，不是优化重点。
- `lax.scan(unroll=2)` 实测无收益（rollout 5.31s vs 5.07s，编译时间翻倍），不采用。

---

## 6. 已知剩余优化点（按收益排序）

1. **FP16 / mixed precision（T4 上最大的单项）**：rollout 的模型前向约占 rollout 的
   55%，policy update 又占 update 的 42%；T4 的 FP16 Tensor Core 远快于 FP32。
   建议分两步：
   - 先只对 **rollout 前向** 用 FP16（输出 cast 回 FP32，无 loss scaling），风险低；
   - 稳定后再对 policy/oracle 的 forward+backward 用 FP16 + FP32 master weights +
     loss scaling。
2. **Trinity 的 4 次 GRP 调用合并成 1 次**：把 current + 31 张 hypothetical table
   一次性 vmap，减少 3 次 kernel launch 并提高小 batch（3/16）的占用率，预计 3–5%。
3. **`grp_ema_next` 延迟到 round-end 行**：`lambda_inter=0` 时它只在回合结束用到，
   现在每个 step 都算（1.7 ms/step）。改为存 29 维 GRP 特征、在 prepare 里只对
   round-end 行做 GRP，预计 2–3%。
4. **Oracle replay 常驻显存**：每个 update 把 13 个 key 搬到 host 再采样上传；
   在 Kaggle 的弱 CPU 上预计 5–10%。
5. **B=512（256/卡）**：进一步提高 env 占用率，预计 3–5%，但要确认 16GB 显存。
6. **`evaluate_duplicate_1v3` 每条规则重新编译**：把权重/w_rule 改成 runtime 参数，
   phase0 的 3 条规则可共享一次编译。
7. **notebook 的 `CompiledDuplicate1v3Evaluator` 仍是固定 `max_steps` 的 scan**：
   迁移到 `play_duplicate_chunk` 可拿早停收益。
8. **pmap 的第二次编译**（update 2 ≈ 200s）：一次性，不占总稳态；用
   `JAX_LOG_COMPILES=1` 定位后可再省 ~3 分钟/新 shape。
5. **Kaggle host 侧开销**：Oracle replay 每个 update 都会把 `ORA_BUF_KEYS` 搬到 host
   （numpy 单线程），2~4 vCPU 上可能成为瓶颈。可考虑 replay 常驻显存、
   或降低 host 拷贝频率。
6. **`distinct_deck_ratio`**：每个 update 都 `jax.device_get(env_state.round_state.deck)`
   强制同步；可改为在 device 上算哈希/比例，或每 N 个 update 检查一次。
7. **指标回传**：每个 update 有 ~8 个单独标量的 `device_get`，可合并成一次
   `jax.device_get(tuple_of_scalars)`，减少 host 同步次数。

---

## 7. 复现

```bash
# 1) Trinity 正确性
python -m pytest tests/test_v6_legal_ron_and_trinity.py -q

# 2) 端到端流水线（Belief → GRP → BC → RL → Distill → Duplicate 1v3）
python -m pytest tests/test_v6_pipeline_smoke.py -q

# 3) 双卡 trace 回归（单卡机器上伪装 2 个设备）
python -m pytest tests/test_v6_multigpu_trace.py -q

# 4) 纯环境吞吐（不含 V6 栈）
python benchmarks/step_speed.py --batch 256 --steps 256

# 5) 训练吞吐：看日志里的 `xxx sps`
python -m mahjong_rl.trainers.run_local_rl --num-updates 2 --warmup-updates 0 \
  --eval-every 0 --checkpoint-every 0 --log-every 1 --no-resume

# 6) 在单卡机器上验证 pmap 控制流（2 个 CPU 设备）：
#    结果应与单设备完全一致
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
  python -m pytest tests/test_v6_multigpu_trace.py -q
```

> 双卡 pmap 的 `evaluate_duplicate_1v3` 与 `_CompiledDuplicateVsBCEvaluator`
> 已用上述 CPU-2-device 方式在单卡机器上验证过：统计结果与单设备逐位一致。
