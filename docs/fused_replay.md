# SGA fused replay — 离线重放加速引擎

> 对应代码: `sga/replay_fused.py` (引擎) + `sga/replay.py` (命令行入口、dense 缓存) + `sga/trace_convert.py` (bf16 副本转换)。
> 读者: 需要用离线重放做 SGA 策略扫描的同学。§5 精度与 §6 使用手册不依赖 §4 的实现细节, 可直接跳读。

## 1. 概述

SGA (selective gradient accumulation, 选择性梯度累积) 训练里, 一个策略由**打分函数 × 通信比例 × 周期长度**三元组描述。评价策略的标准做法: 让训练从同一个分支点出发、按该策略跑一段固定窗口, 用**偏离度** (distortion, §2.3) 给分。策略扫描要在几十个候选里选优, 就得对每个候选各评价一遍 —— 这是主要成本: 在线做, 每个候选是一场完整训练。

fused replay 是 SGA 仓库的离线重放加速引擎, 分两步压缩这份成本:

1. **在线改离线**: 训练时把每步梯度录制下来 (trace), 之后任意策略在录制数据上确定性重放 —— 免前向反向, 结果可逐位复现;
2. **逐配置改单趟流式**: 旧离线引擎每个策略独立重放一遍窗口; fused 引擎让一组 (默认至多 12 个) 策略共享**同一次窗口遍历**, 参数空间切分到多卡并行推进 (参数分片), dense 基线走缓存、每份 trace 只算一次。

实测 (llama-1B, 1000 步窗口, 8×A100, §5.3): 72 配置扫描从在线的 ~42 小时、逐配置离线的 ~2 天, 降到 **~12.6 小时**; 数值上与逐配置重放逐位一致 (72/72 配置偏离度相对差 = 0, 选点阈值逐位相等), 离线选出的最优策略与在线实验结论一致。

## 2. 背景: SGA 策略与偏离度

### 2.1 SGA 周期掩码

SGA 是大模型分布式训练的梯度通信压缩方法: 训练正常前向/反向, 但**梯度同步**不再每步传全部坐标, 而是 ——

1. 用打分函数给每个参数坐标一个重要性分数 (score, 如按梯度绝对值 `abs_g`、按动量绝对值 `abs_m`);
2. 每个周期开始时选出分数最高的前 k 个坐标组成**掩码** (mask), k = 通信比例 ratio (%) × 总坐标数 N;
3. 周期内每步只同步掩码内坐标; 掩码外坐标的梯度不清零, **累积为残差** (residual), 到下个周期边界折叠进全量同步;
4. 优化器状态 (AdamW 一阶/二阶动量) 各自轨道正常更新。

策略 = (打分函数 score, 通信比例 ratio, 周期长度 cycle_len)。研究问题: 哪个组合在压缩通信的同时损失最小? 例如 4 种分数 × 3 种比例 × 6 种周期 = **72 个配置**。

### 2.2 三种评价途径

| 途径 | 梯度来源 | 一场评价几个策略 | 成本特征 |
|---|---|---|---|
| 在线实验 | 每步真实前向反向 | 1 | 每策略一场完整训练, 混入数据顺序/调度噪声 |
| 离线逐配置重放 | 录制 trace | 1 | 免前向反向; 窗口数据被反复重读, 单卡串行 |
| fused replay | 录制 trace | 一组 ≤ 12 | 窗口每组只读一遍, 多卡分片并行, 完全确定 |

### 2.3 偏离度 (distortion)

设分支点 (branch point, 训练全局步号, §6.1) 处权重为 `w_start`, 取**dense 参照轨迹** (ratio=100%、每步全量同步的基线, 由 dense 臂重放产生)。策略重放 h 步后的权重为 `w_sparse(h)`, dense 参照同步重放 h 步后为 `w_dense(h)`, 定义

```
distortion(h) = ‖w_sparse(h) − w_dense(h)‖₂  /  ‖w_dense(h) − w_start‖₂
```

即策略轨迹相对 dense 轨迹的偏离, 除以 dense 轨迹自身位移作归一 (便于跨窗口/跨模型比较)。h 称为**视界** (horizon), 一场重放可同时报告多个视界 (如 1/10/100 步)。distortion 越小策略越好; 扫描决策 = 在配置空间取最小者 (argmin) 或按 distortion 排序。

## 3. 方法演化: 在线实验 → 逐配置离线重放 → fused replay

三代的差别只有两处本质: **梯度从哪来** (重算还是读录制) 与 **窗口数据读几遍** (每配置一遍还是每组共享一遍)。三代伪代码用统一命名: `score_fn`(打分函数注册表), `k`(该配置每周期通信的坐标数), `mask`(掩码), `residual`(残差), `update`(待提交的裁剪前更新), `clip_coef`(裁剪系数), `w/m/v`(AdamW 参数与一阶/二阶动量), `adamw_step_inplace`(AdamW 原地步进, 真实函数名)。周期边界 = 每 `cycle_len` 步一步, 该步折叠残差做全量同步并重选掩码。

### 3.1 在线实验

每个策略各起一场真实训练, 从同一 checkpoint 出发:

```text
# 在线实验: 每个配置 = 一场完整训练
for config in all_configs:                          # 例如 72 个配置
    model, optimizer = load_llama_1b(checkpoint)    # 从同一分支点快照恢复
    w, m, v = optimizer 的参数与 AdamW 状态
    k = round(config.ratio / 100 * N)               # N = 总坐标数
    residual = zeros_like(w)
    grad = 分支点处快照自带的梯度                    # 初始选点用
    mask = select_topk(score_fn[config.score_name](w, m, v, grad), k)

    for step in window_steps:                       # 窗口, 如 1000 步
        loss = model(next_batch())                  # 前向
        loss.backward()                             # 反向: 梯度每步重算
        grad = w.grad
        if step 是周期边界:
            update = grad + residual                # 折叠残差 → 全量同步
            residual = 0
        else:
            update = grad * mask                    # 只同步掩码内坐标
            residual += grad * (~mask)              # 掩码外累积, 延后同步
        clip_coef = min(1, clip_grad / (norm(update) + 1e-6))
        if step 是周期边界:                          # 用本步待提交更新重选, 供下周期
            mask = select_topk(score_fn[config.score_name](w, m, v, clip_coef * update), k)
        adamw_step_inplace(clip_coef * update, m, v, w)

    # distortion 需要 dense 基线 (ratio=100%) 作参照 → 又是一场训练
```

成本: llama-1B、8×A100、1000 步约 35 分钟/场, 72 配置 ≈ **42 小时**; 且各场混入数据顺序/通信时序噪声, 配置间不严格可比。

### 3.2 逐配置离线重放

关键观察: AdamW 逐坐标独立, "同步哪些坐标"只影响每个坐标自己的更新序列 —— 因此**录下一次训练每步的梯度, 任何策略都能在录制数据上离线复算**, 不再需要模型、数据与前向反向。录制产物即 **trace** (目录布局, `_g.pt` 是每步一个的梯度文件):

```
trace_w1000/
  catalog.pt              # 参数清单: 名字、元素数、所属超参组
  step_000040_state.pt    # 分支点 40 的完整优化器状态 (w/m/v + 当步梯度)
  step_000041_g.pt        # 第 41 步的全部坐标梯度 (每步一个文件)
  ...
```

N ≈ 1.2×10⁹ 时每步梯度 fp32 约 4.7 GB, 1000 步窗口 ≈ 4.7 TB。逐配置重放:

```text
# 逐配置离线重放: 梯度读录制文件; 每个配置把窗口的梯度文件读一遍
for config in all_configs:                          # 72 个配置
    w, m, v = load_state(trace, branch_point)       # 从录制快照出发
    k = round(config.ratio / 100 * N)
    residual = zeros_like(w)
    mask = select_topk(score_fn[config.score_name](w, m, v, 分支点梯度), k)

    for step in window_steps:                       # 1000 步, 每步读一个 ~4.7 GB 文件
        grad = load_grad(trace, step) / clip_coef_recorded   # ← IO; 除回录制端裁剪 (见下)
        if step 是周期边界:
            update = grad + residual;  residual = 0
        else:
            update = grad * mask;  residual += grad * (~mask)
        clip_coef = min(1, clip_grad / (norm(update) + 1e-6))
        if step 是周期边界:
            mask = select_topk(score_fn[config.score_name](w, m, v, clip_coef * update), k)
        adamw_step_inplace(clip_coef * update, m, v, w)

    distortion[config] = 距离(w, dense_reference)    # dense 臂: 同窗口再读一遍 (可缓存)
```

`clip_coef_recorded`: 录制时训练端对每步梯度做过全局范数裁剪 (乘了系数才落盘), 重放先除回该系数还原成裁剪前梯度, 之后每步再按与在线相同的公式独立裁剪 —— 两段裁剪的语义见 §4.4。

成本: 免前向反向, 但瓶颈是 IO —— 共享文件系统有效读取约 2 GB/s, 4.7 TB 窗口读一遍约 40 分钟, 72 配置需读几十遍, ≈ **2 天**; 且单卡串行。

### 3.3 fused replay: 单趟流式重放

fused 引擎把"逐配置外层循环"翻到内侧: **一组配置** (group, 任意 (score, ratio, T) 组合混装 — 每 config 自带周期长度, 由 `--cycle-lens` 集合展开; 至多 12 个, 由 `--fused-max-configs` 控制, 或按 `--fused-state-budget-mb` 常驻状态字节预算切组; 例如 4 分数 × 3 比例 × 3 周期恰好 6 组) 在**同一次窗口遍历**中齐头并进; 参数空间按连续坐标区间切分到分片进程 (每卡一片); 每步跨分片只交换标量:

```text
# fused: 一组配置共享一次窗口遍历; 参数坐标空间连续分片, 每卡一个分片
group = 本场同时推进的配置集合 (混合周期长度, 至多 max_configs 个)
dense_cache = load_or_populate(trace, branch_point, window)   # 父进程, 每份 trace 一次

for shard in shards (并行; 分片 r 用第 r 块 GPU, 持连续坐标区间 [lo, hi)):
    # 每配置一份独立优化器状态, 都只含本分片的坐标
    states = {config: (w, m, v ← 分支点快照的 [lo, hi) 切片;
                       residual ← 0;  mask ← 分支点选点的本片切片)}
    prefetcher = GPrefetcher(...)                  # 后台线程预取本片的每步梯度 (§4.5)

    for step in window_steps:                      # 窗口每个梯度文件每组只读一遍
        grad_shard = prefetcher.next()[lo:hi] / clip_coef_recorded
        boundary_c = (si % T_c == 0 对每 config 判定)   # per-config 周期日历

        # pass 1: 每配置的 clip 范数 — 跨分片只传标量
        for config in group:
            pending = grad_shard + states[config].residual   if boundary \
                      else grad_shard * states[config].mask
            norm_sq = 本片 Σ pending²
        global_norm[config] = sqrt(allreduce_fp64(norm_sq))   # 12 配置 = 12 个 float
        clip_coef[config] = min(1, clip_grad / (global_norm[config] + 1e-6))

        # pass 2: (仅周期边界) 用待提交更新重算分数, 跨分片精确 top-k 重选
        if boundary:
            for config in group:
                scores = score_fn[config.score_name](w, m, v, clip_coef[config] * pending)
                states[config].mask = shard_select(scores, k, comms)

        # pass 3: 提交 (与在线/逐配置版逐位同构)
        for config in group:
            if boundary:  update = (grad_shard + residual) * clip_coef;  residual = 0
            else:         update = grad_shard * mask * clip_coef
                          residual += grad_shard * (~mask)
            adamw_step_inplace(update, m, v, w)
            if step 是某视界步 (horizon, 如第 100/1000 步):
                num_sq[config] += Σ ‖w − dense_cache 的该视界快照‖²  → allreduce

distortion[config][horizon] = sqrt(num_sq) / dense_cache 的分母
```

对第二代的三处结构变化:

| 手段 | 逐配置 | fused |
|---|---|---|
| 窗口遍历 | 每配置一遍 (+dense 臂一遍) | 每组一遍; dense 臂永远走缓存 |
| 参数布局 | 单进程持全量参数 | 分片持连续坐标区间, 每卡只放本区间的全部配置状态副本 |
| 跨进程通信 | 无 (单进程) | 每步每配置 1 个 fp64 标量; 边界步加一次 32769 桶整数直方图 allreduce |

**分片的意义**。按参数坐标切、而不是按配置切, 是读放大与显存的同时最优: 每个分片 mmap 同一批梯度文件、只读自己区间的页, 宿主页缓存把 NFS 拉取去重 (每个文件每趟全机只拉一次); 组内 12 个配置对**同一片数据**在各自 GPU 上并行推进, 一遍读取服务整组。若按配置分卡, 每卡要么全量读窗口 (读放大等于组内配置数倍), 要么引入跨卡分发。连续区间切分还有正确性红利: 全局坐标序恰好 = 分片序 + 片内序, 掩码并列层按此序补齐, 选点确定可复现 (§4.3)。

**数学等价性**。对任一配置, fused 引擎执行的运算序列 —— 逐参数、逐 chunk 的操作种类与顺序、裁剪公式、AdamW 步进 —— 与逐配置重放完全相同; 分片只改变数据驻留位置, 不改变配置内部的运算。跨分片交换仅两类 (fp64 标量求和、整数直方图), 归并顺序差异属于浮点纯重排类, 量级 ≤ 1×10⁻⁶ (§5)。因此: 单分片运行与逐配置引擎的完整 JSON 报告**逐位一致** (真实 trace 上验证); 多分片偏差被预注册精度带覆盖, 实测 72/72 配置相对差为 0 (§5.1)。

### 3.4 三代对比

| 维度 | 在线实验 | 逐配置重放 | fused replay |
|---|---|---|---|
| 梯度来源 | 每步前向反向重算 | 录制 trace | 录制 trace |
| 一场评价配置数 | 1 | 1 | 一组 ≤ 12 (同周期长度; 扫周期轴 = 多场运行) |
| 窗口读取次数 | — (重算) | ≈ 配置数 + 1 遍 | ≈ 配置数/12 + 1 遍 (dense 走缓存) |
| 72 配置耗时 | ~42 小时 | ~2 天 (IO 下限) | **~12.6 小时** (6 场 × 12 配置; 冷趟另付 dense 填充) |
| 可复现性 | 混入运行噪声 | 确定性 | 确定性, 与逐配置逐位一致 |

## 4. 核心实现

代码摘自 `sga/replay_fused.py` (为讲解有裁剪, 语义不变)。

### 4.1 入口与 dense 产物缓存

```python
def run_fused_branch(args, ctx, t0):
    """重放一个分支点; 返回与逐配置重放完全相同结构的报告字典。"""
    # gates: 仅实现 --replay-mode cycle; 拒绝 --self-check; cycle 必须给 --dense-cache
    # 父进程在拉起任何分片之前探测/生成 dense 缓存:
    ident = R._dense_cache_ident(ctx["trace_dir"], t0, window_steps,
                                 eff_horizons, replay_dtype,
                                 args.trace_dtype, engine_variant)
    hit = R.load_dense_cache(args.dense_cache, ident) is not None
    if not hit:
        _compute_dense_and_cache(cfg, R, t0, ...)   # 冷缓存: 算一遍 dense 臂并落盘
    cfg["dense_cache_hit"] = hit                    # 报告如实记录 miss/hit

    if world == 1:
        _worker(0, 1, cfg, 0, frag_path)            # 单分片: 同进程内跑
    else:
        mp.spawn(_worker, args=(world, cfg, port, frag_path), nprocs=world)
    return torch.load(frag_path, ...)               # 分片 0 写的完整报告
```

dense 缓存身份键 (ident) 含 trace 路径 + 分支点 + 窗口步列 + 视界集 + 重放精度 + trace 精度 + 语义版本 + 引擎变体 (cpu/dev) —— 任一项变化键即变化, 旧缓存自然失效, 永不误命中。父进程填缓存的原因: 冷缓存时 dense 臂要算数小时, 若发生在分片进程组内, 等待的分片会撞进程组超时; 父进程先算好, 分片起来后只做切片消费。

### 4.2 参数分片

```python
def shard_bounds(N, world):
    """全局平坦索引空间 [0, N) 连续近均分; 分片 r 拥有 [lo_r, hi_r)。
    连续切分保证 '并列分数按全局索引序补齐' 的选点规则在分片后仍成立。"""
    base, rem = divmod(N, world)
    out, lo = [], 0
    for r in range(world):
        hi = lo + base + (1 if r < rem else 0)
        out.append((lo, hi)); lo = hi
    return out

class ShardParams:
    """与 [lo, hi) 相交的参数表: 每参数在本片的偏移/元素数,
    及其自带 AdamW 超参 (beta1/beta2/eps/权重衰减/学习率组)。"""
```

分片边界按全局展平坐标 (参数目录序) 切, 单个参数可跨分片 —— 每片只持自己那段, 超参随参数条目走, AdamW 逐参数用各自的 β1/β2/eps/wd/lr。

### 4.3 分布式精确 top-k: shard_select

问题: 分数向量 (长 N ≈ 1.2×10⁹, 非负 fp32) 按坐标分片在各卡, 要选全局前 k, **恰好 k 个**、并列按全局坐标序补齐。全量 allgather 不可行 (单卡 4.8 GB × world), 全局排序不可行。解法: **位模式整数直方图** —— 非负 fp32 的位模式与数值同序, 取位模式高 16 位 (指数 + 高位尾数) 当桶号, 只有 32769 个可能值, 直方图 allreduce 的代价与 N 无关:

```python
def shard_select(score, k, comms):
    """跨分片选全局 top-k; 返回 (本片布尔 mask, 全局阈值)。
    与单进程在拼接向量上选择逐位一致。"""
    top16 = ((score.view(torch.int32) & 0x7FFFFFFF) >> 16)   # 位模式高 16 位
    hist = torch.bincount(top16, minlength=32769).cpu()
    if comms.distributed:
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)          # 整数直方图相加: 精确
    cum = torch.cumsum(hist, 0)
    b = int(torch.searchsorted(cum, total - k + 1).item())   # 第 k 大所在的桶
    # 只有该桶的候选值需要跨分片收集 — 通信量从 4.8 GB 降到 KB 级
    cand = score[top16 == b].cpu().clone()
    whole = torch.cat(comms.all_gather_pad(cand))
    thresh = float(torch.kthvalue(whole, kk).values.item())  # 精确第 k 值 (逐位)
    # 并列补齐: 恰等于阈值的坐标可能超过剩余名额, 按全局坐标序
    # (= 分片序 + 片内序) 取前若干个, 与单机选择规则一致
    ...
    return gt_mask, thresh
```

桶内用真实值精化, 阈值因此**逐位精确**, 不受直方图粒度限制。两点工程细节: 入口拒绝 NaN/Inf 分数 (它们会落进直方图最高桶, 静默挤掉真实坐标); 并列计数按 64M 元素分块累加, 避免一次性物化 12 亿 int64 中间量 (单片 9+ GB, 会 OOM)。

**与在线引擎的关系**。在线训练侧选点实现在 `sga/plugins/selection/_histogram.py`, 是同一族算法 (同样的 shift-16 / 32769 桶几何), 但契约不同, 当前为两份独立实现:

| | 在线引擎 (`_histogram.py`) | fused 引擎 (`shard_select`) |
|---|---|---|
| 输入分布 | DP allreduce 后每卡持**同一份全量**分数 | 分数按坐标**分片**, 每卡只有本段 |
| 直方图 allreduce | 可跳过 (每卡本地直方图 × world 即全局, 有门控开关) | 必需 |
| 并列处理 | 并列坐标**全选** (允许略多于 k) | 按全局坐标序补齐到**恰好 k** |
| 返回 | 选点掩码 | 本片掩码 + 全局阈值 (报告字段) |

能否复用: 不能原样复用 —— 在线版输入是复制、并列全选、只要掩码; 离线版输入分片、需要恰好 k 的确定性掩码和可跨运行比对的全局阈值。并列策略差异的原因: 在线训练并列多选几个坐标无害; 离线重放是测量工具, 必须对"选了哪些坐标"给出确定答案 —— 恰好 k + 固定并列序保证同一 trace 任意分片数、任意次重放, 选点逐位一致。设计约束是两侧选出的阈值逐位可比。

### 4.4 流式主循环

分片起来后先做一次性**流前准备**: 从分支点快照克隆本片状态; 按注册表公式算分支点分数 (fp64 逐元素 → 存 fp32) 并对每个 (分数, 比例) 跑 `shard_select` 得初始掩码; 从 dense 缓存切出各视界的参照快照与 label 向量; 算 NDCG/recall 等排序指标。CUDA 下流前准备整体设备化 (选点与掩码索引在设备副本上执行, 载荷收集在单机时走 NCCL 快道, 主机原件仍供 fp32 求和等主机归约, 报告与纯主机路径逐字节相等; 机制与运维开关见 §6.7 末段)。之后按组进入流式循环, 每步三遍:

```python
for si, (_gstep, gmeta, local) in enumerate(prefetcher):     # 预取线程后台拉下一步
    coef_dense, _ = R.coef_dense_for(gmeta)   # 录制的是裁剪后梯度, 先除回还原
    # per-config 周期日历: config (s, r, T_c) 在 si % T_c == 0 处到边界
    # (折叠残差 + 全量同步 + 重选); 组内混合 T 时各 config 日历独立判定,
    # 到达边界的 config 集合由日历确定性推出 → 全 rank 一致
    boundary_c = [si % t == 0 for (_s, _r, t) in group]

    # pass 1 — 每配置裁剪范数: 本片平方和 → 跨分片 fp64 标量 allreduce
    #          (每配置每步只通信 1 个标量; 12 配置 = 12 个 float)
    qn2 = [本片逐参数×chunk累加的 ‖pending‖²]
    qn2 = comms.sum_f64(qn2)
    coefs = [min(1.0, clip_grad / (math.sqrt(q) + 1e-6)) for q in qn2]

    # pass 2 — 边界重选 (per-config: boundary_c[ci] 且 si > 0 且 si + 1 < W):
    #   待提交更新 (grad/coef_dense + residual) * coef 代入分数公式 → shard_select
    for ci, hit in enumerate(boundary_c):
        if hit and si > 0 and si + 1 < W:
        score = spec.formula(ws, ms, vs, (gd + res) * coef, ...,
                             update_includes_wd=True).abs()
        newsel, _th = shard_select(score, ks[r], comms)

    # pass 3 — 提交: AdamW 原地步进 + 残差记账 + 视界分子
    for config in group:
        if boundary:  q = (gd + res) * coef;  res = 0        # 全量步
        else:         q = gd * sel * coef;  res += gd * (~sel)  # 稀疏步
        R._adam_step_inplace(q, ms, vs, ws, ...)             # 与在线引擎同一实现
        if (si + 1) in horizons:
            num_sq += Σ ‖ws − dense_snapshot[horizon]‖²   → allreduce
```

三条语义契约是等价性的根:

1. **两段裁剪还原**: 录制的梯度是在线训练裁剪**之后**的值 (录制端已乘裁剪系数), 重放先除回该系数得到裁剪前梯度; 每步再对每个配置的待提交更新按与在线相同的公式独立裁剪 —— 与在线 SGA 训练实际发生的裁剪语义一致。
2. **边界重选用裁剪后梯度打分**: 待提交更新 (还原梯度 + 残差) × 裁剪系数代入分数注册表公式 (`update_includes_wd=True`, 即在线全同步选点分数), 离线选点与在线选点同源。
3. **同一实现基座**: AdamW 步进函数与分数注册表为在线训练、在线 shadow 对照、离线重放三方共用 (`sga/plugins/scores.py`) —— 轨迹差只能来自策略本身, 不来自实现差异。

边界重选的两个豁免: 首边界 (窗口第 0 步) 不重选, 分支点掩码已在流前算好; 最后一个边界不重选, 此后不再有稀疏步消费新掩码。

### 4.5 预取线程 GPrefetcher

```python
class GPrefetcher:
    """后台线程把本片每步的梯度切片物化为连续 fp32 CPU 张量;
    缺页 + 升精度 + 切片与当前步计算重叠, 计算侧只付一次 H2D 拷贝。"""
    def __init__(self, g_files, window_steps, sp, upcast_bf16, depth=2):
        self.q = queue.Queue(maxsize=depth)        # 背压: 预取最多领先 2 步
        self.exc = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for gstep in self.window_steps:
            d = _load(self.g_files[gstep])          # mmap 读 → 触发缺页 (NFS 拉页)
            pl = dict(d["payload"])
            for k in [bf16 payload]:                # bf16 副本: 载入即升回 fp32
                pl[k] = pl[k].float()
            local = {}
            for e in self.sp.entries:               # 只物化本片的坐标切片
                local[e["name"]] = pl[e["name"]].reshape(-1)[off:off+n].clone()
            self.q.put((gstep, d["meta"], local))

    def __iter__(self):
        for _ in self.window_steps:
            item = self.q.get()
            if item is None:
                raise RuntimeError("g prefetch failed") from self.exc
            yield item
```

三个细节:

1. **重叠的内容**: 梯度文件是 mmap 读的, 真正代价是缺页 (NFS 把页拉进宿主缓存)、bf16→fp32 升精度、本片区间切片物化 (`.clone()`) —— 三件事全在后台线程; 计算线程拿到现成 fp32 切片, 只再付 host→device 拷贝。
2. **背压**: `Queue(maxsize=2)` 让预取最多领先 2 步。一步全参数 fp32 梯度是 GB 级, 不设上界会撑爆宿主内存; 深度 2 已够掩盖单步缺页延迟。
3. **异常哨兵**: 后台线程出错时记录异常并 `q.put(None)`; 消费端收到 None 立即 `raise ... from self.exc`。没有哨兵, 计算线程会永远阻塞在 `q.get()`, 表现为无报错挂死。

## 5. 精度

### 5.1 等价性判据与实测

精度主张分三层, 由严到宽:

1. **数值层**: 单分片 fused 与逐配置引擎的完整 JSON 报告**逐位一致** (真实 trace 上验证, 唯一允许差异是 `dense_cache_hit` 来源标记); 多分片唯一偏差源是跨分片 fp64 标量求和的加法顺序 (纯重排类, 量级 ≤ 1×10⁻⁶)。实测 72/72 配置在视界 1000 步的 distortion 相对差 **= 0**, 选点阈值 **72/72 逐位相等**, 同配置重跑互差 rel = 0。
2. **决策层**: 扫描的最终产出是"选哪个策略"。实测离线 argmin 与在线实验 argmin 严格一致 (均为 `abs_m` × 比例 40% × 周期 5, 1/1); 按 distortion 排序严格一致 (18/18)。
3. **决策安全余量**: 等价噪声量级 (1×10⁻⁶) 与扫描中最接近的两个配置间的真实 distortion 差距 (rel 3.9×10⁻⁴) 相差 **393 倍** —— 全部数值噪声加起来也不可能颠倒任何两个配置的优劣次序。这是"离线扫描结论可信"的定量依据。

### 5.2 bf16 筛选副本

trace 默认 fp32 录制 (4.7 TB/千步)。`sga.trace_convert` 可将其转成 **bf16 副本** (逐张量 round-to-nearest; 转换实测 upcast 相对误差 max 3.891×10⁻³, 恰为 bf16 数学上界 2⁻⁸ = 3.906×10⁻³), 副本体积减半 (2.5 TB)。三个常见问题:

- **如何开启**: 先 `trace_convert --src <fp32 trace> --dst <bf16 副本>`; 重放对副本目录加 `--trace-dtype bf16`。载入时自动升回 fp32, 打分/重放算术全程 fp32/fp64 不变 —— bf16 只存在于存储层。忘加 flag 会直接报错提示。
- **是否默认**: 不是。默认 `--trace-dtype fp32`, bf16 是可选筛选轨道。
- **实验用不用**: 专项验证 (§5.3 同空间) 表明 bf16 轨道逐配置 distortion 相对偏差最大 **4.9×10⁻⁵**, 判定门限 1×10⁻³, 留有 20 倍余量; argmin 与排序均与 fp32 轨道完全一致。端到端耗时上稳态每场只缩短 ~15-20% (124 → 99 min) 而非体积比例的 2×: 瓶颈已在计算 (IO 供给速率远超计算消耗速率), bf16 的收益主要在存储减半, 不在端到端耗时。

### 5.3 实测设定与结果

设定 (全量验证):

- 模型/数据: llama-1B (约 1.2×10⁹ 参数), OpenWebText, 真实训练;
- trace: 分支点 40 后 1000 步窗口; fp32 约 4.7 TB, bf16 副本 2.5 TB (转换 8207 s);
- 配置空间: 4 种分数 (`update`/`abs_g`/`abs_m`/`coi`) × 3 种比例 (20%/30%/40%) × 6 种周期 (5/10/20/50/100/200) = **72 配置**; 每场运行固定一个周期、评 4×3 = 12 配置 (一组), 共 6 场;
- 引擎: 8×A100-40GB, `--fused-shards 8`, `--fused-max-configs 12`, dense 缓存开启, 重放算术 fp64, NFS 共享存储。

结果:

| 指标 | fp32 trace | bf16 副本 | 门 |
|---|---|---|---|
| 逐配置 distortion (视界 1000) 相对差 (对逐配置引擎) | 72/72 = **0** | max **4.9×10⁻⁵** | 1×10⁻⁶ / 1×10⁻³ |
| 选点阈值 | 72/72 逐位相等 | — | 逐位 |
| argmin (在线 vs 离线) | 严格 1/1 | 与 fp32 一致 | 严格 |
| 按 distortion 排序 | 严格 18/18 | 严格 18/18 | 严格 |
| 最小决策差距 / 等价噪声 | **393×** | — | — |

计时与瓶颈口径: fp32 轨稳态每场均值 ~117 min (94-124); 冷趟 (含 dense 填充) 229 min (一次性, 此后走缓存); 72 配置全程 ≈ 12.6 h。bf16 轨稳态 ~99 min (94-109)。**"旧引擎 IO 主导、fused 计算主导"并不矛盾, 是同一台机器上瓶颈随读量变化的翻转**: 逐配置引擎每配置重读窗口, IO 是短板; fused 把读量摊薄到每组一遍后, IO 供给能力 (~2.3 GB/s 页缓存快照) 已远超计算消耗速率 —— 折叠/掩码/AdamW 阶段的有效吞吐只有 0.63-0.92 GB/s, 短板翻转为 8 张卡的算力。进一步加速应投向计算 (更多分片、融合 kernel), 不是 IO; 从 12.6 h 到 ~1 h 需要 ~12×, 对应的改造量级见 §7。

## 6. 使用手册

### 6.1 采集 trace

trace 用 **shadow 模式**采集: 训练行为与 dense 完全一致 (不影响训练), 引擎在旁路录制。训练配置加三个键 (外加一个可选的存储精度键; Hydra 透传到 `sga/config/arguments.py` 的 `--sga-trace-*`):

```bash
python run.py --config-path examples/llama3 --config-name train_1b_shadow \
    ... \
    ++sga.sga_trace_dir=/data/traces/run1 \
    ++sga.sga_trace_branch_points='40' \
    ++sga.sga_trace_window=1000
```

| 配置键 | 含义 |
|---|---|
| `sga_trace_dir` | trace 输出目录; **设置即开启采集** |
| `sga_trace_branch_points` | 分支点列表 (训练全局步号), 逗号分隔, 如 `'40'` 或 `'40,1040'` |
| `sga_trace_window` | 每个分支点之后录多少步梯度 |
| `sga_trace_dump_dtype` | g 梯度文件的存储精度: `fp32` (默认) 或 `bf16` — `bf16` 在录制时直接落 bf16 dump (体积减半, 见 §6.4), state 文件两种取值下都保持 fp32 |

**分支点的确切含义**: 分支点是训练全局步号, 即第几个 train iteration。`'40'` 表示训练推进到第 40 个 train iteration 时落一份完整状态快照 (参数/动量/方差, 附当步梯度), 此后每步的裁剪后梯度各落一个文件; 重放 = "从第 40 步快照出发, 依次消费第 41..1040 步录制梯度" (配 `--window 1000`)。重放不需要模型、数据集与前向代码。磁盘注意: 1.2B 模型每步约 4.7 GB。

### 6.2 运行重放

冒烟 (单进程 CPU, 小窗口):

```bash
python -m sga.replay \
    --trace-dir /data/traces/run1 \
    --branch-point 40 --window 100 \    # 从第 40 个 train iteration 的快照出发, 用其后 100 步梯度
    --ratios 20,30,40 --horizons 10,100 \
    --score-names abs_m,update \
    --replay-mode cycle --cycle-len 20 \
    --replay-dtype fp64 --device cpu \
    --dense-cache /data/cache/run1 --out report.json
```

正式扫描 (8 卡)。**不用 `--device cuda:N`**, 用 `CUDA_VISIBLE_DEVICES` 圈卡, `--fused-shards` 定分片数 (分片 r 固定用第 r 块可见卡):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sga.replay \
    --trace-dir /data/traces/run1 --branch-point 40 --window 1000 \
    --ratios 20,30,40 --horizons 100,1000 \
    --score-names abs_g,abs_m,update,coi \
    --replay-mode cycle --cycle-len 20 \
    --device cuda --fused-shards 8 \
    --dense-cache /data/cache/run1 --out report.json
```

一场运行评 `|分数| × |比例|` 个配置 (上例 4×3 = 12, 上限 12), 周期由 `--cycle-len` 固定; 扫多个周期就多跑几场 (换 `--cycle-len`, dense 缓存自动复用)。`--branch-point all` 遍历 trace 内全部分支点。首次运行自动填充 dense 缓存 (报告 `dense_cache_hit: false`), 此后所有同 trace/窗口/精度的运行直接命中 (`true`)。

跨多个周期同场混装用 `--cycle-lens 5,100,200` (配置空间 = 分数 × 比例 × 周期全集, 组内任意 T 混装, 混跑与分跑逐位一致); 组大小可交给 `--fused-state-budget-mb` 按常驻状态字节预算自动推导 (§6.7)。

跨节点分片 (2 台 8 卡 → world16) 不换命令面, 加 `--nodes` 即走多机拉起器 (§6.8):

```bash
python -m sga.replay \
    --trace-dir /data/traces/run1 --branch-point 40 --window 1000 \
    --ratios 20,30,40 --score-names abs_g,abs_m,update,coi \
    --replay-mode cycle --cycle-lens 5,20,100 \
    --device cuda --fused-shards 16 \
    --nodes <node0-ip>,<node1-ip> \
    --dense-cache /data/cache/run1 --out report.json
```

### 6.3 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--trace-dir` | 必填 | trace 目录 (§6.1 产物) |
| `--branch-point` | 必填 | 分支点训练全局步号 (train iteration 序号) 或 `all` |
| `--window` | 100 | 重放窗口步数 |
| `--replay-mode` | zerofill | `cycle` = 周期掩码语义 (本文, 走 fused 引擎); `zerofill` = 冻结掩码、未选坐标置零的对照语义 (进程内路径, 无需缓存) |
| `--cycle-len` | 20 | 周期长度: 每隔多少步折叠残差、全量同步并重选 (单值兼容入口; 配置空间见 `--cycle-lens`) |
| `--cycle-lens` | — | 周期长度集合 (逗号分隔, 自动排序去重), 如 `5,100,200`; 配置空间 = 分数 × 比例 × 周期全集, 组内混装任意 T (每 config 独立周期日历, 混跑与分跑逐位一致); 与 `--cycle-len` 二选一 |
| `--ratios` | 10,20 | 通信比例 (%) 列表 |
| `--score-names` | 全部注册分数 | 逗号分隔; 注册表: `update`/`abs_g`/`abs_m`/`coi` (与在线引擎同一注册表) |
| `--horizons` | 1,10,100 | 输出 distortion 的视界步列表 (自动截到窗口长; 周期语义下第 1 步是全量同步步, distortion 恒为 0, 有信息量的视界从 2 起) |
| `--dense-cache` | — | dense 缓存目录; **cycle 模式必填** (首跑自动填充) |
| `--fused` | 关 | **已废弃 no-op** (旧行为现已无条件生效); 报告 `config.fused` 仅回显此 flag, 生效引擎看 `config.fused_effective` |
| `--fused-shards` | 8 | 分片进程数 = 用到的 GPU 数 (分片 r → 第 r 块可见卡) |
| `--fused-max-configs` | 12 | 单趟同时推进的 (分数×比例×周期) 数; 组内每配置优化器状态驻留显存, 显存紧张可调小 (或改用 `--fused-state-budget-mb` 按字节预算切组) |
| `--fused-state-budget-mb` | 0 (关) | 常驻状态字节预算 (MiB): 组大小改由 `floor(预算 / 每配置状态字节)` 推导, >0 时覆盖 `--fused-max-configs`; 记账口径 = {w,m,v,res} × 重放精度字节 × 最宽分片 (⌈N/world⌉), bool 掩码 / chunk 工作集 / g 驻留缓冲在预算外; llama-1B 8 分片 fp32 每配置 ≈2359 MiB → 30720 推导 R=13 (§6.7) |
| `--fused-g-residency` | 关 | 每步梯度切片一次上传显存驻留, 全配置 × 3 pass 共享 (数值逐位不变; §6.7) |
| `--fused-dev-reduce` | 关 | 范数/视界分子改设备端 fp64 累加, 每 pass 一次主机同步 (数值逐位不变; §6.7) |
| `--nodes` | — | 逗号分隔节点地址表: 走多机拉起器 (§6.8); 本进程须在**首节点**上且与各节点共享同一 NFS 路径视图; 分片 r → 节点 `r // (shards/节点数)` 的 `cuda:(r % 每节点分片数)` |
| `--ssh-key` | env/祖先目录 | `--nodes` 拉起用的集群私钥 (默认 `$SGA_SSH_KEY`, 再退到 workspace 祖先的 `cluster_ssh_key`) |
| `--preflight-timeout` | 600 | 多机预检 rendezvous 有界时限 (秒), 到期大声失败并给各节点诊断 |
| `--run-deadline` | 0 (关) | 运行段可选墙钟上限 (秒): 到期杀掉全部 node-worker 并带各节点日志尾大声失败 (0 = 不设界) |
| `--launch-log-dir` | — | `--nodes` 运行的各节点日志目录 (缺省: 运行临时文件旁的时间戳目录) |
| `--device` | cpu | `cpu` 或 `cuda`; `cuda:N` 不支持 (分片固定映射, 报错有提示) |
| `--replay-dtype` / `--norm-dtype` | fp64 / fp64 | 重放算术精度 / 每步范数累加精度 (`fp32` 偏差有冻结带覆盖, 一般不动) |
| `--trace-dtype` | fp32 | trace 存储精度; bf16 副本须填 `bf16` (§6.4) |
| `--chunk-mb` | 256 | 大参数分块工作集预算 (MB) |
| `--seed` | 无 | 可复现种子 |
| `--out` | 无 | JSON 报告输出路径 |

### 6.4 bf16 筛选副本

一次性转换 (读 4.7 TB 写 2.5 TB), 之后所有扫描省一半存储与 IO:

```bash
python -m sga.trace_convert --src /data/traces/run1 --dst /data/traces/run1_bf16 \
    --from-step 40 --to-step 1040
python -m sga.replay --trace-dir /data/traces/run1_bf16 --trace-dtype bf16 \
    ...   # 其余参数不变; 载入即升 fp32, 精度见 §5.2
```

dense 缓存身份键含 trace 精度, fp32/bf16 缓存互不串用。

`trace_convert` 之外的第二条路: 新录制的 trace 可在训练配置里直接加 `++sga.sga_trace_dump_dtype=bf16`, 录制时就落 bf16 dump (免 4.7 TB fp32 中间副本; catalog 自带 `g_dtype=bf16` 标记, 重放同样加 `--trace-dtype bf16`)。`trace_convert` 仍适用于已录制好的 fp32 trace。

### 6.5 输出报告

`report.json` (数值四舍五入到 8 位小数):

```
config                       # 运行参数回显 (含 fused_effective / dense_cache_hit)
branches["40"]
  ├─ per_ratio["20"]                  # 通信比例 20%
  │    ├─ k                           # 实际选中坐标数
  │    └─ scores["abs_m"]
  │         ├─ threshold              # top-k 阈值 (可跨运行逐位比对)
  │         ├─ distortion["100"]      # 视界 100 步的 distortion (§2.3), 越小越好
  │         ├─ ndcg_at_k              # 分数排序对"真实重要坐标"的排序质量 (1 = 完美)
  │         ├─ recall_at_k            # 选中坐标对真实重要坐标的召回率
  │         ├─ missed_label_mass      # 漏掉的真实重要性质量占比 (越小越好)
  │         └─ missed_score_mass      # 漏掉的分数质量占比
  └─ replay                           # dense 对照范数轨迹等重放元数据
```

"真实重要坐标" (label) = dense 对照重放中逐坐标累计更新量的平方 —— 即全量同步时每个坐标实际有多重要。distortion 回答"权重偏了多少", NDCG/召回回答"分数选得准不准", 两者互补。

### 6.6 常见报错

| 报错 | 原因与解法 |
|---|---|
| `--replay-mode cycle requires --dense-cache ...` | cycle 模式必填 `--dense-cache` (首跑自动填充) |
| `--fused cannot be combined with ...` / deprecated 提示 | `--fused` 已废弃, 去掉即可 (行为已无条件生效) |
| `cycle cannot be combined with --self-check` | 两组件不兼容, 去掉 `--self-check` |
| `--device cuda:N is not supported ...` | 分片 r 固定映射第 r 块可见卡; 改 `--device cuda` + `CUDA_VISIBLE_DEVICES=4,5` 圈卡 |
| `--fused-shards N exceeds the visible CUDA devices` | 分片数多于圈定卡数, 调小或放开圈卡 |
| `bf16 payload ... but --trace-dtype is fp32` | bf16 副本忘加 `--trace-dtype bf16` |
| `--score-names [...] not a registered score` | 分数名拼错; 可用名见注册表 (`update`/`abs_g`/`abs_m`/`coi`) |
| `shard_select requires finite scores` | 分数出现 NaN/Inf, 查 trace 完整性与分数配置 |
| `k > N` | 比例换算的 k 超过总坐标数, 查 `--ratios` |

### 6.7 加速旗标与计时

两个 opt-in 加速旗标, 默认关; 开启后报告值与关闭时**逐位恒等** (上传/升精度/除法都是逐元素可交换运算, 归约加数与合并序不变), 只是执行策略变快:

- `--fused-g-residency` — 每步把本片各参数的梯度切片**一次性**上传显存驻留, 组内全部配置 × 3 个流式 pass 共享同一份。旧路径每配置每 pass 重复上传同一批字节 (profile 实测冗余 24×, 每 rank 每 pass ~6.9 GB H2D); 驻留后每步每参数只剩一次上传。CUDA 下预取线程改用锁页缓冲 + `non_blocking` (host 不阻塞; H2D 本身仍在流上串行); 边界步无梯度载荷时读缓存的设备端零张量。内存口径: 驻留量 = 本片每参数切片全量, 与 `--chunk-mb` 工作集预算**无关**, 每步的 clip 系数除法瞬时再翻倍, 锁页主机内存约 (depth+1) 步切片。
- `--fused-dev-reduce` — pass1 clip 范数分子与 pass3 视界分子改为逐配置 fp64 **设备标量**累加, 每 pass 一次主机同步。旧路径每 chunk 一次 `float()` 同步 (profile 实测每 rank 每千步 6 万-44 万次)。
- `--fused-state-budget-mb` — 按常驻状态字节预算自动推导每趟配置数: 组切分从计数口径 (`--fused-max-configs`) 升级为字节预算口径, 预算 >0 时组上限 = `floor(预算 / 每配置状态字节)` 并覆盖计数旋钮。记账只含 {w,m,v,res} × 重放精度字节 × 最宽分片 (⌈N/world⌉ — 全 rank 无集体地一致, 分区纯函数); bool 选点掩码 (1 B/elem)、`--chunk-mb` 工作集与 g 驻留缓冲**不计入**预算。组切分只改 "哪些配置同趟", 每配置数值与所在分区无关 (逐臂状态机 + 逐 lane fp64 归约 + 逐配置独立选点), 故报告与计数旋钮**逐位恒等** (同分区与跨分区皆然); 报告回显 (`fused_state_budget_mb` / `state_budget_mb` / `group_cap`) 仅在开启时出现 — 默认路径的报告字节不变。`REPLAY_S0_PROFILE=1` 的诊断行同步增列 `state_gb` (本组常驻状态全额口径: 四状态张量 + bool 掩码)。

环境变量 `REPLAY_S0_PROFILE=1` 输出每配置组诊断计数 (主机同步数 / gloo 次数 / 每 pass H2D 字节与耗时 / 显存峰), 并在 rank0 报告 fragment 附 `s0_profile` 键。计数是诊断口径而非精确测量: 驻留模式下共享上传全部记入 pass1, `mem_peak_gb` 是进程生命周期累计。同一开关下 rank0 另增 `S0FRONT` 行, 把流前固定开销分解为分支分数 / dense 缓存载入 / 选点 / 排序指标四段耗时, 并回显本臂排序落在设备还是主机 (`front_sort=dev/cpu`)。

流前排序指标 (NDCG/recall 的 argsort 与 idcg 排序) 在 `--device cuda` 下自动在设备上计算: 设备只产出置换与排序值, 全部归约 (fp32 求和、log2、fp64 加权) 留在主机原表达式, 且稳定排序对有限非负键与后端无关, 报告与纯主机路径逐字节相等。显存守卫按 32 B/elem 对 `torch.cuda.mem_get_info` 记账 (先释放本进程持留的空闲缓存块再量), 放不下即回退主机排序并打印一行诊断; 若设备排序自身抛出 CUDA OOM (守卫测量与显存分配之间的竞态窗口), 同样回退。环境变量 `REPLAY_FUSED_FRONT_SORT=cpu` 是运维紧急开关, 强制逐位等价的主机路径; 其他取值报错退出。实测 (20 步窗口 × 6 配置, dense 缓存命中): 流前合计 995.6 → 193.6 s (5.14×), 其中排序段 885.2 → 85.1 s (10.4×), 全臂墙钟 ~1055 → 251 s; 72 配置 demo 形状的流前推算约 280-400 s (推算, 未经实测)。

流前选点与载荷收集 (§4.4 流前准备的主体) 在 `--device cuda` 下同样自动设备化。机制: 各分数与 label 向量在主机原件 (供 fp32 求和等主机归约 — 该求和的设备归约序与主机不逐位一致) 之外各持一份设备副本, `shard_select`、布尔掩码索引与收集载荷都在设备副本上执行, 收集到的各分片载荷一律落回主机 (纯搬运) 再进排序指标的主机归约 — 每个报告比特因此与原主机路径同源, 逐字节相等。变长载荷收集在单机且 NCCL 可用时走单节点 NCCL 快道 (纯数据搬运: 长度仍走 gloo 交换, 按 rank 序定长拼接, bool 载荷以 uint8 上线); 多机或无 NCCL 时回退 gloo 路径。设备与传输两个决定都经全 rank 一致投票 (gloo all_reduce 取 MIN): 任一 rank 否决即全群走主机 — 部分回退会把集合通信撕成两半, 绝不允许。显存守卫按常驻向量 (4 × (分数数 + 1) B/elem) + 单次选点峰 (14 B/elem, 含全并列最坏情形的 nonzero int64 索引) + 收集扇出记账 (期望值加裕量, 非穷尽记账), 量不下即大声回退主机路径; 守卫只保证拒绝点的回退 — 投票完成后显存被第三方抢占 (测量与分配之间的竞态) 会按设计中止运行, 因为集合通信无法在块中途回卷。环境变量 `REPLAY_FUSED_SELECT=cpu` 是运维紧急开关, 语义同 `REPLAY_FUSED_FRONT_SORT`; 其他取值报错退出。`S0FRONT` 诊断行同步增列本臂回显 `front_sel=dev/cpu` 与 `gather=nccl/gloo`。流前循环结束后设备驻留 (设备副本与选点掩码) 即释放并清空缓存, 不抬高流式阶段的显存峰。实测 (72 配置 demo 形状, 20 步窗口, dense 缓存命中, 8×A100-40GB, A/B 同机安静门): 选点桶 (选点 + 载荷收集 + 掩码索引) 330.2 → 96.6 s (3.42×), 流前合计 591.4 → 346.6 s (1.71×), 全臂墙钟 1015 → 744 s (~1.36×); 两臂报告逐字节相等, 设备臂显存峰 36.8 GB (40 GB 卡)。生产尺度确认 (llama-1B 真实训练 trace, 1000 步窗口 × 12 配置, 8×A100-40GB, bf16 trace + fp32 重放, 同门): 选点桶 170.4 → 44.9 s (3.80×), 流前合计 295.9 → 157.9 s (1.87×), 两臂报告逐字节相等, 设备臂零回退。

实测 (llama-1B 真实训练 trace, 1000 步窗口 × 12 配置, 8×A100-40GB, bf16 trace + fp32 重放):

| 配置 | 稳态耗时 | 相对基线 |
|---|---|---|
| 旗标关 (三趟均值, CV 1.3%) | 6452 s | 1× |
| 仅 `--fused-g-residency` | 5220 s | 1.24× |
| 双旗标 + 预取切片先行 | 2906 s | 2.22× |

仅驻留时 pass 计时合计只占墙钟 ~13% (profile 归因), 其余耗在预取线程的全载荷升精度; 预取改为"先切本片窗口再升精度"后该份额消除, 上表末行即终态组合 (小窗 w100 口径同向: 双旗标 1405 s vs 基线 2254 s)。`--chunk-mb` 在驻留下不再影响耗时 (旧路径 c512 的 -12.5% 收益正来自已消除的 per-config 上传冗余)。全部组合逐位恒等 (A/B 对照相干 trace 产物 rel==0)。

### 6.8 多机分片拉起 (--nodes)

按坐标分片的数据面天然可跨机: 每个分片只 mmap 读自己坐标区间的文件页, 跨分片交换只有 fp64 标量与 32769 桶整数直方图 (gloo over tcp 足够)。`--nodes` 把这套能力接进同一命令面 — `sga.replay` 的 main 检测到 `--nodes` 即转入 `sga/replay_launch.py` 多机拉起器:

- **拓扑**: 分片 r → 节点 `r // (world / M)` 的 `cuda:(r % (world / M))`; 拉起进程在节点 0 上, 所有节点必须共享同一 NFS 树的同一路径视图 (容器内即 `/workspace` 视角)。
- **启动序**: 节点 0 父进程先做 dense 缓存 (miss 时本进程填充 — dense 段绝不在进程组内跑), 缓存发布后**带外 ssh 确认** (不走 NFS 轮询, 避负 dentry 盲窗); 然后全节点**有界预检 rendezvous** (`--preflight-timeout`, 起不来的节点几分钟内带诊断大声失败, 而不是挂在 2 h gloo 超时上); 再逐节点 ssh :2222 拉起 node-worker (每 worker 自带 `SGA_MASTER_ADDR/SGA_MASTER_PORT/SGA_GLOBAL_RANK`, 本地分片号留在 `rank`); 运行段全程监管, 任一 node-worker 早退即杀群收尸。
- **数值面**: 全局 rank 键控 (选点 tie-fill 按进程组序), 单机数值不受影响; 多机产物落在冻结纯重排带内 (与单机对照: 非 sum 域逐位, sum/mean rel ≤1e-6)。
- **免 ssh 仿真**: `SGA_LAUNCH_EXECUTOR=local` 把节点拉起降级为本机子进程, 单机即可演练完整拓扑 (单测与 CI 用)。
- **冒烟**: `tests/smoke_replay_multinode.sh <master-ip> <worker-ip> <NFS-workdir>` — CPU world=2 双节点全链路 (冷缓存填充 + 带外发布 + ssh 拉起), 判据 = 纯重排带。

计时带 (world16 vs world8 加速比) 为预注册待补项。

## 7. 边界与限制

- **默认单机, 多机 opt-in**: 控制面默认走本机回环 gloo (单机 8 卡零配置); 跨节点须显式 `--nodes` (§6.8), 页缓存去重收益只在同宿主分片间存在 (跨机各读各的页, 正确性不受影响)。
- **cycle 语义专用**: fused 引擎只实现 `--replay-mode cycle`; `zerofill` 仍走 `sga/replay.py` 进程内路径。
- **计算主导**: 实测瓶颈在分片算力而非 IO (§5.3); 单机上提速手段是融合 kernel 或压缩每步的冗余搬运 (同一步的梯度切片目前被 host→device 拷贝最多 3 次: pass 1/pass 2/pass 3 各一次), bf16 副本主要省存储。
- **到 ~1 h 的路径 (12.6 h → 1 h 需 ~12×)**: (a) 多机坐标分片**已落地** (`--nodes`, §6.8; 计算主导下分片数近线性加速, 前提是 NFS 前端聚合带宽跟上, 计时带待补); (b) 融合 kernel: 把范数/掩码/残差/AdamW 多趟扫描合成一遍 Triton kernel (在线优化器侧已有同族 kernel 可借鉴), 估计 2-3×; (c) 组大小上限受显存: 每配置每分片约 2.4 GB 状态 (w/m/v/res), 40 GB 卡上 12 配置已近上限, 加大组需同时加分片 (组大小可交由 `--fused-state-budget-mb` 按状态字节预算自动推导, §6.7)。
- **等价性范围**: 单分片 = 逐位一致; 多分片偏差属纯浮点重排类, 由预注册带覆盖。分数公式与在线引擎同源 (同一注册表), "离线结论外推到在线"取决于 trace 的代表性 (分支点/窗口是否覆盖感兴趣的训练阶段), 不取决于重放引擎本身。
