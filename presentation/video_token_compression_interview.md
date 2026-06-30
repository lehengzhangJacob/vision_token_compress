---
marp: true
theme: default
paginate: true
size: 16:9
---

# Video LLM 视觉 Token 压缩
## 三篇论文算法精读

**汇报人**：张同学  
**顺序**：PruneVid → VidCom² → AOT  
**共同设定**：Training-free、plug-and-play、保留率 R=10%–25%

---

# 背景：为什么需要 Token 压缩？

```
T 帧视频 → ViT → T×N 个 visual tokens → LLM (O(n²) attention)
```

- 32 帧 × ~200 token/帧 ≈ **6000+ tokens**，prefill 是主要瓶颈
- 视频帧间 **高度冗余**（背景静态、相邻帧相似）
- 目标：在 **不微调** 的前提下，把 token 数压到 R%，尽量不掉点

**三类压缩位置**：
1. Pre-LLM（ViT 输出后）
2. LLM 内（prefill / decode 阶段）
3. Hybrid（PruneVid 采用 1+2）

---

# 三篇论文定位

| 论文 | 会议 | 核心思路 | 压缩位置 |
|:---|:---:|:---|:---|
| **PruneVid** | ICLR'25 | 时空合并 + 问题引导剪枝 | Pre-LLM + LLM内 |
| **VidCom²** | EMNLP'25 | 帧级自适应 + token 独特性 TopK | Pre-LLM |
| **AOT** | CVPR'26 | 锚点 + 最优传输信息聚合 | Pre-LLM |

**演进**：均匀压缩 → 帧/token 自适应 → 聚合替代丢弃

---

# ① PruneVid — 问题与动机

**论文**：Visual Token Pruning for Efficient Video LLMs (ICLR 2025)

**观察**：
- 视频有大量 **静态背景**（相机不动、物体不动）
- 相邻帧 token **高度相似**（时序冗余）
- 但并非所有视觉 token 都与 **问题** 相关

**设计原则**：
1. Training-free
2. 先消减视频固有冗余
3. 再保留与问题相关的 token

---

# PruneVid — 整体流程

```
输入视频 (T帧)
    ↓
[Stage A] 时空 Token 合并 (Pre-LLM)
    ↓ 压缩后的 visual tokens
[Stage B] LLM 第 M 层：Q→V attention 剪枝
    ↓ 保留 Top-α% tokens
[Stage B] KV cache 压缩 → 后续层 + decode 加速
```

**关键超参**：τ=0.8（静态阈值）、γ=0.25（时序段数）、β=0.5（聚类比例）、α=0.4（保留比例）、M=10（剪枝层）

---

# PruneVid Stage A：时序分段

**Step 1 — 帧级聚类（DPC-KNN）**

- 每帧平均池化得特征 f^(t)
- DPC-KNN 将 T 帧分为 B 个时序段 {T₁…T_B}（B ∝ γ·T）
- 同段内帧内容相似（同一场景/镜头）

**直觉**：先把视频切成「语义连续」的片段，再在段内做冗余消除。

---

# PruneVid Stage A：静态 token 时序合并

对每个时序段 T_b、每个空间位置 i：

1. 取段内所有帧该位置的 token 序列
2. 计算两两 cosine 相似度，得平均相似度 s̄_i
3. 若 s̄_i ≥ τ → **静态 token**（背景）
4. 静态 token：**时序平均** → 1 个 token 代表整段

$$\tilde{X}_v^{(b)}(i) = \frac{1}{|T_b|}\sum_{t \in T_b} X_v^{(t)}(i)$$

**动态 token**（运动区域）保留每帧独立表示。

---

# PruneVid Stage A：空间聚类合并

对每帧的静态/动态 token **分别** DPC-KNN 聚类：

- 簇数 ∝ β × |静态或动态 token 数|
- 簇内 token **取平均** → 1 个代表 token

**效果**：Stage A 结束后 token 数大幅下降，且主要在 Pre-LLM 完成，不破坏 FlashAttention 兼容路径（若只走 Stage A）。

---

# PruneVid Stage B：问题引导剪枝

**在 LLM 第 M 层**（非最后一层）：

1. 计算 cross-attention：问题 token Q → 视觉 token V
2. 对每个 visual token，取所有问题 token 上的 **max attention**
3. 保留 Top-α%（默认 40%）高 attention 的 visual tokens

**为什么用 max pooling？** 并非每个问题词都重要，取 max 捕获「至少被一个关键词关注到」的 token。

**KV cache 压缩**：前 M 层的 KV 只保留选中 token → decode 也加速。

---

# PruneVid — 算法小结

| 模块 | 输入 | 操作 | 输出 |
|:---|:---|:---|:---|
| 时序分段 | T 帧特征 | DPC-KNN | B 个片段 |
| 静态合并 | 段内 token | 时序平均 | 少 token/段 |
| 空间聚类 | 静/动 token | DPC-KNN + 均值 | 更少 token |
| 问题剪枝 | Q+V @ layer M | Top-α% attention | 最终 token 集 |

**特点**：唯一使用 **文本信息** 的方法；Hybrid 路线。

---

# PruneVid — 论文结果（参考）

| 模型 | MVBench | VideoMME | 压缩率 |
|:---|:---:|:---:|:---:|
| PLLaVA-7B baseline | 47.6 | 45.3 | 100% |
| + PruneVid | ~持平 | ~持平 | **~16–20%** tokens |

FLOPs ↓74–80%，TTFT ↑1.55×

---

# ② VidCom² — 问题与动机

**论文**：Video Compression Commander (EMNLP 2025)

**两篇前作留下的问题**：
1. **均匀压缩各帧** → 关键帧（转场、动作突变）信息损失
2. LLM 内剪枝 → 不兼容 FlashAttention 等高效算子

**核心思想**：每帧「独特性」不同 → **分配不同压缩强度**

**全程 Pre-LLM**，输出固定平均保留率 R。

---

# VidCom² — 两阶段框架

```
Stage 1: Frame Compression Adjustment
  → 计算每帧独特性 u_t → 分配每帧保留率 r_t

Stage 2: Adaptive Token Compression
  → 计算每个 token 综合独特性 → 每帧 TopK(r_t × M)
```

**与 PruneVid 对比**：不做 LLM 内操作；不做静态/动态分离；用 **独特性分数** 替代 attention。

---

# VidCom² Stage 1：帧独特性

**全局视频表示**：
$$g_v = \frac{1}{T \cdot M}\sum_{t,m} x_{t,m}^v$$

**Token 视频级独特性**（与全局越不像 → 越独特）：
$$u^{video}_{t,m} = -\cos(x_{t,m}^v,\; g_v)$$

**帧分数** = 帧内 token 独特性均值：
$$u_t = \frac{1}{M}\sum_m u^{video}_{t,m}$$

**直觉**：若一帧充满「和整段视频都不一样」的 token，该帧更关键。

---

# VidCom² Stage 1：保留率分配

Softmax 归一化帧权重：
$$\tilde{u}_t = \frac{u_t - \max(u)}{\tau}, \quad \sigma_t = \text{softmax}(\tilde{u}_t)$$

每帧保留率（平均仍为 R）：
$$r_t = R \times \left(1 + \sigma_t - \frac{1}{T}\right)$$

- 独特帧：r_t > R（多留 token）
- 冗余帧：r_t < R（多删 token）
- 全局约束：Σr_t / T ≈ R

---

# VidCom² Stage 2：Token 独特性

**帧内独特性**（与帧均值越不像 → 越独特）：
$$u^{frame}_{t,m} = -\cos(x_{t,m}^v,\; g_{f,t})$$

**综合分数**：
$$u_{t,m} = u^{frame}_{t,m} + u^{video}_{t,m}$$

**选择**：每帧保留 TopK(u, r_t × M)

**直觉**：同时「帧内少见」且「全片少见」的 token 优先保留。

---

# VidCom² — 算法小结

| 步骤 | 公式/操作 | 作用 |
|:---|:---|:---|
| 全局均值 g_v | mean pooling | 视频摘要 |
| 帧分数 u_t | mean(-cos to g_v) | 帧级重要性 |
| 分配 r_t | R × softmax 权重 | 差异化预算 |
| token 分数 | u_frame + u_video | 双层独特性 |
| 选择 | per-frame TopK | 最终压缩 |

**优势**：R=25% 时达 upper bound **99.6%**；可叠加到其他压缩器上。

---

# VidCom² — 论文结果（LLaVA-OV, R=25%）

| Method | MVBench | VideoMME | Avg% |
|:---|:---:|:---:|:---:|
| DyCoke | 49.5 | 51.0 | 87.0 |
| SparseVLM | 56.4 | 57.3 | 97.5 |
| **VidCom²** | **57.2** | **58.6** | **99.6** |

LLM 生成延迟 ↓70.8%

---

# ③ AOT — 问题与动机

**论文**：Token Reduction via Local and Global Contexts Optimization (CVPR 2026)

**AOT** = **A**nchors + **O**ptimal **T**ransport

**前人做法的问题**：
- TopK / 聚类 / 直接丢弃 → **被删 token 的信息彻底丢失**
- 低分 token 未必无用（可能含细粒度语义）

**AOT 思路**：
- 选 **锚点 token** 留下
- 用 **最优传输（OT）** 把删掉 token 的信息 **聚合** 进锚点

---

# AOT — 整体流程

```
每帧 visual tokens
    ↓
[Phase 0] 选 Global + Local 锚点 (M个)
    ↓
[Phase I]  帧内 OT：unanchors → 聚合到 anchors
    ↓
[Phase II] 帧间 OT：clip 内跨帧聚合 + 保留动态 token
    ↓
压缩后的 token 序列 → LLM
```

**两阶段 OT**：先空间（帧内），后时间（帧间）。

---

# AOT Phase 0：锚点选取

**Global Anchors**（语义重要）：
- ViT 最后层，按 [CLS]→patch 的 attention Top-K

**Local Anchors**（空间细节）：
- 特征图划 W 个 grid window
- 每窗按 [CLS] attention Top-K_w

$$\mathbf{X}^{anchors} = \mathbf{x}^{global} \cup \mathbf{x}^{local}$$

其余 token = **unanchors**，等待 OT 聚合。全局/局部各约一半预算。

---

# AOT Phase I：帧内 OT

**代价矩阵**：C_ij = 1 - cos(anchor_i, unanchor_j)

**最优传输**（Sinkhorn-Knopp 快速求解）：
$$\min_{T} \langle T, C \rangle \quad \text{s.t. } T\mathbf{1}=\mathbf{u},\; T^\top\mathbf{1}=\mathbf{v}$$

**锚点更新**（聚合 unanchor 信息）：
$$\tilde{x}^a_j = \frac{x^a_j + \lambda \sum_i T^*_{ij} x^u_i}{1 + \lambda m_j}$$

**直觉**：不是扔掉低分 token，而是按相似度「分配」其语义给最匹配的锚点。

---

# AOT Phase II：帧间 OT

1. 视频分为若干 **clip**（均匀或自适应聚类）
2. 每 clip 首帧 intra 压缩结果 = **temporal anchor**
3. 对后续帧：anchor 与当前帧 token 做 OT
4. 若 token 与 anchor 匹配度 q_i < τ → **保留**（时序变化大）
5. 否则 → **聚合进 anchor**（时序冗余）

**效果**：相似帧内容合并，运动/转场 token 单独保留 → 保留动态。

---

# AOT — 算法小结

| 阶段 | 机制 | 解决什么 |
|:---|:---|:---|
| 锚点选取 | CLS attention + grid | 空间多样性 |
| 帧内 OT | Sinkhorn 聚合 | 不丢被删语义 |
| 帧间 OT | clip + 动态保留 | 时序冗余 + 动态 |

**与 PruneVid**：不用 text；不做 LLM 内剪枝  
**与 VidCom²**：不用独特性分数；用 OT 聚合替代 TopK 丢弃

---

# AOT — 论文结果（LLaVA-OV, R=25%）

| Method | MVBench | EgoSchema | VideoMME | Avg% |
|:---|:---:|:---:|:---:|:---:|
| PruneVid | 57.4 | 59.9 | 57.4 | 98.6 |
| VidCom² | 57.2 | — | 58.6 | 99.6 |
| **AOT** | **58.7** | **61.3** | 57.5 | **100.0** |

10% 保留率仍保持 **97.6%** 性能；FLOPs 最低 **8.3%**

---

# 三篇算法横向对比

| 维度 | PruneVid | VidCom² | AOT |
|:---|:---|:---|:---|
| 压缩位置 | Pre-LLM + LLM内 | Pre-LLM | Pre-LLM |
| 时序 | 静态 token 时序均值 | 帧级 r_t 分配 | OT 跨帧聚合 |
| 空间 | DPC-KNN 聚类 | TopK 独特性 | Global+Local 锚点 |
| 文本感知 | ✓ Q→V attention | ✗ | ✗ |
| 信息处理 | 合并 + 丢弃 | TopK 丢弃 | **OT 聚合** |
| 核心数学 | 聚类 + attention | Softmax 权重 | Sinkhorn OT |

---

# 方法演进（算法视角）

```
PruneVid                VidCom²                  AOT
────────                ───────                  ───
识别静态区域      →     量化帧独特性      →     选锚点
时序/空间合并           差异化 TopK              OT 聚合
+ 问题 attention                                   保留动态
```

**趋势**：
1. 压缩粒度：均匀 → 帧级 → 帧内+帧间联合
2. 信息处理：丢弃 → 聚合
3. 位置：Hybrid → 纯 Pre-LLM

---

# 复现工作概览

三篇论文均在 **lmms-eval** 框架下复现，training-free、plug-and-play。

| 论文 | 环境 | 状态 |
|:---|:---|:---|
| **PruneVid** | conda `PruneVid` | VideoMME ✓；其余 benchmark 进行中 |
| **VidCom²** | conda `VidCom2` | 环境+smoke test ✓；**卡在 MLVU 预下载** |
| **AOT** | conda `AOT` | LLaVA-OV 大部分 ✓；**VideoMME 20% 跑批中** |

**共同坑**：`nohup` 不加载代理 → HF 超时；EgoSchema 无公开 GT，用作者 submission 比对。

---

# PruneVid 复现

**设置**：LLaVA-OV-7B，R=25%，32 帧，官方 `prunevid` 分支

| Benchmark | 复现 | 论文 | Δ |
|:---|:---:|:---:|:---:|
| VideoMME | **44.89** | 45.0 | -0.11 |
| MVBench | 进行中 | 47.6 | — |
| EgoSchema | 进行中 | 49.0 | — |

**结论**：VideoMME 与论文几乎一致，说明 Stage A/B 实现可信。

---

# VidCom² 复现

**设置**：`run_all.sh` 串行跑 Table1/2 + 效率 + Qwen

| 阶段 | 状态 |
|:---|:---|
| MVBench smoke (limit=5) | ✓ 通过 |
| 数据 prefetch | MVBench / EgoSchema / VideoMME ✓ |
| **MLVU_dev** | **卡住**（285GB，仅 ~35GB，ReadTimeout） |
| Table 1 正式评测 | 未开始 |

**风险**：MLVU 8 个 `video_part_*.zip` 经 xethub CDN，无代理时极易超时；当前进程在 resume 循环中，**大概率失败或极慢**。

**建议**：先跳过 MLVU，跑 MVBench/LVB/VideoMME；MLVU 单独 `source env.sh` 后断点续传。

---

# AOT 复现

**设置**：LLaVA-OV-7B，`run_eval.sh`，Sinkhorn OT + 锚点选择

| Benchmark | 10–25% | 与论文 |
|:---|:---|:---|
| MVBench | 完成 | Δ ≤ 1pt |
| EgoSchema | 完成 | 94–99% 预测一致 |
| LongVideoBench | 完成 | 低 2–3pt |
| VideoMME | 10/15/25% ✓，**20% 进行中 (~36%)** | 低 ~3pt |

**观察**：MVBench 贴近论文；VideoMME/LVB 系统性偏低，可能与视频缓存或帧采样有关。

---

# 面试可讨论（算法向）

1. PruneVid 的 Q→V 剪枝 vs AOT 的 text-agnostic：各适合什么任务？
2. VidCom² 的帧独特性 vs AOT 的 OT 聚合：能否组合？
3. OT 代价矩阵用 cosine 距离的合理性？Sinkhorn 收敛性？
4. 静态/动态分离（PruneVid）vs 独特性分数（VidCom²）哪个更稳？
5. R=10% 时谁更能扛？AOT 论文 claim 97.6% 的原因？

---

# 谢谢老师！
## 欢迎提问与指正

- PruneVid: arXiv:2412.16117
- VidCom²: arXiv:2505.14454
- AOT: arXiv:2603.01400
