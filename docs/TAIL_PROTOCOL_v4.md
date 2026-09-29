# TAIL-CERT 研究协议 v4（蓝海重定义版）——健康尾部假设 × 双不变量 × 统计显著的减速老化

**日期**：2026-09-26 · **状态**：呈报待批准，零修改 · 上游文件：`ADVERSARIAL_AUDIT.md`（数据层审计 S1-S12）、`REDESIGN_v3.md`（八层审计 + D-1 新外部 + 兵器库）
**一句话**：数据不换、问题重立——从"做一个更好的 aging clock"（红海）重立为 **"证明分子时间在健康尾部是 off-scale 的，并且可以单侧保证地判定"**（领域空档）；深度学习模型 = 为该问题量身重训的 **TAIL-CERT**；所有不变量是模型无关的后验统计（老 25 ckpt 也能出故事），重训是"尾部对齐目标"增强（挂了故事不死）。

---

## 0. 问题重定义（蓝海立题，三句讲清）

| 旧问题（红海，弃） | 新问题（蓝海，立） |
|---|---|
| "深度双模态模型能否比线性基线更好地判别 RLL/RYC（70/50 年龄分带）？" | **"在例外长寿（90+）尾部，基于普通人群标定的分子时间（aging clock）系统性失准：尾部是 off-scale 的。这一失准由两个方法学独立的不变量刻画——表征几何的减速（步速不变量（pace invariant））与同预测年龄下的波动稳态（噪声不变量）——并且可以有限样本保证的方式判定（statistically guaranteed decelerated aging）。"** |

为什么这样立题才是蓝海（撞车实据，EPMC 检索 2026-09-26）：
- "centenarian AND cross-ethnic" transcriptome = **0 hits**；"conformal AND transcriptome aging" = **0 直接先例**（表观遗传 UQ 有 3 篇先例：2024 CQR-in-epigenetic-clocks、2026 BayesAge conformal UQ——**引用并声明 first-on-transcriptome + first one-sided guaranteed-deceleration**）；"clock miscalibration at the healthy tail" = **0**（没人批评过 clock 在尾部的标定，也没人提出"尾部校准"概念）。
- 立题策略 = **field-level 命题**（对整个 clock 文献的批评 + 新范式），不是"我们模型更好"。审稿人不会拿 clock 文献堆压你——你在批评的就是那一堆文献。

**标签 v3（循证，替 70/50）**：`Y ≤50（n=182）/ M 51–89（n=909）/ L ≥90（n=624）`；三柱支撑：sebastiani2012/deelen2019（正文已引的 90+/100+ 例外长寿约定）+ 内部 labels.csv v1 的 ≥90 规则 + D-1a（Cell Rep Med 2026, 811 LLI+940 YC）摘要的 LLI 表型。cohort2/GTEx/D-1a 同切三带。

---

## 1. 故事主干（7 beats，journal-facing）

1. **问题**：所有 aging clock 都在"主干"（普通人群）上拟合；例外长寿者（90+）落在分布尾部——**尾部分子时间 off-scale**（命题：tail 上时钟失准，deceleration 加速）。
2. **资产**：东亚 1,400+ 90+ 例（内部 624 + D-1a 811 LLI，待 GSE 取证）× 两族裔（中/以）× 两平台（RNA-seq/芯片）× 组织参考（GTEx）。
3. **不变量 A（几何）**：尾部的表征"速度"变平——分子时间沿 90+ 的斜率显著小于 70–89（**步速不变量（pace invariant）**，I6b 已有种子：RLL age~PC1 ρ−0.253 vs RYC +0.386）。
4. **不变量 B（噪声）**：同预测年龄下，L 带的预测波动更窄——**波动稳态**（同方差增益，F5 "难老化亚群"是第一块砖）。A/B 方法学独立（确定性轨迹 vs 随机性，不同读法、不同统计量）。
5. **方法学赠品**：**statistically guaranteed decelerated aging**——split-conformal 单侧（上界）区间，"PI 上界 < 历龄"= 有限样本误差保证的"统计显著的减速老化"（超越 3 篇表观 UQ 先例：转录组 + 单侧 + 组级保证）。
6. **机制**：尾部信号由**免疫活性（ssGSEA 通道）而非组成（LM22 通道）**承载（通道换消融）× **性别分叉路由**（女读 BTNL3/HIST1H3B 系、男读 EBF1/IL24/FCRL2 系，4-way 分层折证除混杂）× **NK-effector 接线**（wiring 头跨平台迁移 ≥ level 头）。
7. **含义**：(a) 领域——clock 应做尾部校准（tail-calibrated clocks，field call）；(b) 转化——**guaranteed 5 基因验证面板**（挖掘第二梯队 ∩ ssGSEA 驱动 ∩ FiLM top，摘要级文献各 1+ 条）；(c) 方法——guaranteed aging-phenotype 可作为可检验表型进入 longevity 研究。

**英文 claim 一句话**（abstract 首句候选）："Aging clocks calibrated on the general population miscalibrate at the healthy tail: in over 1,400 individuals aged 90+ across two ancestries and two platforms, decelerated molecular aging manifests as two independent invariants—flattened representational pace and noise homeostasis at matched predicted age—and can be guaranteed with finite-sample guarantees."

---

## 2. TAIL-CERT 主模型设计（重训方案：做什么、目标怎么定、损失怎么写）

> 原则（DSR INNO + 用户"模型要重训、问题要重定义"）：**模型为目标而设计**——把"off-scale 假设"写进目标函数（不是事后统计）；同时**所有主不变量设计成模型无关读法**（老 ckpt 也能出故事 → 重训失败不塌故事）。
> 沿用资产：backbone（GeneModulePool + Transformer + FiLM + 免疫支 + Ridge meta-feature 残差头）不推翻，改动=问题对齐的五处（T1–T5）；H58 逐项 2-seed 快筛。

### 2.1 输入侧（三塔，"状态-潜力-组成"）
| 塔 | 内容 | 来源 | 融合规则（takeover T5 六规） |
|---|---|---|---|
| 状态 | 3000 HVG 表达（rank-gauss/z 双支，现状保留） | 内部 1715 / D-1a / cohort2(QTran) / GTEx(QTran) | — |
| 潜力 | **PLM 序列第三塔**：PIACE human 塔在盘 embedding 切 3000 HVG CDS（esm2_650M + esm_c_600M + prot_t5_xl 三塔起步），静态序列特征 | `WorkBuddy/PIACE/data/embeddings/human/`（零下载、零重编码，slicing 原则） | 单塔先独立（AUC 须超当前最弱腿）→ bottleneck-128 维一统 → **USDF 权重内部 CV 定** → 小 sigmoid 门 → 同折 CV 负优化即删 |
| 组成/活性 | 双免疫通道：LM22-QC（自动弃 3 常数细胞列，22→19）∥ ssGSEA 4 通路分（复用 d=1.064 管线） | 现有脚本 | 通道换消融 {LM22 / ssGSEA / 双 / 无}——**机制假设 H4 的载体** |

### 2.2 输出侧（4 头，各对应一个问句）
| 头 | 目标 | 设计创新 |
|---|---|---|
| H-1 全局年龄头 | z-scored 历龄（Ridge meta-feature 残差，**α 协议钉死 0.5**——O-1 复算确认） | 现状保留（主锚） |
| **H-2 尾部膨胀带内位头（核心创新）** | 带内位置 τ ∈[0,1]：Y/M 带按自然跨度，**L 带按 1.4× 膨胀跨度**（τ_L=(chrono−90)/33 vs 自然 /23）——把"90+ 减速"假设**写进监督信号**（"时间膨胀先验"） | **可证伪**：2-seed 快筛"膨胀目标 vs 自然目标"，OOF tail-gap/AUC 任一腿退化 → 关线回自然目标（H58 铁律）；通过=假设的主动证据 |
| H-3 序数三带头 | P(Y)/P(M)/P(L) 序数约束 + label smoothing；全 1715 + 逆频率权重（弃 162/162 下采样） | 数据效率 19%→100%；L 带真标签（≥90） |
| H-4 不确定度读法（非训练头） | 5-seed std（epistemic）+ MC-dropout K=20（aleatoric 代理）+ **split-conformal 单侧上界**（324 OOF 内部 fit，coverage 0.90 预注册） | statistically guaranteed decelerated aging 生成器（§3 不变量 B + 认证） |
| H-5 wiring 头（后验，不回传） | 模块级注意力边统计（入度/枢纽负载）→ 小 MLP 预测带内位 offset | 假设 H5（wiring>level 迁移）的载体；快筛 + 置换安慰剂门 |

### 2.3 损失（写全，预注册）
`L = L_age(SmoothL1, z) + L_ordinal(序数 logit 三档, label smoothing 0.1) + L_bandpos(H-2, SmoothL1 on τ) + w_fus·Σ门L1`
- **尾部对齐项 T-v**（H-2 膨胀目标即先验；快筛两臂：膨胀 1.4× vs 自然 1.0×，单变量）
- **域对抗项 T-d（可选，INNO-4）**：MFM 预训练 + 梯度反转域判别（域=cohort/平台，无标签池含 GTEx/cohort2）→ 模块表征跨队列不变；门=TI 不掉 + EX(QTran) AUC≥+0.01
- 弃项：SupCon（现状死代码——要么真挂双损失快筛、要么删头，二选一）；OPS 类复合目标彻底不进损失（指标侧已废，§4）

### 2.4 结构手术（INNO-5，随重训 0 额外成本）
FiLM 末层 2.16M→135k（r=16 低秩）；删 `ridge_proj`/`supcon_proj` 死参数；`w_gene/w_imm` → 2 参数可学 sigmoid 门 + L1；fold1 `ridge_residual=0.0` 硬编码 → 两臂快筛统一值。

### 2.5 训练协议
- 数据：全 1715（三带标签，逆频率权重）；外层 324 池 5-fold OOF 口径（H8 嵌套：折内全数据训练、折外 OOF 评估）；5 seeds×5 folds=25 ckpt；早停与报告**同 α**（M5 修复）；run_registry + config_used 随 ckpt 存（T1）。
- D-1a（GSE 到手后）并入为"扩展训练腿"（同族裔同表型 → 直接加样；不重训老模型，只训新协议版）。
- 预算：快筛批（每创新 2-seed×TI/EX，2–4h/项）≈ 2–3 天 GPU 空档；全量 25 ckpt 重训 8–12h（共享卡规则：不杀他人任务，NT/EB 动态批+断点纪律照旧）。

### 2.6 技术波次（教授点单：BLM/PLM · MoE · 对比学习 · 强化学习 · 自适应门控 · 微调 + 蓝海加件）

> 纪律（DSR INNO 铁律）：**每个技术=一个假设载体，不为炫而炫**；单一变量、2-seed 双腿快筛（TI 324-OOF + EX QTran 外部）、任一键退化关线+三折阴性档案；波次依赖：W2 需 W1 过线、W3 需 W2 过线。
> **红线自曝**：diffusion/合成 L 带样本增广——**故意不做**（DSR Appendix-C Type-IV：训练集含合成样本=直接红卡）；"尾部数据少"的解法=真数据（D-1a 811 LLI），不造数据。

**W1 · 核心对齐波（随 Phase-3 首筛，6 项）**

| 技术 | 在 TAIL 故事里的角色 | 落地 | 快筛门（2-seed 双腿） |
|---|---|---|---|
| **自适应门控融合**（教授点） | "哪个模态带尾部信号"本身是假设 → 门读**样本统计量**（表达均值/平台嵌入/带先验）逐样本出模态权重，替全局固定 2 参数门 | gate-net：小 MLP(stats→softmax over {表达/序列/免疫})，L1 稀疏 | 自适应门 vs 全局门：全族指标不掉 + 门分布跨平台可解释（D-1a vs cohort2 门谱不同=平台自适应证据） |
| **对比学习·三视图恒等**（教授点，复活 supcon 死代码） | "同一个人的表达/免疫/序列三塔必须指向同一身份"= 跨模态 InfoNCE（正样本=同一样本三塔，负=批内） | supcon_proj 真挂双损失（与 §2.3 弃项二选一裁决）；dropout 双视图沿用 SimCLR 形 | TI 序数-ECE 不掉 + EX tail-gap 不亏；恒等读出（三塔 embedding 互检 cos）进补充 |
| **BLM/PLM 序列塔·微调档**（教授点） | 静态切片=基础档；**LoRA 微调档**：esm2_650M 末 4 层 LoRA r=16，在 1715 队列 CDS（按三带逆频率）上对比式微调 → "longevity-adapted ESM" 本身是可上传资产 | PIACE 在盘权重，零下载；LoRA 微调 ~4h/塔 | 微调塔 vs 冻结塔：单塔 AUC 须先过"当前最弱腿"（T5 规则 1）+ 双腿不亏 |
| **尾部微调阶段**（微调×故事） | "模型自己尾部对齐"：主干全数据训完后，**L 带专属 20-ep tail-adapt 阶段**（尾部膨胀目标头激活）→ 尾部假设写进训练轨迹 | 两阶段（trunk → tail-adapt），断点式 | tail-gap 提升 且 主干（Y/M decile AUC）不回退（尾训不得伤主干=防"过拟合尾部"指控） |
| **MoE·尾部专家**（教授点） | "尾部跑不同的计算"= 可检验命题：Transformer FFN → 3 专家稀疏 MoE，**路由条件于带先验**（Y/M/L 条件 router，尾专家 L 带高激活）→ expert utilization 本身读出"off-scale 计算" | 3-expert 稀疏（top-1），router 读 τ̂；专家均衡正则 | MoE vs Dense（参数量匹配）：tail-gap 不亏 + 尾专家 L 带激活率显著>Y 带（利用度=假设读数，非装饰） |
| **neural aging-flow**（蓝海加件，替后验速度统计） | 不变量 A 从"后验统计"升级为"学出来的对象"：小 flow 头学 d·repr/d·chrono 衰老步速场（pace field, dM/dt；学术名=Pace of Aging，canonical=Belsky 2020 DunedinPACE/PNAS，我们=转录组×横断面×90+尾部×保证 三个 first） → pace-ratio 直接读 flow 输出（后验版保留为对照） | flow 头（d=128 两层）+ 速度一致性损失（沿 chrono 有限差） | flow 版 pace-ratio 与后验版相关 ρ>0.7（一致性门）+ H2 判死线不变 |

**W2 · 强化学习波（教授点：RL，两项都"炫且可杀"）**

| 技术 | 角色 | 落地 | 快筛门 |
|---|---|---|---|
| **RL-1 反事实抗衰老搜索**（in-silico geroprotection） | "哪些基因集/细胞分扰动能把预测分子龄拉下来"= 冻结 TAIL-CERT 当环境，PPO 小策略（动作=Top-32 模块通路的表达偏移档位），奖励=分子龄下降 且 带约束不越界 + 平滑正则 → 输出"计算抗衰老干预面"（5 基因面板的上位概念） | 纯推理期 RL（不重训主干）；策略 100 万样本 rollout；输出=干预面+轨迹 | vs 梯度 oracle（同预算）：RL 找到的干预面 EF@top-5 ≥ 梯度 0.8× 且 泛化到 EX 队列方向一致；**输了=诚实写"梯度更优"（阴性入 Limitations，不藏）**；湿实验措辞="in-silico predicted, pre-registered for validation"（takeover T9 纪律） |
| **RL-2 主动显著减速筛查预算**（guaranteed-fraction maximization under measurement budget） | "预算有限时测哪些样本能最大化 guaranteed 人数"= 序贯 conformal：PPO 选下一批测量对象（不确定性 × 带先验），奖励=新增 guaranteed 数/成本 → "targeted sig-decel screening protocol"（筛查×主动学习×aging=三线交集，0 先例） | 同上推理期 RL；内部 OOF 模拟预算 | significant-decelerator fraction@预算 k 显著 > 随机选 k（bootstrap CI 不交）；否则撤 |

**W3 · 加件波（蓝海加件，W2 过线后启）**

| 技术 | 角色 | 落地 | 快筛门 |
|---|---|---|---|
| **cohort 快适应**（meta-lite） | "换队列只要 k 个样本"：线性 probe + 5-shot 适应曲线（GTEx/cohort2/D-1a 各 k∈{5,20,100}）→ "cohort-adaptive clock" 读数 | 零重训（探针） | 适应曲线单调 + k=100 时 AUC gap ≤0.02（vs 无适应） |
| **不确定性校准**（conformal 家族补齐） | H4 之外补"区间宽度-难度"关系：MC-dropout σ 与 conformal 宽度互检（一致性=两种不确定性互证） | 推理期 | 互检 ρ>0.6；否则只用 conformal |

**W4 · LLM 工具波（赶时髦件，W3 后；不进主主张=工具层+湿实验优先级器，算力全卡本机边界）**

> 算力边界（本机实据）：WDDM 假 free，每进程真实可分 ~3.5GiB VRAM → 微调上限 = 4B 级 4-bit QLoRA（TEXWPT I5b 先例）；27B 级只推理（在盘 Bonsai2 27B 三元 PQ2_0，CPU ~1tok/s）；NVIDIA 学术 GPU 额度=升级档（8B QLoRA 过夜，可选）。

| 件 | 角色 | 落地 | 门/预算 |
|---|---|---|---|
| **W4-a TinyTail-LLM（蒸馏，主推）** | "in-silico gerontologist"：老师=Bonsai2-27B 在盘（零下载），仅 ~100 个 curated 样本（L 带高/低 guaranteed + 高不确定度亚群）生成减速评估报告（~2k token，CPU 过夜 5–6h）→ **蒸馏进 Qwen3.5-4B 4-bit QLoRA 学生**（~2.5GB 一次性下载走 1099 代理；GPU 1–2h，GPU-only 不设 CPU 档）→ 输入受试者分子画像→输出减速评估报告+5 基因验证优先级 | 学生=补充材料"AI 工具"展示+湿实验优先级排序；**主主张零依赖**（审稿面安全） | 蒸馏保真：student 报告关键结论与 teacher 一致率 ≥0.8（10 项 rubric）；不达则只留 teacher 版（CPU 慢但准） |
| **W4-b LLM 先验通道（炫技+预期阴性）** | 4B LoRA 训"模块 aging-relatedness + hallmark 激活"打分器 → 第五特征通道 | H58：vs 无先验/随机先验双腿（TI 324-OOF + EX QTran） | **诚实预期：死于与 ssGSEA 冗余（d=1.064 已占该生态位）→ 关线归档三折阴性**（三层检查记录，阴性=纪律展示） |
| 不做 | 27B 级微调（超算力）；LLM 蒸馏替代 clock 主张（循环论证）；新 27B 级下载（老师只在盘取） | — | — |

**波次预算**：W1 六项=Phase-3 快筛批本体（2–3 天 GPU 空档）；W2 两项=各 4–6h（推理期 RL，GPU 轻）；W3 各 <2h；**W4-a≈1 晚 CPU + 2h GPU + 2.5GB 下载；W4-b 并入 W3 窗口（2-seed 快筛）**。全量重训（25 ckpt）只在 **W1 核心四项（自适应门/三塔/膨胀目标/双免疫）过线**后启动；W2–W4 均为推理期/工具件，不占全量窗口。

---

## 3. 测量协议（测什么、判死线、多重性）——全部写进 protocol_lock v4 先于跑数

> 统计底座：不变量读法全部**模型无关**（在任意年龄预测器上计算）→ 老 25 ckpt（7.4 口径修复后）出第一版数字，TAIL-CERT 出加强版；两版并列（诚实 + 故事不塌）。

| 假设 | 主指标（公式级定义） | 判死线（预注册） | 多重性 |
|---|---|---|---|
| **H1 off-scale 主命题（尾部标定差）** | **tail-gap** = [median(M−chrono, 90+ 段)] − [median(M−chrono, 70–89 段)]（M=预测分子龄；OOF + D-1a 全 90+）；配套：逐年龄 decile 残差曲线（hero 图） | tail-gap ≤ −2 分子岁 且 BH 校正 p<0.05（324 OOF + D-1a 双腿一致）→ 确认；否则"未确认"，转机制搜救（三层：特征/模型类/功效，不判死路） | primary family |
| **H2 几何不变量 A（速度）** | **pace-ratio** = 表征沿 90+ 的年均移动距离 / 沿 70–89 的年均移动距离（25-ckpt 均值表征，10 岁 bin；I6b PC1 读法为简化版） | vr ≤ 0.85 且置换 p<0.01（管线级置换：同管线跑标签置换 1000 次取分布） | primary family |
| **H3 噪声不变量 B（波动稳态）** | **noise-ratio** = σ_5seed(L 带, 同带内 decile) / σ_5seed(Y 带, 同 decile)（MC-dropout K=20 复算同式） | nr ≤ 0.85 且置换 p<0.01；与 H2 相关性披露（>0.8 则合并表述为"单一致性因子"） | primary family |
| **H4 认证（statistically guaranteed decelerated aging）** | **significant-decelerator fraction** = P(PI_upper < chrono, L 带)；保真 = 经验覆盖率 ≥0.90±0.02（split OOF）；对照：M 带、Y 带 significant-decelerator fraction（应更低） | L 带 > M 带 > Y 带 单调 且 保真达标；单调性破 → 只报保真，弃组级声明 | primary family |
| **H5 机制-免疫（活性>组成）** | 通道换 Δ：{ssGSEA 通道}/ {LM22 通道} 在 tail-gap 与 significant-decelerator fraction 上的增量 | Δ_ssGSEA ≥ Δ_LM22 + 0.1d 且单侧 p<0.1（效应量口径，小样本） | secondary（披露） |
| **H6 机制-性别分叉** | 4-way 分层折（L-F/L-M/M-F/M-Y）内 FiLM γβ 基因 load × 组；组内 sex-AA 差 | 组内差 n.s.（证除混杂）且 两性 γβ top-genes 交集 ≤2（分叉证据，I5 已有种子） | secondary |
| **H7 机制-接线（炫技件）** | wiring 头 vs level 头在跨平台（cohort2/GTEx/D-1a）的 tail-gap 保持率 | wiring 保持率 ≥ level 保持率 且 置换安慰剂塌线；否则仅补充材料 | secondary |
| **H8 迁移矩阵** | 全部指标 × {内部/D-1a/cohort2/GTEx} 4 队列矩阵 + **CIQ**（双标定 d 极差 ≤0.3 预注册） | 方向 4/4 一致 或 3/4 一致且失败队列可归因（ECDF 脆弱项） | primary family |

**多重性协议**：primary family = {H1,H2,H3,H4,H8}（5 项 BH）；secondary = {H5,H6,H7}（仅披露不校正）；**管线级置换安慰剂**（组标签带内 shuffle × 全管线 × 1000 次）预注册为 H1/H2/H3 的假阳性对照。
**指标纪律（H59，替 OPS 的"不伦不类"）**：上表每个指标单一用途、单位/定义可权威溯源（分子岁、速度比、保证率、效应量）或预注册；**禁止任何阈值 sweep 复合分**；OPS 以"OPS-legacy（协议兼容项，阈值 sweep）"单行进补充材料并自曝身份。头号数字只报两个：**tail-gap（分子岁）+ significant-decelerator fraction（%）**——都 plain，都审稿人秒懂。

---

## 4. 数据与算力账（"数据更新了，能做多"）

| 资产 | 状态 | 解锁的能力 |
|---|---|---|
| 内部 1715（含 90+ 624 人） | ✓ 在盘 | H1/H2/H3 第一版（324 OOF 口径，老 ckpt 即可出数） |
| **D-1a：Cell Rep Med 2026 中国 811 LLI + 940 YC**（PMID 42030936, PMC13198318, OA） | **GSE 号待取**（NCBI/CNCB/Cell.com 本机不通；EPMC 全文未入库；下步：citation-finder 四源 / 换网络窗口一步取 Data Availability 段） | 尾部样本 624→**1,435**（L 带统计功效跨门槛）+ 同族裔真 90+ 主外部腿（H7 腿①） |
| cohort2（以，芯片，90+ n=10） | ✓ | 跨族裔腿（H7 腿②）+ ≥90 探针 |
| GTEx（西方，bulk 血/肌） | ✓（降格为参考腿，3 带重分带 + QTran + 弱目标披露） | 组织/族裔扩展 |
| D-1b：eBioMed 2025 三队列百岁老人 scRNA+质量细胞（NK 签名） | GSE 待取 | 机制 H5 的第三独立模态互证 |
| PIACE 11 塔 embedding（人 65,057 基因） | ✓ 在盘，零下载 | 序列第三塔（§2.1） |
| 老 25 ckpt + 挖掘产物（F/I 系列） | ✓ | 模型无关第一版数字 + 故事兜底 |

**算力预算**：Phase-0 取证/复算 ~1 天（CPU 为主）；快筛批 2–3 天 GPU 空档（每项 2–4h）；全量重训 8–12h 一次；MC/conformal/wiring 读法 <2h。

---

## 5. 执行路线（Gate 化，每条有判死/通过判据）

```
Phase 0（0.5–1d，P0 诚实急救 + 取证，零重训）
  O-1 α 口径复算钉死 protocol_lock · O-2 坏件/台账纠偏（19/25、±13000 源、aa_gtex_blood stale 隔离）
  O-3 plain 头对头落 CSV · D-1a/D-1b GSE 取证（Data Availability→GEO→年龄/许可/零交叠五查）
  GATE-0：α 口径定位完成 + OPS 台账修订 + GSE 取证 ≥1 项成功（失败→后备腿 GSE63027/cohort2-90+ 激活）
Phase 1（1d）协议冻结 protocol_lock v4（标签 v3、H1–H8 判死线、指标菜单、分析家族、QTran、α、fold/seed）
Phase 2（1d）特征层 v3（HVG k 行 + 生物模块池 + LM22 QC 门 + ssGSEA 双通道 + PLM 切片 + QTran/OVD 门）
Phase 3（2–3d GPU 空档）H58 快筛批 = **W1 六项**（§2.6：自适应门控/三视图对比/PLM-LoRA/尾部微调/MoE 尾专家/aging-flow）+ FiLM 低秩双门/双免疫通道/4-way
  GATE-1：双腿（TI 324-OOF + EX QTran）不亏才进全量；关线项留三折阴性档案
Phase 4（8–12h GPU）全量 25 ckpt 重训（TAIL-CERT，W1 过线配置）+ MC/K20 + wiring 头后验训练 + **W2 两项 RL（推理期）** + **W3 加件（W2 过线后）**
Phase 5（1–2d）测量全阵列（§3 八假设 + 置换 + 保真）+ H59 双族出表 + D-1a 腿并入
  GATE-2：H1 双腿确认 或 诚实降档（"未确认"+机制搜救三层）——两种结果都有可投版本
Phase 6（1–2d）写作重定位（§1 七拍主干 + claim-证据-威胁三栏表 + 图 v4 规划）
Phase 7 复现/交付（单源化 + run_registry + 九绿 submittable）
```

**风险登记（PUA：先查三层再判死）**：① D-1a GSE 取不到 → 后备腿 + "内部 624 人尾部"单腿版（命题降为 hypothesis-test 而非 confirm）；② H-2 膨胀目标快筛退化 → 回自然目标，H1 仍靠模型无关读法出数（故事不塌，只少一个主动证据）；③ 尾部 n 功效不足 → 判死线已按此设计（−2 分子岁 BH），不达标转 H5–H7 机制搜救；④ wiring 头过拟合 → 置换塌线即撤（只留阴性声明，诚实加分）。

---

## 6. 与两技能的条款映射（执行即按条走）

- **DSR**：H0f 三腿改判 = {324 OOF / D-1a（ext-1）/ cohort2（ext-2）}，GTEx=参考腿（H7 dev-uncontacted 独立腿 = D-1a，满足度最高）；H8 嵌套 CV；H58 快筛制（Phase 3）；H59 菜单（§3）；GATE-1 七坑 + 新三坑（口径复现/标签循证/自造指标）全进 protocol_lock v4。
- **takeover**：T2 头号数字 live 复算（tail-gap/significant-decelerator fraction 必须 raw prediction→指标，hearsay 不收）；T5 PLM 融合六规=§2.1 直接执行；T9 citation-finder 核验 D-1a/D-1b + 表观 UQ 三篇先例 + 5 基因文献（摘要级）；T10 submittable 九绿（旧六绿 + 口径复现/标签循证/conformal 保真/家族 FDR）。
- **handoff-archaeology**：标签定义考古（v3 单一事实源）+ index 去重坑 + 双树 provenance（O-8）并入 Phase 7。

---

## 7. 批准清单（v4，替 REDESIGN_v3 §7）

| 批 | 内容 | 说明 |
|---|---|---|
| **B0** | Phase 0 全项（O-1/O-2/O-3 诚实急救 + D-1a/D-1b GSE 取证） | 3h+网络窗口，零重训，先拆雷 |
| **B1** | protocol_lock v4 冻结（标签 v3 ≥90 三带 + H1–H8 判死线 + 指标菜单替 OPS + 分析家族 + QTran） | 先于跑数落盘 |
| **B2** | TAIL-CERT 快筛批 = W1 六项（§2.6）+ 工程件（H58 制，GPU 空档 2–3 天） | 批快筛≠批全量；波次依赖：W2/W3 需 W1 核心过线 |
| **B3** | 全量重训 25 ckpt（W1 过线配置）+ 测量全阵列 + **W2 RL 两项 + W3 加件**（推理期） | 看 B2 结果再批 |
| **B3.5（可选档）** | **W4 LLM 工具波**：W4-a TinyTail-LLM（Bonsai2 教师→Qwen3.5-4B 4-bit QLoRA 学生，1 晚 CPU+2h GPU+2.5GB 下载；学术 GPU 额度升级 8B 可选）+ W4-b LLM 先验通道（预期阴性归档） | 工具层不进主主张；单独批（含下载授权 2.5GB 一次） |
| **B4** | 写作重定位 + 图 v4（§1 七拍；W2 干预面/W3 快适应曲线入补充） | 看 B3 结果再批 |
| 不做 | 换题（L8）；**diffusion/合成样本增广（DSR Type-IV 红卡，自曝纪律：尾部落差用 D-1a 真数据补，不造数据）**；新 PLM 下载（只用 PIACE 在盘塔 + LoRA 微调）；杀他人 GPU 任务 | — |

**命名**：项目代号 **TAIL-CERT**（尾部校准·统计保证），论文标题候选 2：
1. "Molecular time runs off-scale at the healthy tail"（主推，field-level）
2. "Geometric and variational invariants of decelerated molecular aging, with finite-sample guarantees"（保守版）

**复现**：`audit_evidence.py`（审计证据）+ Phase 5 出 `measuring_v4/`（全指标一键脚本，G1 纪律：数字不硬编码，全从 raw prediction CSV 计算）。

## 勘误 COR-1（2026-09-29, 数据级实锤, 优先级最高）
**D-1a 改判=内部同源队列（非独立外部腿）**。取数实证（data/d1a/DATA_REPORT.md）：
Cell Rep Med 2026 (PMC13198318) Mendeley fpf4y72kfz 落盘后核验=
`labels_v3.ID == 该文 phen.ID`（1715/1715 全重合, 0 年龄/性别不一致; 内部矩阵列=论文数字样本码; 基因 11030⊂19229 同源）。
1751 raw − 1715 分析=36 QC 剔除（1 缺龄）无表型不可用。故：
① 本协议 H0f 三腿中 D-1a(ext-1)**作废**——它是训练集同源，计入外部腿=泄漏红卡；
② "尾部 624→1435（靠 D-1a 外部 811 LLI）"前提失效——该 811 LLI 已在训练集内；
③ 独立外部腿=cohort2(以色列芯片)+GTEx(组织参考) 维持不变；90+ 外部尾部的替代来源=
**另找真正独立的 90+/centenarian 血转录组队列**（检索任务已立项, 见 runs_v3 检索台账）；
④ D-1a 包保留价值=19229 基因 raw counts（特征扩展可选实验, 需预注册+教授批, 不污染已认证 B3 OOF 13.60）。
本勘误不动任何已出判决（H1/H4/CADENCE/CADENCE-S/H2/H3 均为内部 OOF/全队列口径, 不受影响）。

## 外部尾腿检索台账 COR-1b（2026-09-29, 登记待办, 未出结论=诚实留白）
目标：找**真正独立**的 90+/centenarian 全血转录组队列（替代作废的 D-1a 外腿）。
- PubMed "centenarian/long-lived × whole blood/PBMC × transcriptome/RNA-seq" = 21 命中，
  efetch 21/21 已拉（runs_v3/pm_hit_all.xml）；GEO db 检索接口当前降级（返默认 fallback），
  elink pubmed→gds 空；PMC 无 GeroScience/bioRxiv 全文。
- **头号候选 LILY/Sebastiani（意大利, n≈419-434 全血）= 待解析 GSE/DDBJ**：
  GeroScience PMID 41723297 / bioRxiv 40791342 / 重复 41361486, 41624884, 41738283。
  下一步=经浏览器直读 GeroScience 全文 Data Availability（eutils 降级不影响期刊站点）→
  拿到 GSE 后做标准三检（平台/90+ 例数/样本非本两队列）再立腿。
- **次候选**：Aging "centenarian vs septuagenarian/octogenarian 转录组"（PMID 27794564, 待核组织）；
  PBMC octo/nonagenarian n=18（PMID 24381713, 太小仅披露级）。
- 台账=runs_v3/{pm_ids.txt, pm_hit_all.xml, gds_search.xml, lily.xml}；**在 GSE 实证前,
  H0f 外部腿维持 {cohort2 + GTEx 参考}, 不引入未取数队列的任何数字（防造假红线）**。

## COR-1b 收口（2026-09-29, 检索墙评估完成, 登记阻塞）
LILY/Sebastiani（LLFS+ILO, 意大利, 全血, n≈419-434）取证链全通到数据层：
GeroScience 10.1007/s11357-025-02090-x → **Data Availability = "LLFS and ILO data available
from the ELITE portal"**（NIA 资助, eliteportal.synapse.org, 数据在 Synapse 上, 大概率 DUA 受限）；
分析脚本=github.com/montilab/li_et_al_llfs_transcriptomics（Zenodo 10.5281/zenodo.18329406,
24 脚本已落盘可查, 数据在 dds_gene rds, 非公开包）。访问墙=① api.synapse.org/api-integ 被网关封
(http=000 TLS EOF) ② Synapse REST 需认证 ③ aging-us 403 机器人墙 ④ eutils GDS 检索接口降级
(返默认 fallback) ⑤ bioRxiv 429 限流。**诚实状态=90+ 独立外部尾腿数据源已定位(ELITE/LLFS+ILO)
但访问未通, 在解锁前外部腿维持 {cohort2 + GTEx 参考}, 不引任何 90+ 外数字。**
解锁路径(按推荐序): A=教授浏览器打开 eliteportal.synapse.org 查 LLFS/ILO 数据集开放级
(公开→我给 5 分钟取数脚本; 受限→走 NIA 门户申请或邮件联系 montilab) B=等 eutils GDS
服务恢复(后台轮询器已挂, 恢复即自动重查) C=放弃 90+ 外尾腿, 90+ 主张靠内部 795 L +
CADENCE-S 性别二态支撑, 外部腿写"90+ 外部复制为 future work"(诚实且站得住)

## 决策锁 COR-2（2026-09-29, 教授定案）
① 19229 基因特征扩展 = **不做**（教授定案：同队列扩特征收益低中、污染已认证 B3 OOF 13.60 口径）；
② cg best-val 25 格重训 = **不做**（Leg C 实证 final≈best Δ0.018=0.14%, cosmetic only）；
③ 外部腿结构锁定：{cohort2（W2 WIN, k20 快适应）+ GTEx 参考（KILL 披露）}；90+ 外尾腿=
LILY/ELITE 解锁成功则补"90+ 外部复制"段落, 失败/超期则按 COR-1c 措辞写 future work。

## COR-1c（90+ 外尾腿 fallback 措辞预写, 供进稿直接贴）
"External replication of the 90+ (L-band) tail was not possible with an independent
centenarian transcriptome at submission: the only identified candidate resource
(ELITE portal, LLFS/ILO Italian whole-blood cohorts) requires restricted-access
application. Our external validation therefore rests on the Israeli whole-blood
microarray cohort (independent population; k=20-shot adaptation reduced MAE by
3.53 y, monotone) and on GTEx tissue reference (cross-platform disclosure only).
Centenarian-scale external replication is registered as future work."

## 决策 COR-1d（2026-09-29 教授裁定, 三路定案）
- **A（教授浏览器查 ELITE 开放级）= 放弃**：教授本机网络无法访问该站（"连网站自己都打不开"）。
- **B（lily_poller 自动解锁）= 继续跑**：低频后台（30min/次×48h），eutils 恢复即自动解析
  LILY GSE，命中落 runs_v3/lily_gds_poll.json 后自动停。**非阻塞**——稿件不等它。
- **C（fallback 措辞）= 定案采用**：外部验证腿结构正式锁定为
  {cohort2（以色列芯片全血, 独立人群, W2 k20 快适应 WIN Δ3.53y）+ GTEx 组织参考（KILL/跨平台披露）}；
  90+ 外部复制按 COR-1c 预写措辞写 **registered future work**（LILY/ELITE 受限申请解锁后补）。
- 效力：稿件 Results/Discussion 的外部验证章节据此定稿；LILY 若 B 命中, 只触发"增补一段
  90+ 外复制"的可选修订, 不改变已定稿结构。
