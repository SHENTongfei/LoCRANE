# MINING_v3 预注册挖掘设计（TAIL-CERT B3 新模型, 2026-09-29, 教授令"新模型上重挖"）

## 对象（与旧 v2 挖掘严格区分，旧版已隔离于 quarantine_v2_mining/）
- 模型 = **B3 stage0 全队列 25 ckpt**（runs_v3/ckpt_cg/cg_k_f{f}_s{s}.pt, CraneZV3,
  n_modules=256 / 19 免疫通道 / ridge_residual 头 / 三带序数 / dil-τ 1.4, final-epoch 状态——披露）
- 数据 = 1715 全队列 OOF（每样本 = 其 val 折 5-seed 集成）, L 带 795 / M 带 738 / Y 带 182
- 泄漏防护 = 所有归因只用各折 val 数据（OOF）, 禁碰 inner-train

## 菜单（M1-M4, 判读规则读数前定死, 防 p-hack）
1. **M1 模块 do-ablation（256 模块）**: 输入级置零模块 k 的全部基因（Xq+Xs, 互斥 assignment）
   → ΔMAE_L / ΔSpearman_L（相对 25-ckpt 集成 base）。读出 = top-10 关键模块 + 其基因内容。
   **无显著性门（rank 读出）；稳定性 = 5 seed 间 ablation 曲线 Spearman 均值（≥0.9 记稳）**。
2. **M2 基因 GIP 重要性（3000 HVG）**: |∂age/∂x|·|x| 于各 ckpt val 折, 5 seed 均值 →
   全队列基因重要性 rank。top-30 进文献矩阵。
3. **M3 尾部差异基因（TAIL-CERT 专属新轴）**: 基因 GIP 在 L 带 vs M 带的对比
   z_gene = (mean_L − mean_M) / pooled_std → top-50 尾上/尾下基因;
   **子对比 L-certified（CADENCE 单侧证书, 全队列 8.8%）vs L-uncertified** 的 GIP 差 →
   "减速签名基因"候选（性别分层披露, 对齐 CADENCE-S 二态）。
4. **M4 免疫通道 19 细胞对比**: GIP wrt 免疫输入, L vs M z-contrast → 尾部信号承载细胞
   （复核旧 F 线"免疫承载尾部"主张在新模型是否成立——旧版隔离后此为新证据）。

## 文献矩阵 v3（top 命中, PubMed eutils 经网关 1099, 摘要级）
- 分级: A=直接相关（该基因+衰老/长寿/血液）; B=家族/通路机制; C=基因已知但组合新;
  NONE=无文献 = **新发现候选（NO-FAIL: 只列候选, 不报"发现"）**
- 输出: runs_v3/mining_v3_lits.json（每基因: 标题 top3 + 分级 + PMID）

## 判读总规则
- 全部数字 = 冻结 ckpt OOF 归因, 零重训零调参; 候选 = 候选（湿实验验证前禁写"标志物"）
- 与旧 v2 结果若重合 = 复现, 若矛盾 = 以 v3 为准（旧版隔离, 禁互引）
