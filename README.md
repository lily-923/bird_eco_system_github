# 鸟类表型—生态性状分析系统

**鸟类表型—生态性状分析系统**（数字图像处理综合实践  
应用场景：**鸟类生态研究 + 保护区监测质控**

> 输入已知物种的鸟类图像（单张或整批）→ 传统图像处理与分割 → 量化体表图案指标
> → 与同物种基线比较得到个体差额/z 分数 → 对照 AVONET 生态性状 → 输出分析报告。
> **不做物种预测；不使用任何深度学习/大模型。**

## 软件定位

| 项目 | 内容 |
|---|---|
| 面向用户 | 生态学/鸟类学研究者、保护区监测人员、课程教学 |
| 解决问题 | 体表图案人工测量费时且主观；测量结果与生态性状脱节；批量图片缺少标准化质控 |
| 核心能力 | 单张个体分析、批量批次筛查、指标差额量化、异常个体筛选、分析报告导出 |
| 技术路线 | OpenCV + numpy 传统图像处理；GLCM/LBP/形貌/颜色/分形/对称性；统计差额分析 |

## 主要功能（界面 5 个页签）

1. **主页**：软件信息、数据集概况、应用场景说明、快捷入口；
2. **数据浏览 / 检索**：检索 200 种鸟、查看样本与 AVONET 性状、数据集统计图；
3. **单张个体分析**：选择已知物种图像 → 自动确定类别（外部图片手动选择）→
   显示“图像指标 vs 同物种基线”差额表 + AVONET 参考指标；
4. **批量批次分析**：选择文件夹（按子文件夹自动识别类别，或整批统一指定物种）→
   进度条处理 → 结果表格 + 异常筛选（|z|>2）→ 导出 CSV/Excel/标注图/HTML 报告；
5. **处理过程可视化**：原图/增强/裁剪/分割/边缘/GLCM/LBP/对称轴/分形/颜色
   共 10 张步骤图，可点击切换，右侧显示算法说明与数值。

## 指标与差额定义

- 图像指标：图案复杂度(分形维数)、左右/上下对称度、纹理对比度/熵/同质性、
翼展/尾长/体长/体宽等归一化形态比例（形态代理，无比例尺）、
  前景面积占比、形状紧致度、边缘密度、形状长宽比、灰度标准差；
- **差额 = 当前图片指标 − 该物种全部图片均值**；
- **z 分数 = 差额 / 标准差**；|z|>2 记为“明显偏离个体”，用于监测质控筛查。

## 运行与启动

```
python scripts/01_build_traits.py      # CUB↔AVONET 映射表（200/200）
python scripts/02_extract_full.py      # 全量特征库（11788 × 126）
python scripts/03_build_baseline.py    # 200 个物种的指标基线
python scripts/04_run_gui.py           # 启动界面（调试模式）
```

Windows 快捷方式：
- `启动应用.bat`：无控制台启动（像普通软件）
- `启动界面.bat`：调试模式（报错可见）
- `打包成exe.bat`：用 PyInstaller 打包成独立应用（需已安装 PyInstaller）

## 目录结构

```
bird_eco_system/
├── assets/app.ico        应用图标
├── data/                 CUB-200-2011 + AVONET + 映射表
├── src/
│   ├── data_io/          数据读取与校验、AVONET 映射
│   ├── preprocessing/    去噪/增强/裁剪/分割
│   ├── features/         126 维特征 + 分步可视化
│   ├── analysis/         指标差额(compare) + 批量(batch) + 报告(report)
│   ├── models/           特征库构建、PCA 无监督聚类（不做预测）
│   └── gui/              Tkinter 应用界面
├── scripts/              01 映射 / 02 特征库 / 03 基线 / 04 界面 / 05 自检
├── models_out/           特征矩阵 + 物种指标基线
├── outputs/              统计图、PCA 图、批量结果与报告
└── docs/                 程序说明
```

## 约束与可复现性
- 数据集不删减：启动自动校验 200 类 / 11788 张 / 官方划分 5994-5794；
- 无预测模型、无深度学习；PCA 图仅作无监督展示；
- 随机种子固定 seed=42。

---

## 可复现性与开源

### 1. 随机种子（seed）
- 全流程随机种子固定为 **42**，定义于 src/config.py：SEED = 42；
- set_seed() 同时设置 Python 
andom 与 
umpy 的随机种子；
- 使用到随机性的环节（如 PCA 降维）均显式传入 
andom_state=SEED；
- 数据读取顺序、特征顺序、训练/测试划分均由固定规则确定，保证结果可复现。

### 2. 原始数据获取（不入库，体积大）
- CUB-200-2011：https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
- AVONET：https://figshare.com/s/b990722d72a26b5bfead
- 放置方式见 data/README_data.md（data/CUB_200_2011/ 与 data/AVONET/）。

### 3. 预处理数据集（随仓库提供）
为满足“代码与预处理数据集上传”的要求，仓库包含以下预处理结果（体积小、可直接复现）：
- models_out/feature_matrix_v1.npy：全量 11788 × 126 特征矩阵
- models_out/feature_meta_v1.csv：每张图像的编号/类别/训练测试标记
- models_out/species_metric_baseline.csv：200 个物种 × 17 项指标的基线
- data/cub_avonet_traits.csv：200 个类别的 CUB↔AVONET 性状映射表

原始图像与 AVONET 原始表格因体积较大未纳入仓库，请按第 2 节链接下载后按目录放置，即可完整复现。

### 4. 复现步骤
`
python scripts/01_build_traits.py
python scripts/02_extract_full.py
python scripts/03_build_baseline.py
python scripts/04_run_gui.py
`

### 5. 上传 GitHub（二选一）

**方式 A：GitHub Desktop（推荐新手）**
1. 安装并登录 GitHub Desktop（用你的 GitHub 账号）；
2. 菜单 File → Add local repository…，选择 D:\\bird_eco_system；
3. 左下角填写提交说明（如 Initial commit），点 **Commit to main**；
4. 点上方 **Publish repository**，填写仓库名（如 ird-phenotype-analysis），
   取消勾选 “Keep this code private”（如需公开），点 Publish 即完成上传。

**方式 B：命令行（需已安装 Git）**
`
cd /d D:\\bird_eco_system
git config user.name  "你的GitHub用户名"
git config user.email "你的GitHub邮箱"
git commit -m "Initial commit: 鸟类表型—生态性状分析系统"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
`
其中 git push 时会要求登录：密码位置请填写 GitHub 的 **Personal Access Token**
（GitHub → Settings → Developer settings → Personal access tokens），
或使用 gh auth login / Git Credential Manager 完成认证。

**上传前检查**：本仓库已通过 .gitignore 排除
data/CUB_200_2011/、data/AVONET/（原始大文件）、work/、outputs/batch_results/、.idea/；
models_out/ 中已包含预处理特征矩阵与物种基线，满足“代码 + 预处理数据集”的上传要求。
