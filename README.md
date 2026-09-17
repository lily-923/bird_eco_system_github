#鸟类表型与生态性状分析

这是数字图像处理综合实践的作业。用来分析已知物种的鸟类图片：
先把鸟从背景里分割出来，算一些图案方面的指标，再和 AVONET 里这个物种的形态、
生态数据做个对比，看看单张图片和这个物种的整体水平差多少。

功能大概有这些：
- 单张分析：选一张图，程序做去噪、裁剪和分割，计算指标，再和同物种的基线比较差额和
  z 分数，同时给出 AVONET 形态比例的分位对照。
- 批量分析：选一个文件夹批量跑一遍，导出 CSV/Excel、标注图和一份 HTML 报告，
  可以把偏离比较明显的个体筛出来。
- 处理过程可视化：把原图、增强、裁剪、分割、边缘、GLCM、LBP、对称轴、分形、
  颜色直方图这 10 个步骤都显示出来，点缩略图就能切换。
- 数据浏览：可以检索 200 个物种，查看样本图片和每个物种的 AVONET 数据。

数据用的是 CUB-200-2011，200 个类别，一共 11788 张图，官方划分是训练 5994 张、
测试 5794 张。AVONET 是鸟类的形态和生态性状数据。原始图片体积太大，没有放进仓库，
需要的话可以从下面两个地址下载：

CUB-200-2011：https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1

AVONET：https://figshare.com/s/b990722d72a26b5bfead

仓库里放的是预处理之后的结果：models_out 里是特征矩阵和物种基线，
data 里是 CUB 和 AVONET 的映射表。

运行方法，先装依赖：

pip install -r requirements.txt

然后按顺序执行：

python scripts/01_build_traits.py
python scripts/02_extract_full.py
python scripts/03_build_baseline.py
python scripts/04_run_gui.py

Windows 上也可以直接双击 启动应用.bat。

目录大概是：src 放代码，scripts 是运行脚本，models_out 是特征和基线，
outputs 是图和报告，data 是数据说明和映射表，docs 是程序说明。

说明：
随机种子固定为 42，跑出来的结果基本可以复现。
