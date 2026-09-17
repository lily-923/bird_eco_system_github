# 数据集放置说明（先读我）

本项目需要两部分数据，**都不要删减**。

## 1) CUB-200-2011 鸟类图像数据集（200 类 / 11788 张）

官方下载（约 1.2 GB）：
- https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
- 备用：http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

下载后解压，把解压出的 `CUB_200_2011` 文件夹内容放进：
`data/CUB_200_2011/`
最终应长这样（这是程序识别数据的依据，目录别改）：

```
data/CUB_200_2011/
├── images/                  # 200 个子文件夹、共 11788 张
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   └── ...
├── images.txt               # 11788 行
├── image_class_labels.txt   # 11788 行
├── bounding_boxes.txt       # 11788 行
├── train_test_split.txt     # 11788 行
└── classes.txt              # 200 行
```

校验：images 子目录数 = 200，图片总数 = 11788。

## 2) AVONET 生态性状数据

官方下载（figshare，作者 Tobias et al. 2022, Ecology Letters）：
- 分享链接：https://figshare.com/s/b990722d72a26b5bfead
- 论文页：https://doi.org/10.1111/ele.13898

推荐下载文件：`AVONET Supplementary dataset 1.xlsx`
（内含 3 个工作表：AVONET1_BirdLife / AVONET2_eBird / AVONET3_BirdTree，
11 个形态性状 + 体重 + 6 个生态变量 + 分布范围）

把文件放进 `data/AVONET/`，命名任意，程序会自动识别；建议命名：
- `AVONET1_BirdLife.xlsx`（首选，11009 种）
- 若只有 csv 也放这里即可（`AVONET1_BirdLife.csv`）

> 注意：程序会同时生成一张 `CUB→AVONET` 映射表
> `data/cub_avonet_traits.csv`（200 行），由代码自动完成并校验。

## 3) 你自己的测试图（可选）

单张/多张鸟类图片（jpg/png）放进 `data/user_images/`，界面里可直接选。

---

---

> 注意：本项目 data/ 目录已随项目自带完整数据集（CUB-200-2011 全量与 AVONET），
> 正常运行无需再下载。若需重新下载，按上方官方链接操作即可。
