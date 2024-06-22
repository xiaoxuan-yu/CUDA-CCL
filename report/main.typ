#import "@preview/arkheion:0.1.0": arkheion, arkheion-appendices

#show: arkheion.with(
  title: "CUDA 驱动的并行连通域标记",
  authors: (
    (name: "余笑轩", email: "xiaoxuan_yu@pku.edu.cn", affiliation: "化学与分子工程学院"),
  ),

  abstract: [
  ],
  keywords: ("连通域标记","并行计算","CUDA"),
  date: datetime.today().display("[month repr:short] [day], [year]"),
)
#set cite(style: "chicago-author-date")
#show link: underline

// set chinese fonts
#set text(
  font: ("Times New Roman", "SimSun", "Noto Serif CJK SC"),
)

#let indent = 2em
#set par(first-line-indent: indent)

#let indent-par(body) = par(h(indent) + body)

= 连通域标记问题

#indent-par[连通域标记（Connected Component Labeling, CCL）是计算机视觉中的一种基本操作，广泛应用于图像分割、目标检测和图像分析等领域。它的主要任务是识别和标记图像中相连的像素块，即连通域。连通域标记在图像处理、模式识别和计算机视觉的许多应用中起着关键作用。]

在图像中，连通域是指所有像素值相同且通过某种连通性准则（如4-连通或8-连通，如 @CCL 所示）相连的区域。连通域标记算法的目标是为每个连通域分配一个唯一的标签，以便后续的图像处理和分析工作。具体而言，本次作业将实现对于二维二值图像的8-连通域标记算法。
#figure(
  grid(
        columns: (auto, auto),
        rows:    (auto, auto),
        gutter: 0em,
        [ #image("./figure/Square_4_connectivity.png",   width: 60%) ],
        [ #image("./figure/Square_8_connectivity.png", width: 60%) ],
    ),
  caption:[4-连通性和8-连通性的示意图 @enwiki:1192036140],
) <CCL>

= 算法
== CCL的串行算法: 并查集
#indent-par[基于并查集的串行算法是一种经典的连通域标记算法。并查集 (Union-Find) 是一种常用的数据结构，能够高效地处理连通域标记问题。并查集主要包含两个操作：查找 (Find) 和合并(Union)，如 @UnionFind 所示。

- 查找: 确定某个元素属于哪个连通域。
- 合并: 将两个连通域合并为一个。]

#figure(
  image("./figure/UnionFind.png", width:60%),
  caption:[并查集示意图 @8798895],
) <UnionFind>

#indent-par[基于并查集的算法通过逐像素扫描图像，使用并查集记录和合并相邻像素的连通信息，从而实现连通域标记。具体步骤如下：

1. 初始化并查集，每个像素作为一个独立的集合。
2. 逐像素扫描图像，对每个像素检查其上方和左方像素的连通情况，进行合并操作。
3. 第二次扫描图像，对每个像素进行查找操作，确定其最终的连通域标签。]

== GPU 并行的 CCL: Komura Equivalence 算法
#indent-par[对于并查集的并行化并不是显而易见的。传统的并查集算法是基于串行处理的, 直接并行化会面临许多挑战, 尤其是在处理等价类合并时, 需要解决多个线程之间的同步和冲突问题。为了有效地在GPU上实现并行的连通域标记, 研究者们提出了多种改进方案, KE(Komura-Equivalence) 算法就是其中之一。KE算法@KOMURA201554 通过一系列步骤来实现高效的GPU并行连通域标记, 其过程如 @KE 所示:]

- 初始化: 为每个像素分配一个唯一的初始标签，通常使用其线性索引值。
- 等价类更新: 在此步骤中, 多个GPU线程并行处理像素, 检查每个像素与其相邻像素的连通性，并更新等价类信息。这一步骤通常需要多次迭代, 直到所有像素的标签稳定下来。
- 标签压缩: 使用路径压缩技术对等价类进行压缩，确保所有等价像素的标签一致。这一步骤通过在并查集的“查找”操作中进行路径压缩来实现。
- 标签传播: 将最终标签传播到所有连通像素，确保每个连通域的所有像素共享相同的标签。

#figure(
  image("./figure/KE.jpg", width:80%),
  caption:"KE算法示意图",
) <KE>


#indent-par[
  KE算法通过分阶段处理和并行化技术, 有效地克服了传统并查集在GPU上的并行化困难, 提高了连通域标记的效率。2018 年，Allegretti 等人给出了 8 连通的 KE 算法 @8708900，该算法主要对 @KE 中的 reduction 部分进行了改进，如 @KE8 所示。
]
#figure(
  image("./figure/KE8.png", width:80%),
  caption:"8-连通的 KE 算法示意图",
) <KE8>

= 实现
我们使用 `CUDA` 实现了 KE 算法，并对其进行了性能测试，参见 `algorithm/` 文件夹下的相关代码。
```bash
.
├── CMakeLists.txt
└── src
   ├── alg
   │  ├── KE.cu
   │  └── UFTree.cuh
   ├── CMakeLists.txt
   ├── include
   │  ├── alg_runner.cuh
   │  ├── Matrix.cuh
   │  └── timer.cxx
   ├── main.cu
   └── serial.cxx
```


= 结果与讨论
== 正确性验证
#indent-par[以作业中给定的串行程序作为参考，我们对并行程序的结果进行了验证。在正确性验证中，一个难以处理的问题是即使结果正确，标签的顺序和值也都可能不同。因而，我们实现了一个映射算法，将并行程序的标签映射到串行程序的标签，从而验证两者的结果是否一致。映射算法的实现非常平凡，我们将每个标签对应的像素坐标全部记录并进行匹配，从而得到一个标签之间的映射表。根据这个映射表执行映射后，我们可以直接比较两个 `label` 数组是否一致从而验证正确性。正确性验证的相关代码参见 `validation/val.py` 和 `validation/validation.ipynb`。此处给出核心功能函数的实现。]
```python
def get_match_dict(labels_a, labels_b):
    uni_labels_a = np.unique(labels_a)
    uni_labels_b = np.unique(labels_b)
    labels_a_dict = {label: [] for label in uni_labels_a}
    labels_b_dict = {label: [] for label in uni_labels_b}
    for i in range(labels_a.shape[0]):
        for j in range(labels_a.shape[1]):
            labels_a_dict[labels_a[i][j]].append(i * labels_a.shape[1] + j)
    for i in range(labels_b.shape[0]):
        for j in range(labels_b.shape[1]):
            labels_b_dict[labels_b[i][j]].append(i * labels_b.shape[1] + j)
    match_dict = {}
    for label in uni_labels_a:
        label_index_list = labels_a_dict[label]
        for b_label in uni_labels_b:
            b_label_index_list = labels_b_dict[b_label]
            if len(label_index_list) != len(b_label_index_list):
                continue
            if all(
                [
                    label_index_list[i] == b_label_index_list[i]
                    for i in range(len(label_index_list))
                ]
            ):
                match_dict[label] = b_label
                break
    return match_dict


def map_with_dict(labels, match_dict):
    labels = labels.copy()
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            labels[i][j] = match_dict[labels[i][j]]
    return labels


def map_label(labels_a, labels_b):
    match_dict = get_match_dict(labels_a, labels_b)
    labels_a = map_with_dict(labels_a, match_dict)
    return np.all(labels_a == labels_b), labels_a
```
== 性能测试

== 计算速度的分析与讨论

= 总结

// Add bibliography and create Bibiliography section
#bibliography("bibliography.bib")



