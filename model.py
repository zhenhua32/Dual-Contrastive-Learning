import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    模型
    """
    def __init__(self, base_model, num_classes, method):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.method = method
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in base_model.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state
        # hiddens shape: (batch_size, seq_len, hidden_size)
        cls_feats = hiddens[:, 0, :]
        # 这边是正常的分类任务, 所以只需要 cls_feats
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))
        else:
            # 这边是对比学习任务, 需要 cls_feats 和 label_feats
            # WARN: 注意下标签的位置和长度, 这要求每个标签只有一个 token
            # label_feats shape: (batch_size, num_classes, hidden_size)
            label_feats = hiddens[:, 1:self.num_classes+1, :]
            """
            Here is the explanation for the code above:
            1. cls_feats: (b, d) the feature of the last layer of the BERT model
            2. label_feats: (b, c, d) the feature of the label embeddings
            3. predicts: (b, c) the predicted scores for each label
            """
            """
            这行代码是使用torch.einsum函数来计算两个张量的乘积并求和。² torch.einsum函数可以用一种简洁的方式来表示多维线性代数运算，它遵循爱因斯坦求和约定（einsum）。³ 这个约定有三条基本规则：³

            - equation是一个字符串，它由输入张量的维度标签、箭头和输出张量的维度标签组成，如"ik,kj->ij"。
            - 输入张量的维度标签按照顺序排列，用逗号分隔，如"ik,kj"表示两个输入张量，第一个有i和k两个维度，第二个有k和j两个维度。
            - 输出张量的维度标签按照想要的顺序排列，只出现在箭头右边的标签叫做自由指标，它们表示输出张量保留的维度；只出现在箭头左边的标签叫做哑指标，它们表示要对输入张量进行乘积并求和的维度。

            根据这些规则，你给出的代码可以解释为：

            - cls_feats是一个二维张量，有b和d两个维度；
            - label_feats是一个三维张量，有b、c和d三个维度；
            - predicts是一个二维张量，有b和c两个维度；
            - predicts中每个元素都是cls_feats中对应行与label_feats中对应平面（第三个维度）进行点积并求和得到的。

            源: 与必应的对话， 2023/3/12(1) torch.einsum — PyTorch 1.13 documentation. https://pytorch.org/docs/stable/generated/torch.einsum.html 访问时间 2023/3/12.
            (2) 一文学会 Pytorch 中的 einsum - 知乎. https://zhuanlan.zhihu.com/p/361209187 访问时间 2023/3/12.
            (3) einsum is all you needed - 知乎. https://zhuanlan.zhihu.com/p/542625230 访问时间 2023/3/12.
            (4) torch.einsum详解 - 知乎. https://zhuanlan.zhihu.com/p/434232512 访问时间 2023/3/12.
            """
            # 总之, 求了个和, predict 的 shape 是 (batch_size, num_classes)
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }
        return outputs
