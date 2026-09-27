# 菌踪旅图 · 169 类菌类图像匹配模型

本仓库公开“菌踪旅图”项目的 PyTorch 模型权重、类别配置与最小推理代码。模型以 ViT-Base/16 为骨干，输入为 592 × 592 RGB 图片，输出训练标签中的 169 类匹配分数。

## 文件

| 文件 | 内容 |
| --- | --- |
| `best_model.pth` | PyTorch `state_dict` 权重，在 [v1.0 发布页](https://github.com/xiaohan-2005/junzong-mushroom-vit/releases/tag/v1.0)下载 |
| `config.json` | 输入尺寸、归一化参数及 169 类标签映射 |
| `predict.py` | 单张图片的 Top 3 推理示例 |
| `requirements.txt` | 推理所需 Python 依赖 |

训练图片、数据集、申报材料、项目成员资料及网站源码不在此仓库中。

## 快速使用

需要 Python 3.10+。先克隆仓库，再从发布页下载 `best_model.pth`，放到仓库根目录：

```bash
git clone https://github.com/xiaohan-2005/junzong-mushroom-vit.git
cd junzong-mushroom-vit
python -m pip install -r requirements.txt
python predict.py path/to/your-image.jpg
```

权重文件的 SHA-256：`FC5BC54C36A78101624012C83533B5A3E77FBC8070FA2C54E228267512BAC61F`。下载后可用文件哈希核对完整性。

代码按 `config.json` 对图片缩放和归一化，加载权重后输出 Top 3 标签及各标签在本次 169 类结果中的相对分数。相对分数不是经独立校准的真实概率。

## 模型边界

- 类别名称来自训练标签；169 类并不覆盖云南或其他地区的全部野生菌。
- 本仓库不提供独立测试集、物种鉴定准确率或毒性、可食用性验证结果。
- 不能依据本模型结果采摘、购买、加工或食用野生菌。物种与食用安全判断须由专业人员结合实物特征确认。
- 项目网站的浏览器端 ONNX 推理是该模型的另一种部署形式；本仓库提供原始 PyTorch 权重。

## 许可

本仓库中由项目团队拥有权利的代码与模型权重按 [MIT License](LICENSE) 提供。第三方训练图片及其许可不随本仓库发布。
