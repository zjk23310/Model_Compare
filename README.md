# Models Compare

基于魔搭 ModelScope 平台的大语言模型部署与横向比较实验项目。

## 1 项目简介

本项目主要基于魔搭 ModelScope 平台，对多个国产大语言模型进行部署与测试，并比较不同模型在中文语义理解场景下的表现。

实验主要完成以下内容：

- 使用魔搭平台与阿里云 CPU 云服务器部署大语言模型
- 使用 Jupyter Notebook 构建模型运行环境
- 对多个国产大语言模型进行推理测试
- 针对中文复杂语义问题进行横向比较分析

本项目测试的模型包括：

- Qwen-7B-Chat
- ChatGLM3-6B
- Baichuan2-7B-Chat

---

## 2 项目环境

| 项目 | 配置 |
|---|---|
| 平台 | ModelScope |
| 操作系统 | Linux |
| Python版本 | Python 3.10 |
| AI框架 | PyTorch |
| Notebook环境 | Jupyter Notebook |

---

## 3 模型下载

进入工作目录：

```bash
cd /mnt/workspace
```

下载对应模型：

```bash
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git

git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git

git clone https://www.modelscope.cn/baichuan-inc/Baichuan2-7B-Chat.git
```

---

## 4 环境配置

创建虚拟环境：

```bash
conda create -n qwen_env python=3.10 -y
```

激活环境：

```bash
source /opt/conda/etc/profile.d/conda.sh

conda activate qwen_env
```

安装依赖：

```bash
pip install torch transformers modelscope
```

---

## 5 模型运行

编写推理脚本：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "./Qwen-7B-Chat",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "./Qwen-7B-Chat",
    trust_remote_code=True
)

query = "上海有哪些景点"

response, history = model.chat(tokenizer, query)

print(response)
```

运行：

```bash
python run_qwen_cpu.py
```

---

## 6 测试场景

为了测试不同模型的中文理解能力，项目采用了一些中文歧义句与长难句进行测试，例如：

```text
冬天：能穿多少穿多少
夏天：能穿多少穿多少
```

```text
他知道我知道你知道他不知道吗？
```

```text
明明明白白白喜欢他，可她就是不说。
```

```text
领导：你这是什么意思？
```

通过这些问题观察模型对于中文语义、指代关系以及双关含义的理解能力。

---

## 7 模型比较

### ChatGLM3-6B

- 中文理解能力较强
- 资源需求较低
- 适合轻量化部署
- 在本次测试中表现最稳定

### Qwen-7B-Chat

- 综合能力较均衡
- 对话生成自然
- 代码与文本生成能力较好
- 部分复杂歧义句理解略弱

### Baichuan2-7B-Chat

- 多领域能力较强
- 中文任务表现较好
- 对硬件资源要求较高
- 部分复杂长难句理解存在偏差

综合测试结果来看：

- ChatGLM3-6B 整体表现最好
- Qwen-7B-Chat 表现较稳定
- Baichuan2-7B-Chat 在复杂语义问题中表现略逊

---

## 8 项目结构

```text
Models_Compare
│
├── README.md
├── run_qwen_cpu.py
├── run_chatglm_cpu.py
├── run_baichuan_cpu.py
├── notebooks
│   └── model_compare.ipynb
└── results
    └── compare_result.txt
```

---

## 9 实验总结

本实验完成了多个国产大语言模型的部署与测试，并对不同模型在中文语义理解场景下的表现进行了比较。

实验结果表明，不同模型在中文复杂语义任务中的表现存在明显差异。在实际应用中，应根据任务需求、硬件资源以及部署场景选择合适的大语言模型。

---

## 10 项目地址

ModelScope：

```text
https://modelscope.cn/models/ZJK23310/Models_Compare
```

GitHub：

```text
https://github.com/zjk23310
```
