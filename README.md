# Agentic RAG

**无向量、无切分的文档问答 Web UI**：让模型先读懂文档的树状目录，再顺着目录逐层"翻阅"，命中哪几页就把那几页交给模型作答。

基于 [PageIndex](https://github.com/VectifyAI/PageIndex) 的推理式检索（reasoning-based retrieval）能力，用 Streamlit 把「模型自主翻阅文档」这套 agentic 检索流程做成一上手就能用的本地知识库界面。

<p>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="streamlit" src="https://img.shields.io/badge/streamlit-1.x-ff4b4b">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

![界面截图](docs/screenshot.png)

▶ **[75 秒演示视频](docs/demo.mp4)**：在「数据集市场」勾选财报数据集 → 输入问题 → 模型翻阅目录树定位页码 → 流式给出带引用的回答。演示数据为公开财报，完整过程无剪辑。

## 它和向量 RAG 有什么不同

| | 向量 RAG | 本项目（PageIndex） |
| --- | --- | --- |
| 索引 | 切块 + 向量化 + 向量库 | 生成文档的**层级目录树**（章节 → 小节 → 页） |
| 检索 | 相似度 top-k | 模型**读目录 → 选章节 → 取原文**，逐层缩小范围 |
| 可解释性 | 命中若干片段 | 命中的**章节节点与页码**会高亮出来 |
| 超长文档 | 切块丢上下文 | 结构完整保留，跨章节问题也能答 |

没有向量库、没有 chunk 参数，索引结果就是一个 JSON 树 + 页面文本。

## 功能

- **三步建库**：`上传文档` → `数据集目录` → `建立索引`，全程在左侧边栏
- **数据集市场**：右侧顶部把每个目录渲染成一张卡片，勾选即切换问答范围（都不勾 = 全库问答）
- **过程可见**：章节树随模型调用实时高亮、已读取页码可展开、思考过程独立展示
- **流式可中断**：回答边生成边渲染，随时可停，已生成内容保留在会话里
- **纯本地存储**：索引落在 `.pageindex/`，上传件落在 `uploads/`，不依赖任何数据库
- **单文件应用**：全部 UI 与编排逻辑都在 `app.py`，方便按需改造

## 快速开始

要求 Python **3.10+**（CI 覆盖 3.10 / 3.12）。

```bash
git clone https://github.com/<your-name>/agentic_rag.git
cd agentic_rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # 国内网络慢可加: -i https://pypi.tuna.tsinghua.edu.cn/simple

# 配置模型服务(Windows 用 copy)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 编辑 secrets.toml, 填入 llm_base / llm_key / llm_model
```

启动：

```bash
streamlit run app.py     # 或 python -m streamlit run app.py
./run.sh                 # Windows: run.bat（会自动找项目内 .venv）
```

打开 http://localhost:8503，左侧上传一份 PDF → 建立索引 → 在右侧提问。

> 首次启动只有空库：`.pageindex/` 和 `uploads/` 会在第一次建索引时自动创建，不需要手工准备任何数据。
> 还没填 key 也能打开界面，只是建索引和问答会提示补配置。

## 配置模型服务

任意 **OpenAI 兼容网关**都能用（官方 OpenAI、自建网关、公司内网网关均可），会请求 `{llm_base}/chat/completions`。

| 键（secrets.toml） | 环境变量 | 说明 |
| --- | --- | --- |
| `llm_base` | `LLM_BASE` | 网关地址，默认 `https://api.openai.com/v1` |
| `llm_key` | `LLM_KEY` | API Key，**必填** |
| `llm_model` | `LLM_MODEL` | 模型标识，**必填**；需支持 function calling |

优先级：`secrets.toml` > 环境变量。想先确认通不通：

```bash
python scripts/check_llm.py
```

未配置时应用仍可打开，但建索引与问答会提示补配置。

## 使用流程

1. **左侧 · 创建数据集**：上传 PDF → （可选）选一个数据集目录，或新建目录 → 点「建立索引」
2. **右侧 · 数据集市场**：勾选要问的目录；**都不勾 = 全库问答**
3. **提问**：输入框或下方示例问题；回答上方可展开章节树、命中页码与思考过程
4. **左侧 · 数据集浏览**：只读查看某目录下有哪些文档，不参与问答范围

> 数据集 = 文档的 `metadata.category` 分组。`datasets.json` 可以给每个数据集写一句人工介绍（不提供则自动生成）。

## 项目结构

```
.
├── app.py                      # Streamlit 应用(UI + 索引/问答编排)
├── pageindex/                  # 内置 PageIndex SDK 0.2.10(MIT, 见 NOTICE.md)
├── .streamlit/
│   ├── config.toml             # 主题、性能、关闭文件监听
│   └── secrets.toml.example    # 模型服务配置模板
├── datasets.example.json       # 数据集人工描述模板
├── scripts/check_llm.py        # 模型服务连通性自检
├── requirements.txt
├── pyproject.toml              # 元数据 + ruff 配置(不走打包安装)
├── run.sh / run.bat            # 一键启动
├── .github/workflows/ci.yml    # 语法检查 + 导入自检 + 启动自检
├── LICENSE                     # MIT
├── NOTICE.md                   # 内置 SDK 与第三方说明
└── README.md
```

运行时生成（均已在 `.gitignore` 中）：

```
uploads/          # 上传的 PDF
.pageindex/       # 索引库(树结构 + 页面文本 + manifest)
```

## 工作原理

1. **建索引**：`flash` 模式解析 PDF 的版面与标题层级 → 生成目录树，每个节点带一句 LLM 摘要（这一步会调一次模型）
2. **问答**：模型先拿到目录树 → 决定打开哪些章节 → 通过 `get_page_content` 取回对应页面原文 → 基于原文作答
3. **范围约束**：勾选数据集后，工具层会对 `doc_id` 做硬限制，模型只能读到范围内的文档

## 常见问题

**`pip install` 卡住 / 很慢？**
换国内镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`。依赖里 `litellm`、`openai-agents` 依赖树较大，首次安装几分钟属正常。

**提示 `streamlit: command not found`？**
用 `python -m streamlit run app.py`，不依赖脚本是否在 PATH 里。

**改完代码刷新页面没变化？**
`.streamlit/config.toml` 里关闭了文件监听（避免每次 rerun 扫描 `.venv` 上万文件）。改完代码重启进程即可。

**建立索引一直转圈／报模型不可用？**
应用在建索引前会先发一次极小的探活请求；失败时直接把网关返回的错误贴出来。先跑 `python scripts/check_llm.py` 定位是地址、Key 还是模型名的问题。

**索引数据在哪？怎么清空？**
在 `.pageindex/`（固定在项目目录下，与启动时的 cwd 无关），删掉目录即回到空库。`.pageindex/manifest.json` 记录文档名、页数与摘要。

**可以用哪些模型？**
任何支持 function calling / tool use 的模型。索引阶段的摘要生成对模型能力要求不高，问答阶段建议用强一些的模型。

## 安全与隐私

- 密钥只存在本地 `.streamlit/secrets.toml` 或环境变量，仓库仅提供模板
- 文档与索引只落本地磁盘；问答时仅**命中的页面文本**会随请求发给你配置的模型网关
- 仓库不含任何示例文档、索引或业务数据

## 许可

MIT。内置的 PageIndex SDK 版权归 Vectify AI，详见 [NOTICE.md](NOTICE.md)。
