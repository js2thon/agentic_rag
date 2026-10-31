# 第三方组件与来源说明

本仓库以开源方式发布，应用代码（`app.py` 等）与内置的 PageIndex SDK 均采用 **MIT** 许可（见 `LICENSE`）。

## PageIndex SDK（内置源码）

| 项目 | 说明 |
| --- | --- |
| 上游仓库 | https://github.com/VectifyAI/PageIndex |
| 内置版本 | 0.2.10 |
| 许可 | MIT，Copyright (c) 2025 Vectify AI |
| 位置 | `pageindex/` |
| 是否修改 | 未修改（原样内置） |

**为什么内置而不是 pip 依赖**：应用依赖该版本的 flash 索引与本地存储（`storage_path`）行为，内置源码可保证克隆后开箱即用、行为可复现。若要跟随上游升级，用上游同名目录整体替换即可。

## 运行时依赖

见 `requirements.txt`，均为公开发布的开源包（Streamlit、LiteLLM、OpenAI SDK、PyPDF2、pypdfium2 等），各自的许可条款以对应项目为准。

## 数据集与应用数据

本仓库**不包含**任何示例文档、索引库或用户上传内容：

- `uploads/`：运行时保存上传的 PDF，已被 `.gitignore` 忽略
- `.pageindex/`：本地索引存储，已被 `.gitignore` 忽略
- `.streamlit/secrets.toml`：模型服务密钥，已被 `.gitignore` 忽略（仓库只提供 `secrets.toml.example` 模板）
- `datasets.json`：数据集的人工描述，已被忽略（仓库只提供 `datasets.example.json` 模板）
