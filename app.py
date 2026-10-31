# -*- coding: utf-8 -*-
"""Agentic RAG — 本地 Web UI(基于 PageIndex 树索引)

参考 ChatGPT / Gemini / Grok 的前端设计:
  左侧边栏: 创建数据集(上传/目录/建索引) + 数据集浏览(选数据集看文档清单, 只读)
  右侧主区: 数据集市场(唯一的数据集/问答范围选择入口) + 对话区
           (空会话时"市场+欢迎区+输入条"整体居中; 提问后切成"对话区 + 吸底输入条")
           (两态由 CSS 依据 .pi-chat-mode 切换, 元素每轮重建, 不残留)
后端: 任意 OpenAI 兼容网关 (见 .streamlit/secrets.toml)

文档选择(参考 PageIndex SDK 原生能力), 入口统一在主区「数据集市场」:
  - 不选 = 全库问答(doc_id=None, agent 用 browse_documents 自主发现)
  - 勾选数据集 = doc_id=[该数据集全部文档id], 工具层 _allowed_ids 硬限制范围
  - 数据集 = 分类(metadata.category)分组; 左侧「数据集浏览」只是查阅, 不改范围
"""
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import streamlit as st

sys.stdout.reconfigure(encoding="utf-8")

from pageindex import PageIndexClient

# ---------- 页面配置 ----------
st.set_page_config(page_title="Agentic RAG", page_icon="🌳", layout="wide")

# ---------- 模型服务配置 ----------
# 任意 OpenAI 兼容网关都能用: POST {base}/chat/completions, Bearer 鉴权, 请求体带 "model"。
# 配置优先级: .streamlit/secrets.toml > 同名大写环境变量 > 默认值
#   llm_base  / LLM_BASE   网关地址(默认官方 OpenAI, 自建网关填自己的)
#   llm_key   / LLM_KEY    API Key(必填)
#   llm_model / LLM_MODEL  模型标识(必填)
DEFAULT_BASE = "https://api.openai.com/v1"


def _setting(name: str, default: str = "") -> str:
    """读配置: .streamlit/secrets.toml[llm_<name>] > 环境变量(LLM_<NAME>) > default"""
    secret_name = f"llm_{name}"
    val = ""
    try:
        val = st.secrets.get(secret_name, "") or ""
    except Exception:
        val = ""
    return (val or os.getenv(secret_name.upper(), default) or "").strip()


LLM_BASE = _setting("base", DEFAULT_BASE)
LLM_KEY = _setting("key")
LLM_MODEL = _setting("model")
# litellm 的 openai/ 前缀只用于选协议(OpenAI 兼容), 真正发给网关的是前缀后的名字
MODEL = f"openai/{LLM_MODEL}" if LLM_MODEL else ""

UNCATEGORIZED = "未分类"

# 项目内固定路径: 与启动时的 cwd 无关, 在任意目录执行 streamlit run 都落到同一份索引库
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(BASE_DIR, ".pageindex")

# ---------- 数据集(目录)分组与市场卡片 ----------
# 数据集 = 文档 metadata.category 分组; 每个数据集在主区顶部渲染成一张可选卡片。
_DATASET_DESC_FILE = os.path.join(BASE_DIR, "datasets.json")


@st.cache_data(ttl=60, show_spinner=False)
def dataset_descriptions() -> dict:
    """datasets.json: {"数据集名": "人工撰写的介绍"}, 缺失时回落到自动生成"""
    try:
        with open(_DATASET_DESC_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def group_by_category(docs: list) -> tuple:
    """[(数据集名, [文档...])] — 按分类分组, 未分类排最后"""
    buckets = {}
    for d in docs:
        cat = ((d.get("metadata") or {}).get("category") or "").strip() or UNCATEGORIZED
        buckets.setdefault(cat, []).append(d)
    order = sorted(buckets, key=lambda c: (c == UNCATEGORIZED, c))
    return [(c, buckets[c]) for c in order]


def clean_title(name: str) -> str:
    """文档名 → 可读标题: 去扩展名/序号/语言标注/文号"""
    t = re.sub(r"\.pdf$", "", name or "", flags=re.I)
    t = re.sub(r"^\s*\d+\s*[.\-、_]\s*", "", t)
    t = re.sub(r"[（(]\s*(中文|英文|Chinese|English|\d+)\s*[)）]", "", t)
    t = re.sub(r"[_\-\s]*TS\d{6,}[_\-\s]*", "", t)
    t = re.sub(r"\s+", " ", t).strip(" _-—.")
    return t or (name or "未命名文档")


def dataset_blurb(cat: str, items: list) -> str:
    """数据集内容描述: 优先用 datasets.json 的人工描述, 否则自动概括"""
    custom = (dataset_descriptions().get(cat) or "").strip()
    if custom:
        return custom
    titles = [clean_title(d.get("name", "")) for d in items]
    if len(titles) == 1:
        desc = re.sub(r"\s+", " ", (items[0].get("description") or "")).strip()
        if desc:
            return desc[:150] + ("…" if len(desc) > 150 else "")
        return f"《{titles[0]}》，可整篇纳入问答。"
    head = "、".join(titles[:3])
    more = f" 等 {len(titles)} 份文档" if len(titles) > 3 else f" 共 {len(titles)} 份文档"
    return f"收录 {head}{more}，可整类纳入问答。"


def set_scope(ids):
    """问答范围的唯一写入口(只由主区「数据集市场」调用)"""
    new = sorted(ids) if ids else None
    if new == st.session_state.doc_id:
        return
    st.session_state.doc_id = new
    st.session_state.messages = []
    # 范围控件靠 rev 换 key 重建, 否则控件仍停留在旧状态
    st.session_state.pi_scope_rev = st.session_state.get("pi_scope_rev", 0) + 1
    st.rerun()


# 全局样式: 仿 ChatGPT/Gemini 的简洁观感, 统一字号 + 响应式自适应
# 注意: st.markdown 注入的 <style> 落在主容器, 但 CSS 选择器全局生效,
# 侧栏规则仍有效; :root 变量对整个文档生效。
st.markdown("""<style>
/* ---------- 字号统一(rem 基准, 全站一套比例) ---------- */
:root {
  --pi-font: 0.95rem;      /* 正文 */
  --pi-font-sm: 0.82rem;   /* 次要/说明/树 */
  --pi-font-xs: 0.75rem;   /* 徽章/页脚 */
}
html, body, .stApp, .block-container,
.stChatMessage, [data-testid="stChatInput"] {
  font-size: var(--pi-font);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Microsoft YaHei", "Hiragino Sans GB", "Source Han Sans SC", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.stCaption, [data-testid="stCaptionContainer"] {font-size: var(--pi-font-sm);}

/* ---------- 对话消息: 提问与回复分两级字号, 回复略小更耐读 ---------- */
/* 提问(用户): 0.92rem; 回复(模型): 0.84rem, 行高 1.7 —— 只动 assistant 气泡 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stMarkdownContainer"] p {
  font-size: 0.92rem !important; line-height: 1.6;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] {
  font-size: 0.84rem !important; line-height: 1.7; color: #1f2328;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] td,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] th {
  font-size: 0.84rem !important; line-height: 1.7;
}
/* 段落/列表间距收紧: 长回答不散, 更像文档而不是逐行堆叠 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] p {margin: 0 0 .42em !important;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] ul,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] ol {margin: .2em 0 .5em; padding-left: 1.1em;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] li {margin: .12em 0;}
/* 小标题按 0.84rem 基准等比收敛, 层级清楚但不抢正文 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] h4 {
  line-height: 1.45; margin: .7em 0 .3em; font-weight: 600;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] h1 {font-size: 1.02rem !important;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] h2 {font-size: 0.96rem !important;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] h4 {font-size: 0.9rem !important;}
/* 表格: 比正文再小一档, 单元格留白收窄, 横向不挤 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] table {font-size: 0.8rem; margin: .35em 0 .6em;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] td {padding: .3rem .55rem !important;}
/* 引用块/代码块跟着正文降一档, 不与正文打架 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] blockquote {font-size: 0.82rem; margin: .3em 0 .5em;
  padding-left: .7em; border-left: 2px solid #e5e7eb; color: #4b5563;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stMarkdownContainer"] code {font-size: 0.8rem;}

/* 顶栏: 只隐藏菜单, 保留 header(侧栏折叠按钮挂在 header 上, 隐藏会导致收起后打不开) */
#MainMenu {visibility: hidden; height: 0;}
[data-testid="stAppDeployButton"] {display: none;} /* 去掉右上角 Deploy 字样 */
header[data-testid="stHeader"] {height: 0; min-height: 0; padding: 0;
  background: transparent; border: none; overflow: visible;}
.block-container {padding-top: 1rem; padding-bottom: 0; max-width: 100%;}

/* ---------- 侧栏折叠/展开入口: 保证"收起后还找得到、点得开" ---------- */
/* header 被压成 0 高, 原生展开按钮(stExpandSidebarButton)会被顶到 y<0,
   只露出半截几乎看不见; 这里改成固定定位钉在左上角, 并加浅色底衬 */
[data-testid="stExpandSidebarButton"] {
  position: fixed !important;
  top: 12px; left: 12px;
  width: 34px; height: 34px;
  display: flex !important; align-items: center; justify-content: center;
  border-radius: 9px;
  background: #fff;
  border: 1px solid #e4e7eb;
  box-shadow: 0 2px 6px rgba(17, 24, 39, .10);
  z-index: 1000001;
  transition: background .15s ease, box-shadow .15s ease, transform .12s ease;
}
[data-testid="stExpandSidebarButton"]:hover {
  background: #f2f4f7;
  box-shadow: 0 4px 12px rgba(17, 24, 39, .14);
}
[data-testid="stExpandSidebarButton"]:active {transform: scale(.94);}
[data-testid="stExpandSidebarButton"] svg {width: 18px; height: 18px; color: #4b5563;}
[data-testid="stExpandSidebarButton"]:focus-visible {
  outline: 2px solid #111; outline-offset: 2px;
}
/* 隐藏标题悬停时出现的锚点链接(»), 保持欢迎区干净 */
[data-testid="stHeadingWithActionElements"] [data-testid="stHeaderActionElements"] {
  display: none !important;
}
/* 收起态: 主区顶部让出按钮占的高度, 避免压住第一行内容 */
body:has([data-testid="stExpandSidebarButton"]) .block-container {padding-top: 3.2rem;}

/* 侧栏展开时, 折叠按钮默认只在悬停才显形(不易发现) → 常态淡显, 悬停加深 */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {visibility: visible !important;}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
  visibility: visible !important;
  opacity: .55;
  border-radius: 7px;
  transition: background .15s ease, opacity .15s ease;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover {
  background: #eceef1; opacity: 1;
}

/* ---------- 侧栏: 只调字号与留白, 不动宽度/布局(避免与内联样式冲突) ---------- */
section[data-testid="stSidebar"] {font-size: var(--pi-font-sm);}
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {font-size: 1rem; letter-spacing: .2px;}
section[data-testid="stSidebar"] h3 {margin-bottom: 0;}
section[data-testid="stSidebar"] hr {margin: 0.85rem 0;}
section[data-testid="stSidebar"] [data-testid="stExpander"] details {
  border-radius: 10px; border-color: #e9eaee; background: #fff;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-size: var(--pi-font-sm);
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: #9aa0a6; line-height: 1.5;
}
/* "创建数据集" 的分步小标题: 序号圆点 + 文案, 三步一眼看懂先后 */
section[data-testid="stSidebar"] .pi-step {
  display: flex; align-items: center; gap: 8px;
  margin: .35rem 0 .45rem;
  font-size: var(--pi-font-sm); font-weight: 600; color: #1f2328;
}
section[data-testid="stSidebar"] .pi-step > span {
  width: 17px; height: 17px; flex: 0 0 auto;
  border-radius: 50%; background: #1f2328; color: #fff;
  font-size: .66rem; font-weight: 600; line-height: 1;
  display: inline-flex; align-items: center; justify-content: center;
}
/* "数据集浏览" 的文档清单: 无项目符号, 名称左 / 页数右, 单行截断 */
section[data-testid="stSidebar"] .pi-files {
  list-style: none; margin: .1rem 0 0; padding: 0;
  max-height: 15.5rem; overflow-y: auto;
}
section[data-testid="stSidebar"] .pi-files li {
  display: flex; align-items: baseline; gap: 8px;
  padding: 3px 6px; border-radius: 6px; line-height: 1.45;
}
section[data-testid="stSidebar"] .pi-files li:hover {background: #f4f5f7;}
section[data-testid="stSidebar"] .pi-file-n {
  flex: 1 1 auto; min-width: 0; color: #2b3036;
  font-size: var(--pi-font-sm); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
section[data-testid="stSidebar"] .pi-file-p {
  flex: 0 0 auto; color: #a9aeb4; font-size: .7rem;
}

/* ---------- 对话滚动区: 高度按视口算, 随窗口/缩放自适应 ---------- */
/* 注意: 真正带高度的是外层 LayoutWrapper(flex:0 0 520px), 内层只负责滚动 */
.stMainBlockContainer [data-testid="stLayoutWrapper"]:has(.st-key-pi_history) {
  flex: 0 0 auto !important;
  height: calc(100vh - 210px) !important;
  min-height: 220px;
}
.stMainBlockContainer .st-key-pi_history {
  height: 100% !important;
  flex: 1 1 auto !important;
}

/* ---------- 欢迎页 ---------- */
.pi-hero {text-align: center; padding: 0 0 1.4rem;}
.pi-hero h1 {font-size: clamp(1.4rem, 3.5vw, 1.9rem); font-weight: 700;
  margin: 0 0 8px;}

/* ---------- 数据集市场(主区顶部横向卡片) ---------- */
.pi-market-title {font-size: var(--pi-font); font-weight: 700; color: #111;
  margin: 0 0 .6rem; letter-spacing: .01em;}
[class*="st-key-pi_market"] [class*="st-key-ds_card_"] {
  background: #fff; border: 1px solid #e8eaed; border-radius: 14px;
  padding: .75rem .9rem; min-height: 8.4rem; display: flex; flex-direction: column;
  gap: 3px !important; transition: border-color .15s ease,
  box-shadow .15s ease, background .15s ease;
}
[class*="st-key-pi_market"] [class*="st-key-ds_card_"]:hover {
  border-color: #cfd4da; box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
[class*="st-key-pi_market"] [class*="st-key-ds_card_"]:has(input:checked) {
  border-color: #111; box-shadow: inset 0 0 0 1px #111; background: #fcfcfd;
}
/* 卡内元素紧凑排布: Streamlit 默认 flex gap ~1rem, 这里压到 2px */
[class*="st-key-pi_market"] [class*="st-key-ds_card_"] [data-testid="stVerticalBlock"] {
  gap: 2px !important;
}
[class*="st-key-pi_market"] [class*="st-key-ds_card_"] [data-testid="stElementContainer"] {
  margin: 0 !important;
}
[class*="st-key-pi_market"] [class*="st-key-ds_card_"] [data-testid="stCheckbox"] {
  margin: 0; min-height: 0;
}
[class*="st-key-pi_market"] [class*="st-key-ds_card_"] [data-testid="stCheckbox"] label {
  margin: 0; padding: 0; min-height: 0;
}
[class*="st-key-pi_market"] [class*="st-key-ds_card_"] [data-testid="stCheckbox"] label p {
  font-weight: 700 !important; font-size: 1.06rem !important; color: #111;
  letter-spacing: .01em; line-height: 1.4; margin: 0 !important;
}
/* Streamlit 的 markdown 容器自带 p 字号/边距规则, 这里需 !important 才压得住 */
[class*="st-key-pi_market"] [class*="st-key-ds_card_"] .pi-ds-desc {
  color: #6b7280; font-size: 0.74rem !important; line-height: 1.55 !important;
  margin: 0 0 .3rem !important; display: -webkit-box; -webkit-line-clamp: 3;
  -webkit-box-orient: vertical; overflow: hidden;
}
[class*="st-key-pi_market"] [class*="st-key-ds_card_"] .pi-ds-meta {
  color: #a9aeb4; font-size: 0.68rem !important; margin: 0 !important;
  letter-spacing: .02em;
}

/* ---------- 两态切换: 只切 CSS 显隐, 不增删元素 ---------- */
/* 空会话(<body> 里没有 .pi-chat-mode) */
body:not(:has(.pi-chat-mode)) .stMainBlockContainer [data-testid="stLayoutWrapper"]:has(.st-key-pi_history) {
  display: none !important;          /* 空会话不要对话区 */
}

/* 空会话: "欢迎区 + 输入条"作为一组垂直居中(ChatGPT 式) */
body:not(:has(.pi-chat-mode)) [data-testid="stAppScrollToBottomContainer"] {
  display: flex; flex-direction: column;
  justify-content: safe center;      /* 视口过矮时不会把内容顶出可视区 */
}
body:not(:has(.pi-chat-mode)) [data-testid="stAppScrollToBottomContainer"] > div:not([data-testid]) {
  flex: 0 0 auto;                    /* 关掉 Streamlit 用来自动撑满的弹性占位 */
}
body:not(:has(.pi-chat-mode)) [data-testid="stMainBlockContainer"] {min-height: 0;}
body:not(:has(.pi-chat-mode)) [data-testid="stBottom"] {
  position: static !important;       /* 取消吸底, 跟着欢迎区一起居中 */
  background: transparent !important;
}

/* 对话态: 收起欢迎区、数据集市场与示例按钮, 输入条回到吸底 */
body:has(.pi-chat-mode) .pi-hero,
body:has(.pi-chat-mode) [class*="st-key-pi_market"],
body:has(.pi-chat-mode) [class*="st-key-sample"] {display: none;}

/* ---------- 快捷示例按钮(仅主区): 浅色圆角块, 不抢视觉 ---------- */
.stMainBlockContainer div[data-testid="stButton"] > button {
  border-radius: 16px; border: 1px solid #e6e8eb; background: #fff;
  color: #4b5563; font-size: var(--pi-font-sm); font-weight: 400;
  padding: 0.45rem 0.9rem; box-shadow: none; transition: all .15s ease;
  white-space: normal;               /* 窄屏时换行, 不出现省略号 */
  line-height: 1.3;
}
.stMainBlockContainer div[data-testid="stButton"] > button:hover {
  border-color: #cfd4da; background: #fafafa; color: #111;
}
/* 按钮内文字: 窄屏允许换行, 不用省略号 */
.stMainBlockContainer div[data-testid="stButton"] > button p,
.stMainBlockContainer div[data-testid="stButton"] > button div {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}

/* ---------- 树 ---------- */
.pi-tree{border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;
  font-size:var(--pi-font-sm);max-height:420px;overflow-y:auto;background:#fafafa;
  line-height:1.7}
.pi-node{padding:2px 6px;margin:1px 0;border-radius:4px;
  border-left:3px solid transparent}
.pi-node.hit{background:#fff3cd;border-left-color:#f0ad4e;font-weight:600}
.pi-badge{display:inline-block;margin-left:8px;font-size:var(--pi-font-xs);
  color:#888;background:#eee;border-radius:8px;padding:0 6px;white-space:nowrap}
.pi-sum{color:#999;font-size:var(--pi-font-xs);margin-left:6px}

/* ---------- 提问条下方免责提示: 渲染在底部容器内, 紧跟输入条 ---------- */
[data-testid="stBottom"] [data-testid="stCaptionContainer"] {
  text-align: center; color: #9aa0a6;
  line-height: 1.45; margin: 0.15rem 0 0; padding: 0 0.9rem;
}
/* 这句已占住原来的底部留白, 收掉多余 padding, 让它贴近视窗下沿(ChatGPT 式) */
[data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
  padding-bottom: 0.95rem !important;
}
[data-testid="stBottom"] [data-testid="stCaptionContainer"] p {
  font-size: 0.72rem !important; margin: 0;
}

/* Streamlit 自带的运行状态小挂件(右上角 Stop): 会自动跑到顶栏外露出半截,
   我们已有输入框内的停止键和"思考中"面板, 这里直接隐藏 */
[data-testid="stStatusWidget"] {display: none !important;}

/* ---------- 生成中的"停止"键: 参考 ChatGPT / Grok 的实心圆 + 白色方块 ---------- */
/* 只在 .pi-busy 存在时出现, 常驻 DOM 但平时隐藏(保证 widget key 稳定) */
[data-testid="stBottomBlockContainer"] > div {position: relative;}
.st-key-pi_stop {
  display: none !important;
  position: absolute; right: 16px; top: 12px;
  margin: 0; padding: 0; z-index: 5;
}
/* 脚本运行中 Streamlit 会把 widget 外层压到 opacity .33(看起来发灰、像禁用),
   停止键正是在"运行中"才出现, 必须解除这层压暗并放开点击 */
body:has(.pi-busy) .st-key-pi_stop {
  display: block !important;
  opacity: 1 !important; pointer-events: auto !important;
}
body:has(.pi-busy) [data-testid="stChatInputSubmitButton"] {visibility: hidden !important;}

/* 与原生发送键同规格(30x30 / 圆角 7.6px / #111 / 距右 16px), 生成时只换图标,
   按钮不会变大或移位; 图标为 10px 白色圆角方块(33%) */
.st-key-pi_stop button {
  width: 30px !important; height: 30px !important; min-height: 30px !important;
  padding: 0 !important; border: none !important;
  border-radius: 7.6px !important;
  /* 图标 = 内联 SVG 背景图, 由浏览器在按钮盒子正中绘制(background-position: center),
     不依赖伪元素定位/字体连字, 任何显示缩放下都不会偏心 */
  background: #111
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'%3E%3Crect width='10' height='10' rx='1.5' fill='%23fff'/%3E%3C/svg%3E")
    center / 10px 10px no-repeat !important;
  color: #fff !important;
  display: inline-flex !important; align-items: center; justify-content: center;
  box-shadow: none !important; cursor: pointer !important;
  opacity: 1 !important;
  transition: background-color .15s ease, transform .12s ease;
}
.st-key-pi_stop button:hover {background-color: #000 !important;}
.st-key-pi_stop button:active {transform: scale(.94);}
/* 文案与字体图标都收起来(文案仍在 DOM 里, 读屏可用) */
.st-key-pi_stop button p {display: none !important;}
.st-key-pi_stop button > div {gap: 0 !important;}
.st-key-pi_stop button [data-testid="stIconMaterial"],
.st-key-pi_stop button svg {display: none !important;}

/* ---------- 窄屏/矮屏: 只收留白与字号, 布局交给 flex 居中 ---------- */
@media (max-width: 640px) {
  :root {--pi-font: 0.9rem; --pi-font-sm: 0.78rem; --pi-font-xs: 0.72rem;}
  .pi-tree {max-height: 300px;}
}
@media (max-height: 820px) {
  .stMainBlockContainer [data-testid="stLayoutWrapper"]:has(.st-key-pi_history) {height: calc(100vh - 190px) !important;}
}
@media (max-height: 560px) {
  .pi-hero {padding: 0 0 .8rem;}
  .pi-hero h1 {margin: 0 0 4px;}
}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def build_client(base: str, key: str, model: str):
    """配置变了就重建客户端(参数进缓存 key), 换模型/换 key 立即生效"""
    return PageIndexClient(
        index={"model": model,
               "backend": {"api_key": key, "api_base": base},
               "storage_path": STORE_DIR},
        chat={"model": model,
              "backend": {"api_key": key, "base_url": base}},
    )


@st.cache_data(ttl=30, show_spinner=False)
def load_documents(_client, bust=0):
    """文档列表缓存 30s: 刷新页面不必每次重读磁盘
    (_client 前缀下划线 = 不参与缓存 key; bust 变号即强制重取)"""
    return _client.list_documents(limit=10000)["documents"]


# ---------- 索引前自检 / 索引超时保护 ----------
# 索引最后一步(flash 的摘要/扩展、standard 的整树构建)要调 LLM。
# 代理活着但上游模型不可用时, 请求会一直挂着——界面上就是"无限转圈",
# 用户既看不到原因也拿不到结果。这里先花 <=10s ping 一次, 把"静默卡死"
# 换成一句能照着排查的报错; 索引本身再加一道硬超时兜底。
INDEX_TIMEOUT = 1200.0     # 秒: 单篇文档索引的最长等待(20 分钟)
_probe_exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pi-probe")
_index_exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pi-index")


def probe_index_model(timeout: float = 10.0):
    """索引 lane 的模型能否在 timeout 内正常回一句 → (ok, 说明)"""
    if not (LLM_KEY and LLM_MODEL):
        return False, "未配置模型服务的 API Key 或模型标识"
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        LLM_BASE.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {LLM_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return True, ""
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            raw = e.read().decode("utf-8", "replace")
            detail = (json.loads(raw).get("error") or {}).get("message") or raw
        except Exception:
            pass
        return False, f"HTTP {e.code} · {detail.strip()[:200]}"
    except Exception as e:  # 超时 / 连接被拒 / DNS
        return False, f"{type(e).__name__}: {timeout:.0f}s 内接口无响应"


def index_with_timeout(fn, *args, **kwargs):
    """在独立线程里跑索引, 超时即返回 FutureTimeout(避免永久阻塞页面)"""
    fut = _index_exec.submit(fn, *args, **kwargs)
    return fut.result(timeout=INDEX_TIMEOUT)


# ---------- 树遍历可视化辅助 ----------

def parse_pages(spec):
    """'2-5' / '2,5' / '3' -> [2,3,4,5] / [2,5] / [3]"""
    pages = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                pages.update(range(int(a), int(b) + 1))
            except ValueError:
                pass
        elif part.isdigit():
            pages.add(int(part))
    return sorted(pages)


def _esc(t):
    return (str(t or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def render_tree_html(structure, highlight_pages, max_depth=4):
    """树 HTML: 页码范围与 highlight_pages 相交的节点高亮"""
    if not structure:
        return ""
    hl = set(highlight_pages or [])
    lines = []

    def walk(nodes, depth):
        for n in nodes:
            if depth >= max_depth:
                return
            s, e = n.get("start_index"), n.get("end_index")
            hit = bool(hl) and s is not None and not (
                (e or s) < min(hl) or s > max(hl))
            title = _esc(n.get("title") or "(untitled)")
            badge = (f'<span class="pi-badge">p.{s}-{e or s}</span>'
                     if s is not None else "")
            summary = _esc((n.get("summary") or "")[:90])
            sum_html = f'<span class="pi-sum">{summary}…</span>' if summary else ""
            cls = "pi-node hit" if hit else "pi-node"
            lines.append(
                f'<div class="{cls}" style="margin-left:{depth * 16}px">'
                f'{title}{badge}{sum_html}</div>')
            walk(n.get("nodes") or [], depth + 1)

    walk(structure if isinstance(structure, list) else [structure], 0)
    return '<div class="pi-tree">' + "".join(lines) + "</div>"


def render_pages_md(page_texts):
    """已读取页面内容(折叠展示, 每页截断)"""
    if not page_texts:
        return "（尚未读取页面）"
    parts = []
    for page, text in page_texts:
        t = (text or "").strip()
        if len(t) > 1200:
            t = t[:1200] + "\n\n…（截断，共 %d 字符）" % len(text)
        parts.append(f"**第 {page} 页**\n\n```\n{t}\n```")
    return "\n\n---\n\n".join(parts)


def _pump_events(stream):
    """后端事件流 → 独立线程 → 队列

    主线程用 0.5s 超时轮询队列: 即使后端一帧事件都不吐(代理卡住),
    也能持续调用 st.* 产出心跳, 让 Streamlit 的"停止"请求有机会被处理。
    """
    q = queue.Queue()

    def run():
        try:
            for ev in stream.events:
                q.put(("ev", ev))
            q.put(("done", None))
        except BaseException as e:      # 含网络中断/解码错误
            q.put(("err", e))

    threading.Thread(target=run, daemon=True).start()
    return q


def _tool_payload(ev):
    """tool_result 的 output -> 解析后的 dict(失败返回 None)"""
    out = ev.get("output")
    if isinstance(out, dict):
        text = out.get("text", "")
    elif isinstance(out, list):
        text = "".join(i.get("text", "") for i in out
                       if isinstance(i, dict) and i.get("type") == "text")
    else:
        text = out if isinstance(out, str) else ""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def render_assistant(msg):
    """历史消息回放: 与流式过程同样的静态布局"""
    with st.chat_message("assistant"):
        st.caption("✅ " + " → ".join(
            f"{s['name']}({s.get('args', '')})" for s in msg.get("steps", []))
            or "直接回答")
        tree_html = render_tree_html(msg.get("tree"), msg.get("highlight_pages"))
        if tree_html:
            st.markdown(tree_html, unsafe_allow_html=True)
        if msg.get("thinking"):
            with st.expander("🧠 模型思考过程", expanded=False):
                st.markdown(msg["thinking"])
        if msg.get("page_texts"):
            with st.expander(f"📄 已读取页面内容（{len(msg['page_texts'])} 页）",
                             expanded=False):
                st.markdown(render_pages_md(msg["page_texts"]))
        st.markdown(msg["content"])


# ---------- 会话状态 ----------
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None  # None=全库; list=多选
if "messages" not in st.session_state:
    st.session_state.messages = []

client = None   # 由 secrets/环境变量里的模型配置构建, 见下方 sidebar

# ================= 左侧边栏: 数据集管理 =================
with st.sidebar:
    st.markdown("### 🌳 Agentic RAG")
    st.caption("无向量推理 RAG · 树索引")
    st.divider()

    client = (build_client(LLM_BASE, LLM_KEY, MODEL)
              if (LLM_KEY and LLM_MODEL) else None)

    if client is None:
        st.warning("还没有配置模型服务: 请在 .streamlit/secrets.toml 里填 "
                   "llm_key 与 llm_model(或环境变量 LLM_KEY / "
                   "LLM_MODEL), 否则建立索引与问答都会失败。")

    try:
        docs = load_documents(client, st.session_state.get("pi_docs_bust", 0)) \
            if client else []
    except Exception:
        docs = []

    # 索引/删除后要 rerun 刷新列表, 提示用 session_state 传递一次
    if flash := st.session_state.pop("pi_flash", None):
        st.success(flash)

    # ---- 创建数据集: ① 上传文档 → ② 数据集目录 → ③ 建立索引 ----
    st.markdown("#### ➕ 创建数据集")

    existing_cats = sorted({
        ((d.get("metadata") or {}).get("category") or "").strip() for d in docs
    } - {""})

    # ① 上传文档
    st.markdown('<p class="pi-step"><span>1</span>上传文档</p>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("上传 PDF", type=["pdf"],
                                label_visibility="collapsed")
    if uploaded is not None:
        st.caption(f"{uploaded.name} · {uploaded.size / 1048576:.1f} MB")

    # ② 数据集目录(可选, 写入 metadata.category, 问答时可按目录整类勾选)
    st.markdown('<p class="pi-step"><span>2</span>数据集目录</p>',
                unsafe_allow_html=True)
    NEW_DIR, NO_DIR = "➕ 新建目录…", "（不设目录）"
    dir_choice = st.selectbox(
        "数据集目录", [NO_DIR, *existing_cats, NEW_DIR],
        label_visibility="collapsed",
        help="写入文档 metadata.category; 同一目录的文档会在右侧「数据集市场」里合成一张卡")
    dir_name = ""
    if dir_choice == NEW_DIR:
        dir_name = st.text_input(
            "目录名称", placeholder="如: 规章制度 / 技术文档",
            key="pi_new_dir", label_visibility="collapsed")
    cat_value = (dir_name.strip() if dir_choice == NEW_DIR
                 else "" if dir_choice == NO_DIR else dir_choice)

    # ③ 建立索引(固定 flash 模式: 布局驱动 + LLM 摘要, 快且省)
    st.markdown('<p class="pi-step"><span>3</span>建立索引</p>',
                unsafe_allow_html=True)
    mode = "flash"
    fingerprint = f"{uploaded.name}|{uploaded.size}" if uploaded is not None else ""
    indexed = bool(fingerprint) and st.session_state.get("pi_indexed") == fingerprint
    if uploaded is None:
        st.caption("先在上方选择 PDF, 再建立索引")
    elif indexed:
        st.caption("✅ 该文件已建立索引, 换一个文件可继续")
    if st.button("🚀 建立索引", type="primary", width="stretch",
                 disabled=(uploaded is None or indexed)):
        tmp_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, uploaded.name)
        if not os.path.exists(tmp_path):
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
        # 先自检模型: 上游不可用时立刻报错, 而不是转圈到天荒地老
        with st.spinner("检查模型服务…"):
            model_ok, model_why = probe_index_model()
        if not model_ok:
            st.error(
                "索引中止 · 模型服务不可用\n\n"
                f"模型 `{LLM_MODEL or '（未填）'}` @ {LLM_BASE}\n\n"
                f"{model_why}\n\n"
                "索引最后一步要调 LLM 生成摘要/树结构。请检查 .streamlit/"
                "secrets.toml 里的 llm_key 与 llm_model 是否正确、"
                "该模型是否有权限。")
        else:
            with st.spinner("索引中…（含 LLM 摘要，可能需要几分钟）"):
                try:
                    meta = {"category": cat_value} if cat_value else None
                    result = index_with_timeout(client.submit_document, tmp_path,
                                                mode=mode, metadata=meta)
                    # 范围只在主区市场勾选, 这里回到全库(新文档自动包含在内)
                    st.session_state.doc_id = None
                    st.session_state.messages = []
                    st.session_state.pi_indexed = fingerprint
                    usage = result.get("usage")
                    tip = f"索引完成 · {result.get('name', uploaded.name)}"
                    if usage:
                        tip += f" · tokens: {usage.get('total_tokens', '?')}"
                    st.session_state.pi_flash = tip
                    st.session_state.pi_docs_bust = (
                        st.session_state.get("pi_docs_bust", 0) + 1)
                    st.rerun()
                except FutureTimeout:
                    st.error(
                        f"索引超时（>{INDEX_TIMEOUT / 60:.0f} 分钟）已中止等待。\n\n"
                        "模型响应过慢或中途断开, 常见原因是上游模型不可用。"
                        "请检查 LiteLLM 代理日志后重试。")
                except Exception as e:
                    st.error(f"索引失败: {e}")

    st.divider()

    # ---- 数据集浏览: 选数据集 → 看文档清单 ----
    # 纯浏览(只读), 不参与问答范围; 范围只在主区「数据集市场」里勾选
    st.markdown("#### 📚 数据集浏览")

    if docs:
        groups = group_by_category(docs)
        by_cat = dict(groups)
        cats = [c for c, _ in groups]
        ALL_DIRS = "全部数据集"

        st.caption(f"{len(cats)} 个数据集 · {len(docs)} 篇文档")
        browse = st.selectbox(
            "数据集", [ALL_DIRS, *cats], label_visibility="collapsed",
            key="pi_browse_cat",
            format_func=lambda c: c if c == ALL_DIRS
            else f"{c}（{len(by_cat[c])} 篇）")
        shown = docs if browse == ALL_DIRS else by_cat[browse]

        st.markdown(
            '<ul class="pi-files">' + "".join(
                f'<li><span class="pi-file-n">{clean_title(d.get("name", ""))}</span>'
                f'<span class="pi-file-p">{int(d.get("pageNum") or 0)} 页</span></li>'
                for d in shown) + '</ul>',
            unsafe_allow_html=True)

        with st.expander("🗑️ 删除文档"):
            names = {d["id"]: clean_title(d.get("name", "")) for d in shown}
            drop = st.multiselect(
                "选择要删除的文档", list(names), format_func=lambda i: names[i],
                placeholder="选择要删除的文档", label_visibility="collapsed",
                key=f"pi_del_{browse}")
            if drop and st.button("确认删除", width="stretch"):
                try:
                    for i in drop:
                        client.delete_document(i)
                    st.session_state.doc_id = None
                    st.session_state.messages = []
                    st.session_state.pi_flash = f"已删除 {len(drop)} 篇文档"
                    st.session_state.pi_docs_bust = (
                        st.session_state.get("pi_docs_bust", 0) + 1)
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
    else:
        st.info("暂无文档, 在上方「创建数据集」里上传 PDF 并建立索引")


# 当前问答范围: None=全库, list=多选
scope = st.session_state.doc_id

# ================= 右侧主区: 对话 =================
# 两套布局(保证"空会话简洁 / 提问后专注"), 由 CSS 按 .pi-chat-mode 切换:
#   未提问 → 欢迎区 + 输入条作为一组垂直居中
#   已提问 → 对话区占满主区, 输入条吸底
msgs = st.session_state.messages

# 输入条: 放在脚本最前面, 让"提交 → 回答"在同一次 run 内完成
#   submit_mode="disable": 生成中输入框禁用(同 ChatGPT), 发送键灰掉;
#   停止键由页面自己画(见下方 .st-key-pi_stop), 用 rerun 打断本轮 run,
#   因此停止后界面能立刻干净地重渲染, 不会残留"思考中"的半截卡片。
# 示例问题只"填进输入框"、不直接提交: 点击后把问题写入 session_state,
# 必须在 chat_input 实例化之前写; 预填不会触发提交(chat_input 返回 None),
# 用户可以先把问题改一改再回车。
_seed = st.session_state.pop("pi_seed", None)
if _seed:
    st.session_state.pi_input = _seed
typed = st.chat_input("提问…", submit_mode="disable", key="pi_input")

# 本轮待处理的提问: 只来自输入条提交(示例问题要先落到输入框里)
pending = typed


def _on_stop():
    """停止键回调: 点击 → 触发 rerun → 本轮 run 在下个 yield 点被打断
    (回调先于新 run 的脚本体执行, 所以能在这里把已生成的部分收尾存好)
    """
    live = st.session_state.get("pi_live")
    if not live:
        return
    st.session_state.pi_live = None
    parts = live.get("answer") or ""
    st.session_state.messages.append({
        "role": "assistant",
        "content": (parts + "\n\n_（已手动停止）_" if parts
                    else "_（已手动停止，未生成内容）_"),
        "thinking": live.get("thinking") or "",
        "tree": live.get("tree"),
        "highlight_pages": live.get("hl") or [],
        "page_texts": live.get("pages") or [],
        "steps": live.get("steps") or [],
    })


# 生成中标记: 由流式开始时的 placeholder 填入, CSS 据此把"发送箭头"换成"停止键"
busy_ph = st.empty()

# ================= 右侧主区: 数据集市场(横向卡片) =================
# 每个数据集 = 一张卡: 勾选框 + 名称 + 内容描述 + 文档/页数;
# 勾选即把该数据集全部文档纳入问答范围 —— 这里是范围的唯一选择入口
# (左侧「数据集浏览」只做查阅, 不参与范围)。
if docs:
    with st.container(key="pi_market"):
        st.markdown('<p class="pi-market-title">🗂️ 数据集市场</p>',
                    unsafe_allow_html=True)
        _rev = st.session_state.get("pi_scope_rev", 0)
        _sel = set(st.session_state.doc_id or [])
        _groups = group_by_category(docs)
        for _start in range(0, len(_groups), 4):
            _chunk = _groups[_start:_start + 4]
            _cols = st.columns(len(_chunk), gap="medium")
            for _idx, (_col, (_cat, _items)) in enumerate(zip(_cols, _chunk)):
                _ids = {d["id"] for d in _items}
                _pages = sum(int(d.get("pageNum") or 0) for d in _items)
                with _col:
                    with st.container(key=f"ds_card_{_start + _idx}"):
                        _checked = st.checkbox(
                            _cat, value=_ids <= _sel,
                            key=f"mkt_{_cat}_{_rev}")
                        st.markdown(
                            f'<p class="pi-ds-desc">{dataset_blurb(_cat, _items)}</p>'
                            f'<p class="pi-ds-meta">{len(_ids)} 篇文档 · '
                            f'{_pages} 页</p>',
                            unsafe_allow_html=True)
                        if _checked and not _ids <= _sel:
                            set_scope(_sel | _ids)
                        elif not _checked and _ids <= _sel:
                            set_scope(_sel - _ids)

st.markdown("""<div class="pi-hero">
<h1>有什么可以帮你？</h1>
</div>""", unsafe_allow_html=True)

# 快捷示例: 点一下直接提问(对话态由 CSS 隐藏)
samples = ["这份文档的目录结构是怎样的？",
           "第 3 章讲了什么？",
           "用一段话总结核心结论"]
sample_cols = st.columns([0.9, 2.2, 2.2, 2.2, 0.9], gap="small")[1:-1]
for col, s in zip(sample_cols, samples):
    with col:
        if st.button(s, key=f"sample_{s}", width="stretch"):
            # 只填进输入框, 不直接提问(下一轮 run 由 _seed 预填)
            st.session_state.pi_seed = s
            st.rerun()

chat_mode = bool(msgs or pending)

# 布局标记: 出现即"对话态"(CSS 据此隐藏欢迎区、让输入条吸底)
# 欢迎区元素每轮都重建, 只靠 CSS 显隐, 避免切换布局时上一轮元素残留成鬼影
if chat_mode:
    st.markdown('<div class="pi-chat-mode"></div>', unsafe_allow_html=True)

# 对话区: 高度由 CSS(.st-key-pi_history)按视口算, autoscroll 跟到底
history = st.container(height=520, border=False, autoscroll=True,
                       key="pi_history")
with history:
    for msg in msgs:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            render_assistant(msg)

    if pending:
        st.session_state.messages.append(
            {"role": "user", "content": pending})

        with st.chat_message("user"):
            st.markdown(pending)

        with st.chat_message("assistant"):
            # 过程面板: 状态 + 树高亮 + 思考 + 页面内容 + 流式答案
            status = st.status("🧠 思考中…", expanded=True)
            tree_ph = st.empty()
            with st.expander("🧠 模型思考过程", expanded=False):
                think_ph = st.empty()
            with st.expander("📄 已读取页面内容", expanded=False):
                pages_ph = st.empty()
            answer_ph = st.empty()

            think_parts, answer_parts = [], []
            tree_structure = None
            highlight_pages = []
            page_texts = []
            steps = []
            last_think_update = 0.0

            # 供"停止"回调读取的实时快照(停止时把已生成内容收尾存进会话)
            live = {"answer": "", "thinking": "", "tree": None,
                    "hl": [], "pages": [], "steps": []}
            st.session_state.pi_live = live

            def _snap():
                live["answer"] = "".join(answer_parts)
                live["thinking"] = "".join(think_parts)
                live["tree"] = tree_structure
                live["hl"] = list(highlight_pages)
                live["pages"] = list(page_texts)
                live["steps"] = list(steps)

            try:
                if not (LLM_KEY and LLM_MODEL):
                    raise RuntimeError(
                        "未配置模型服务: 请在 .streamlit/secrets.toml 填入 "
                        "llm_key 与 llm_model")
                stream = client.chat(
                    pending, doc_id=scope, stream=True,
                )
                busy_ph.markdown('<div class="pi-busy"></div>',
                                 unsafe_allow_html=True)
                # 后端事件流放在独立线程里读, 主线程按 0.5s 轮询:
                # Streamlit 只在脚本"产出新消息"时才检查停止/重跑请求, 若后端卡住
                # (代理不吐 token) 主线程会一直阻塞, 那时点"停止"不会生效。
                # 有了心跳, 即使一帧事件都没有, 也能在 0.5s 内响应停止。
                q = _pump_events(stream)
                t_start = time.time()
                while True:
                    try:
                        kind, payload = q.get(timeout=0.5)
                    except queue.Empty:
                        _snap()
                        status.update(
                            label=f"🧠 思考中… {int(time.time() - t_start)}s")
                        continue
                    if kind == "done":
                        break
                    if kind == "err":
                        raise payload
                    ev = payload
                    t = ev["type"]
                    if t == "thinking":
                        think_parts.append(ev["delta"])
                        # 节流: 每 0.4s 刷一次思考区
                        if time.time() - last_think_update > 0.4:
                            think_ph.markdown(
                                "".join(think_parts) + " ▌")
                            last_think_update = time.time()
                            _snap()
                    elif t == "tool_call":
                        name = ev["name"]
                        args = ev.get("arguments")
                        args_str = ""
                        if isinstance(args, dict):
                            if name == "get_page_content":
                                args_str = f"pages {args.get('pages', '')}"
                            elif name == "get_document_structure":
                                args_str = str(args.get("doc_name", ""))
                            elif name == "browse_documents":
                                args_str = "浏览文档库"
                        steps.append({"name": name, "args": args_str})
                        status.update(label=f"🔧 {name}({args_str})")
                        if name == "get_page_content" and isinstance(args, dict):
                            # 调用即高亮: 模型决定读哪些页
                            highlight_pages = sorted(
                                set(highlight_pages)
                                | set(parse_pages(args.get("pages"))))
                            tree_ph.markdown(
                                render_tree_html(tree_structure,
                                                 highlight_pages),
                                unsafe_allow_html=True)
                    elif t == "tool_result":
                        name = ev["name"]
                        payload = _tool_payload(ev)
                        if name == "get_document_structure" and payload:
                            tree_structure = payload.get("structure")
                            tree_ph.markdown(
                                render_tree_html(tree_structure,
                                                 highlight_pages),
                                unsafe_allow_html=True)
                            status.update(
                                label="🌳 已获取文档结构, 定位相关章节…")
                        elif name == "get_page_content" and payload:
                            got = parse_pages(
                                payload.get("returned_pages")
                                or payload.get("requested_pages"))
                            highlight_pages = sorted(
                                set(highlight_pages) | set(got))
                            tree_ph.markdown(
                                render_tree_html(tree_structure,
                                                 highlight_pages),
                                unsafe_allow_html=True)
                            for item in payload.get("content", []):
                                page_texts.append(
                                    (item.get("page"), item.get("text", "")))
                            pages_ph.markdown(render_pages_md(page_texts))
                            status.update(
                                label=f"📄 已读取 {len(got)} 页, 生成答案…")
                    elif t == "answer":
                        answer_parts.append(ev["delta"])
                        answer_ph.markdown("".join(answer_parts) + " ▌")
                        _snap()

                think_ph.markdown("".join(think_parts) or "（无思考输出）")
                answer_ph.markdown("".join(answer_parts))
                status.update(label="✅ 完成", state="complete",
                              expanded=False)
                st.session_state.pi_live = None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "".join(answer_parts),
                    "thinking": "".join(think_parts),
                    "tree": tree_structure,
                    "highlight_pages": highlight_pages,
                    "page_texts": page_texts,
                    "steps": steps,
                })
            except Exception as e:
                st.session_state.pi_live = None
                status.update(label=f"❌ 出错: {e}", state="error")
                answer_ph.markdown("".join(answer_parts) or "（无输出）")

# 底部: 免责提示 + 停止键(绝对定位到输入框右侧内缘, 参考 ChatGPT/Grok)
with st.bottom:
    st.caption("⚠️ 模型输出可能不完整或有误, 请对照原文核验"
               " · 基于 VectifyAI/PageIndex")
    # 图标由 CSS 画(见 .st-key-pi_stop button::before), 不用 Material 连字:
    # 字体图标在本机是一坨虚边小方块, 且依赖外部字体, 加载不到就露出 "stop" 字样
    st.button("停止生成", key="pi_stop",
              on_click=_on_stop, width="content")

