#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TypoAgent Web Tool: 支持上传文本构建树 + 聊天式需求澄清
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 项目根目录
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# 会话存储: session_id -> { conversation_history, tree_path }
_sessions: dict = {}
SESSIONS_DIR = _REPO_ROOT / "output" / "tool" / "sessions"


def _load_sessions():
    """从磁盘加载会话，供 TypoAgent 根据聊天历史进行树排序"""
    global _sessions
    if not SESSIONS_DIR.exists():
        return
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sid = f.stem
            _sessions[sid] = {
                "conversation_history": data.get("conversation_history", []),
                "tree_path": data.get("tree_path", ""),
            }
        except Exception:
            pass


def _save_session(session_id: str, conv: list, tree_path: str):
    """将会话持久化到磁盘"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{session_id}.json"
    try:
        path.write_text(
            json.dumps({"conversation_history": conv, "tree_path": tree_path}, ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
    except Exception:
        pass


_load_sessions()

# 默认树路径（用于聊天）
DEFAULT_TREE_PATH = str(_REPO_ROOT / "output" / "save_tree" / "LLMTree.auto-p.json")
FALLBACK_TREE = str(_REPO_ROOT / "output" / "save_tree" / "Typo_Tree.json")


def _ensure_tree_exists() -> str:
    """确保存在可用的树文件"""
    for p in [DEFAULT_TREE_PATH, FALLBACK_TREE]:
        if Path(p).exists():
            return p
    return DEFAULT_TREE_PATH


def _parse_upload_to_jsonl(content: str, filename: str) -> str:
    """
    将上传内容解析为 JSONL 格式。
    支持：1) 每行一个 JSON 对象；2) 每行纯文本作为 instruction；3) 多行文本块
    """
    lines = content.strip().split("\n")
    rows = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # 尝试解析为 JSON
        try:
            obj = json.loads(line)
            if "instruction" not in obj:
                obj["instruction"] = line
            if "application_type" not in obj:
                obj["application_type"] = "General"
            if "id" not in obj:
                obj["id"] = f"upload_{i+1}"
            rows.append(obj)
        except json.JSONDecodeError:
            rows.append({
                "id": f"upload_{i+1}",
                "instruction": line,
                "application_type": "General",
            })
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/build-tree", methods=["POST"])
def build_tree():
    """上传文本/JSONL 构建 Typo Tree"""
    if "file" not in request.files and "content" not in request.form:
        return jsonify({"error": "请上传文件或提供 content 文本"}), 400

    content = None
    if "file" in request.files:
        f = request.files["file"]
        if f.filename:
            content = f.read().decode("utf-8", errors="replace")
    if "content" in request.form:
        content = request.form["content"]

    if not content or not content.strip():
        return jsonify({"error": "内容为空"}), 400

    try:
        jsonl_content = _parse_upload_to_jsonl(content, "upload")
    except Exception as e:
        return jsonify({"error": f"解析失败: {e}"}), 400

    # 可选参数
    use_layer2 = request.form.get("use_layer2", "false").lower() == "true"
    use_layer3 = request.form.get("use_layer3", "true").lower() != "false"
    enable_sampling = request.form.get("enable_sampling", "false").lower() == "true"
    tree_name = (request.form.get("tree_name") or "").strip()
    # 树文件名：仅允许字母数字、下划线、连字符、点号
    if tree_name:
        tree_name = re.sub(r"[^\w.-]", "", tree_name)
        tree_name = tree_name or "Typo_Tree"
    else:
        tree_name = "Typo_Tree"
    if not tree_name.endswith(".json"):
        tree_name += ".json"

    save_dir = _REPO_ROOT / "output" / "save_tree" / "tool"
    save_dir.mkdir(parents=True, exist_ok=True)

    input_path = str(save_dir / "upload_input.jsonl")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(input_path).write_text(jsonl_content, encoding="utf-8")

    sampled_path = str(save_dir / "train_new.jsonl")
    if not enable_sampling:
        Path(sampled_path).write_text(jsonl_content, encoding="utf-8")

    try:
        tree_output = str(save_dir / tree_name)
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "run_typobuilder.py"),
            "--input", input_path,
            "--sampled-output", sampled_path,
            "--save-dir", str(save_dir),
            "--tree-output", tree_output,
        ]
        if use_layer2:
            cmd.append("--use-layer2")
        if enable_sampling:
            cmd.append("--enable-sampling")
        if not use_layer3:
            cmd.extend(["--no-use-layer3", "--no-clustering"])

        env = os.environ.copy()
        api_key = request.form.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            env["OPENAI_API_KEY"] = api_key

        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if proc.returncode != 0:
            return jsonify({
                "error": "构建失败",
                "stderr": proc.stderr or "",
                "stdout": proc.stdout or "",
            }), 500

        if not Path(tree_output).exists():
            return jsonify({"error": "树文件未生成，请检查输出"}), 500

        return jsonify({
            "success": True,
            "tree_path": tree_output,
            "message": "树构建完成",
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "构建超时（超过 10 分钟）"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """TypoAgent 聊天：用户提问/回答，TypoAgent 返回下一个问题"""
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id")
    tree_path = data.get("tree_path") or _ensure_tree_exists()  # 支持指定树

    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400

    api_key = data.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "请设置 OPENAI_API_KEY 或传入 api_key"}), 400

    base_url = data.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    model_name = data.get("model_name", "gpt-4o")

    if not Path(tree_path).exists():
        tree_path = _ensure_tree_exists()

    # 获取或创建会话；服务重启后可从请求中的 conversation_history 恢复上下文（用于树排序）
    raw_history = data.get("conversation_history") or []
    if session_id and session_id in _sessions:
        sess = _sessions[session_id]
        conv = sess["conversation_history"]
    else:
        session_id = session_id or str(uuid.uuid4())
        conv = [{"role": h.get("role", "user"), "content": (h.get("content") or "").strip()} for h in raw_history if (h.get("content") or "").strip()]
        _sessions[session_id] = {"conversation_history": conv, "tree_path": tree_path}

    # 追加用户消息
    conv.append({"role": "user", "content": user_message})

    try:
        from TypoAgent.retriever import TypoAgentInterviewer

        interviewer = TypoAgentInterviewer(
            api_key=api_key,
            model_name=model_name,
            temperature=0.0,
            max_tokens=2048,
            timeout=60.0,
            base_url=base_url
            if base_url
            else "https://api.chatanywhere.tech/v1",
            fixed_tree_path=tree_path,
            tree_percentage=100.0,
        )

        response = interviewer.ask_question(conversation_history=conv)

        if response:
            conv.append({"role": "interviewer", "content": response})

        _save_session(session_id, conv, tree_path)

        return jsonify({
            "session_id": session_id,
            "response": response or "",
            "finished": _is_finish(response),
        })
    except Exception as e:
        conv.pop()  # 回滚用户消息
        return jsonify({"error": str(e)}), 500


def _is_finish(text: Optional[str]) -> bool:
    """判断是否为结束语"""
    if not text:
        return True
    t = (text or "").strip().lower()
    return "user requirements list" in t or "i have gathered enough" in t or "需求列表" in t


@app.route("/api/chat/init", methods=["POST"])
def chat_init():
    """初始化聊天会话，获取 TypoAgent 的第一个问题"""
    data = request.get_json() or {}
    initial_req = (data.get("initial_requirements") or data.get("message") or "").strip()
    tree_path = data.get("tree_path") or _ensure_tree_exists()

    if not initial_req:
        return jsonify({"error": "请提供初始需求描述"}), 400

    api_key = data.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "请设置 OPENAI_API_KEY 或传入 api_key"}), 400

    base_url = data.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    model_name = data.get("model_name", "gpt-4o")

    if not Path(tree_path).exists():
        tree_path = _ensure_tree_exists()

    session_id = str(uuid.uuid4())
    conv = [{"role": "user", "content": initial_req}]
    _sessions[session_id] = {"conversation_history": conv, "tree_path": tree_path}

    try:
        from TypoAgent.retriever import TypoAgentInterviewer

        interviewer = TypoAgentInterviewer(
            api_key=api_key,
            model_name=model_name,
            temperature=0.0,
            max_tokens=2048,
            timeout=60.0,
            base_url=base_url
            if base_url
            else "https://api.chatanywhere.tech/v1",
            fixed_tree_path=tree_path,
            tree_percentage=100.0,
        )

        response = interviewer.ask_question(conversation_history=conv)

        if response:
            conv.append({"role": "interviewer", "content": response})

        _save_session(session_id, conv, tree_path)

        return jsonify({
            "session_id": session_id,
            "response": response or "",
            "finished": _is_finish(response),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _normalize_base_url(url: Optional[str]) -> Optional[str]:
    if not url or not str(url).strip():
        return None
    url = str(url).strip().rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


@app.route("/api/test-api", methods=["POST"])
def test_api():
    """Test if API key and base_url are valid"""
    data = request.get_json() or {}
    api_key = (data.get("api_key") or "").strip()
    base_url = _normalize_base_url(data.get("base_url"))

    if not api_key:
        return jsonify({"ok": False, "error": "API key is required"}), 400

    try:
        from openai import OpenAI

        kw = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        client = OpenAI(**kw)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=10,
            timeout=15,
        )
        text = (resp.choices[0].message.content or "").strip()
        return jsonify({"ok": True, "message": "Connection successful", "response": text[:50]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200


DEFAULT_TREE_DIR = str(_REPO_ROOT / "output" / "save_tree")


@app.route("/api/trees", methods=["GET"])
def list_trees():
    """列出可用的树文件。tree_dir: default | tool，默认 default = output/save_tree"""
    tree_dir = request.args.get("tree_dir", "default").strip().lower()
    base = Path(DEFAULT_TREE_DIR)
    if tree_dir == "tool":
        search_dirs = [base / "tool", base]
    else:
        search_dirs = [base, base / "tool"]
    trees = []
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            if "Typo_Tree" in f.name or "LLMTree" in f.name:
                trees.append({"path": str(f), "name": f.name})
    if not trees:
        trees.append({"path": DEFAULT_TREE_PATH, "name": "未找到树文件"})
    return jsonify({"trees": trees})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
