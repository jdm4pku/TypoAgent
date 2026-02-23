# TypoAgent/typo_builder.py
# 构建需求澄清树。prompts 从 TypoAgent/prompt/*.txt 加载，输出保存至 output/save_tree。
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# 本地 GPU 聚类可选依赖
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False

_CUR = Path(__file__).resolve().parent
_DATA_DIR = _CUR / "data"
_PROMPT_DIR = _CUR / "prompt" / "builder"
# 构建树输出目录：TypoAgent_release/output/save_tree
_SAVE_TREE_DIR = _CUR.parent / "output" / "save_tree"

ROOT_LAYERS = ["Interaction", "Content", "Style"]


def _load_prompt(name: str) -> str:
    """从 prompt 目录加载 txt 文件内容。"""
    path = _PROMPT_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


# -------------------- prompts 缓存 --------------------
_prompt_cache: dict[str, str] = {}


def _get_prompt(name: str) -> str:
    if name not in _prompt_cache:
        _prompt_cache[name] = _load_prompt(name)
    return _prompt_cache[name]


def load_reference_skeleton(ref_path: str | Path | None = None) -> dict[str, list[str]]:
    """从参考树只读取框架：每个 layer 下的 subcategory 名称列表。"""
    path = Path(ref_path or _DATA_DIR / "LLMTree_struction.json")
    if not path.exists():
        return {}
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    skeleton = {}
    for layer in ROOT_LAYERS:
        if layer not in tree or not isinstance(tree[layer], dict):
            skeleton[layer] = []
            continue
        skeleton[layer] = sorted(tree[layer].keys())
    return skeleton


def format_skeleton_for_prompt(skeleton: dict[str, list[str]]) -> str:
    """把骨架格式化为 prompt 中的允许子类列表。"""
    parts = []
    for layer in ROOT_LAYERS:
        subs = skeleton.get(layer, [])
        if subs:
            parts.append(f"{layer}: {', '.join(subs)}")
    return "; ".join(parts)


def read_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_write_text(path: Path, content: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception:
        pass


# -------------------- JSON 解析（不依赖 ReqElicitGym）--------------------
def _extract_json_candidate(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if "{" in text and "}" in text:
        s = text.find("{")
        e = text.rfind("}")
        if 0 <= s < e:
            return text[s : e + 1].strip()
    return text


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    cand = _extract_json_candidate(text)
    # 尝试 ```json ... ``` 块
    m = re.search(r"```json\s*(.*?)\s*```", cand, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    try:
        return json.loads(cand)
    except Exception:
        return {}


# -------------------- OpenAI helpers --------------------
def _resolve_api_key(api_key: str | None) -> str:
    if api_key and api_key != "null":
        return api_key
    env_key = os.getenv("OPENAI_API_KEY", "")
    if not env_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Set env var or pass api_key to main().")
    return env_key


def _normalize_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _resolve_base_url(base_url: str | None) -> str:
    env_url = os.getenv("OPENAI_BASE_URL")
    return _normalize_base_url(base_url or env_url or "https://api.chatanywhere.tech/v1")


def chat_call(
    system_prompt: str,
    user_prompt: str,
    model_config: dict,
    return_json: bool = True,
    call_tag: str = "call",
    debug_id: str = "unknown",
    debug_dir: str | None = None,
):
    api_key = _resolve_api_key(model_config.get("api_key"))
    base_url = _resolve_base_url(model_config.get("base_url"))
    timeout = model_config.get("timeout", 300.0)
    debug = bool(model_config.get("debug", False))
    debug_dir = debug_dir or str(_SAVE_TREE_DIR / "llm_debug")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    for t in range(3):
        try:
            if debug:
                print("\n[chat_call]", {"tag": call_tag, "id": debug_id})
            resp = client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                temperature=model_config.get("temperature", 0.2),
                max_tokens=model_config.get("max_tokens", 2048),
                timeout=timeout,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not return_json:
                return text
            parsed = _parse_json(text)
            if not parsed and debug:
                _safe_write_text(Path(debug_dir) / f"{debug_id}.{call_tag}.json_candidate.txt", _extract_json_candidate(text))
            return parsed
        except Exception as e:
            print(f"[typo_builder.chat_call] Error({call_tag}, try={t+1}/3): {e}")
            if t == 2:
                return {} if return_json else ""
            time.sleep(2)
    return {} if return_json else ""


def embed_texts(texts: list[str], api_key=None, base_url=None, model="text-embedding-3-small", timeout=300.0):
    client = OpenAI(api_key=_resolve_api_key(api_key), base_url=_resolve_base_url(base_url), timeout=timeout)
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


_local_embedder: Any = None


def embed_texts_local(texts, model_name="sentence-transformers/all-mpnet-base-v2", device=None, batch_size=32):
    global _local_embedder
    if not _HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError("sentence-transformers 未安装")
    if _local_embedder is None:
        _local_embedder = SentenceTransformer(model_name, device=device or "cuda")
    vecs = _local_embedder.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def _get_embeddings(texts: list[str], model_config: dict) -> np.ndarray:
    if model_config.get("use_local_embedding") and _HAS_SENTENCE_TRANSFORMERS:
        return embed_texts_local(
            texts,
            model_name=model_config.get("local_embedding_model", "sentence-transformers/all-mpnet-base-v2"),
            device=model_config.get("embedding_device"),
            batch_size=model_config.get("local_embedding_batch_size", 32),
        )
    return embed_texts(
        texts,
        api_key=None if model_config.get("api_key") == "null" else model_config.get("api_key"),
        base_url=model_config.get("base_url"),
        model=model_config.get("embedding_model", "text-embedding-3-small"),
        timeout=model_config.get("timeout", 300.0),
    )


# -------------------- normalization --------------------
_snake_re = re.compile(r"[^a-z0-9_]+")
_word_split_re = re.compile(r"[^\w]+")


def normalize_key(key: str) -> str:
    k = (key or "").strip().lower().replace("-", "_").replace(" ", "_")
    k = _snake_re.sub("_", k)
    k = re.sub(r"_+", "_", k).strip("_")
    return k


def to_camel_case(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    parts = [p for p in _word_split_re.split(s.replace("_", " ")) if p]
    if not parts:
        return ""
    return "".join(p[:1].upper() + p[1:] for p in parts)


MAX_SUBCATEGORY_NAME_LEN = 50


def normalize_subcategory(sub: str) -> str:
    sub = (sub or "").strip()
    if not sub:
        return ""
    sub = re.sub(r"\s+", " ", sub)
    return to_camel_case(sub)


def choose_better_question(q1: str, q2: str) -> str:
    q1, q2 = (q1 or "").strip(), (q2 or "").strip()
    if not q1:
        return q2
    if not q2:
        return q1
    return q1 if len(q1) <= len(q2) else q2


def node_ok(n: dict) -> tuple[bool, list[str]]:
    reasons = []
    if n.get("layer") not in ROOT_LAYERS:
        reasons.append(f"layer_invalid({n.get('layer')})")
    for f in ["subcategory", "key", "question"]:
        if not n.get(f):
            reasons.append(f"{f}_missing")
    return (len(reasons) == 0, reasons)


# -------------------- LLM 抽取（使用 txt prompts）--------------------
def llm_extract_layer2_core_features(instruction: str, model_config: dict, debug_id: str, current_file_content: str = ""):
    sys_prompt = _get_prompt("layer2_system")
    user_tmpl = _get_prompt("layer2_user")

    _empty_tree = json.dumps({layer: [] for layer in ROOT_LAYERS}, ensure_ascii=False, indent=2)
    user_prompt = user_tmpl.format(
        instruction=instruction,
        current_file_content=current_file_content or _empty_tree,
    )
    data = chat_call(sys_prompt, user_prompt, model_config, return_json=True, call_tag="layer2_extract", debug_id=debug_id, debug_dir=str(model_config.get("debug_dir", _SAVE_TREE_DIR / "llm_debug")))
    data = data or {}
    expand = data.get("expand_existing", [])
    features = data.get("core_features", [])
    return (expand if isinstance(expand, list) else [], features if isinstance(features, list) else [])


def skeleton_to_framework_tree(skeleton: dict[str, list[str]]) -> dict[str, dict[str, dict[str, str]]]:
    tree = {layer: {} for layer in ROOT_LAYERS}
    for layer in ROOT_LAYERS:
        for sub in skeleton.get(layer, []):
            sub_norm = normalize_subcategory(sub)
            if sub_norm:
                tree[layer].setdefault(sub_norm, {})
    return tree


def write_skeleton_to_path(skeleton: dict[str, list[str]], path: str | Path) -> None:
    data = {layer: {sub: {} for sub in skeleton.get(layer, [])} for layer in ROOT_LAYERS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_layer3_skeleton_text(skeleton: dict[str, list[str]]) -> str:
    lines = []
    for layer in ROOT_LAYERS:
        subs = skeleton.get(layer, [])
        if subs:
            lines.append(f"- {layer}: " + ", ".join(subs))
    return "\n".join(lines) if lines else ""


def llm_extract_layer3_details(instruction: str, skeleton: dict[str, list[str]], model_config: dict, debug_id: str):
    skeleton_text = _format_layer3_skeleton_text(skeleton)
    sys_prompt = _get_prompt("layer3_system")
    user_tmpl = _get_prompt("layer3_user")

    user_prompt = user_tmpl.format(
        instruction=instruction,
        skeleton_text=skeleton_text or "(no Layer2 nodes yet)",
    )
    data = chat_call(sys_prompt, user_prompt, model_config, return_json=True, call_tag="layer3_extract", debug_id=debug_id, debug_dir=str(model_config.get("debug_dir", _SAVE_TREE_DIR / "llm_debug")))
    return (data or {}).get("relevant_dimensions", [])


def _format_canon_user_prompt(items: str, skeleton: dict[str, list[str]] | None = None) -> str:
    allowed = ""
    if skeleton and any(skeleton.get(l) for l in ROOT_LAYERS):
        allowed = "- subcategory MUST be exactly one of: " + format_skeleton_for_prompt(skeleton) + ".\n"
    return _get_prompt("canon_user").format(items=items, allowed_subcategories_constraint=allowed)


def llm_canonicalize_cluster(cluster_nodes: list, model_config: dict, debug_id: str, skeleton: dict | None = None):
    items = json.dumps(cluster_nodes[:30], ensure_ascii=False, indent=2)
    user_prompt = _format_canon_user_prompt(items, skeleton=skeleton)
    data = chat_call(_get_prompt("canon_system"), user_prompt, model_config, return_json=True, call_tag="canon", debug_id=debug_id, debug_dir=str(model_config.get("debug_dir", _SAVE_TREE_DIR / "llm_debug")))
    return (data or {}).get("node", {})


CANON_SUBCAT_MAX_ITEMS = 80


def llm_canonicalize_subcategory(nodes: list, layer: str, sub: str, model_config: dict, debug_id: str):
    if not nodes:
        return {}
    use_nodes = nodes[:CANON_SUBCAT_MAX_ITEMS]
    items = json.dumps([{"key": n.get("key", ""), "question": n.get("question", "")} for n in use_nodes], ensure_ascii=False, indent=2)
    user_tmpl = _get_prompt("canon_subcat_user_tmpl")
    user_prompt = user_tmpl.format(layer=layer, subcategory=sub, items=items)
    data = chat_call(_get_prompt("canon_subcat_system"), user_prompt, model_config, return_json=True, call_tag="canon_subcat", debug_id=debug_id, debug_dir=str(model_config.get("debug_dir", _SAVE_TREE_DIR / "llm_debug")))
    raw_nodes = (data or {}).get("nodes", [])
    merged = {}
    for n in raw_nodes:
        key = normalize_key(n.get("key", ""))
        q = (n.get("question") or "").strip()
        if key and q:
            existing = merged.get(key)
            merged[key] = choose_better_question(existing or "", q)
    return merged


# -------------------- clustering --------------------
def cluster_by_embedding_texts(texts: list[str], model_config: dict, embedding_model: str, distance_threshold: float, tag: str):
    if not texts:
        return {}
    if len(texts) == 1:
        return {0: [0]}
    vecs = _get_embeddings(texts, model_config)
    use_local = model_config.get("use_local_embedding") and _HAS_SENTENCE_TRANSFORMERS and _HAS_HDBSCAN
    if use_local:
        eps = float(np.sqrt(2.0 * distance_threshold))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=eps, metric="euclidean", cluster_selection_method="eom")
        labels = clusterer.fit_predict(vecs)
        clusters = defaultdict(list)
        next_id = 0
        for i, lb in enumerate(labels):
            if lb == -1:
                clusters[next_id].append(i)
                next_id += 1
            else:
                clusters[int(lb)].append(i)
    else:
        clustering = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance_threshold)
        labels = clustering.fit_predict(vecs)
        clusters = defaultdict(list)
        for i, lb in enumerate(labels):
            clusters[int(lb)].append(i)
    return dict(clusters)


def cluster_nodes(nodes: list, model_config: dict, embedding_model: str, distance_threshold: float, tag: str):
    texts = [f'{n.get("layer","")} | {n.get("subcategory","")} | {n.get("key","")} | {n.get("question","")}' for n in nodes]
    idx_clusters = cluster_by_embedding_texts(texts, model_config, embedding_model, distance_threshold, tag)
    return {cid: [nodes[i] for i in idxs] for cid, idxs in idx_clusters.items()}


def tree_find_key(tree: dict, layer: str, key: str) -> tuple:
    if layer not in tree:
        return (None, None)
    for sub, kv in (tree[layer] or {}).items():
        if key in kv:
            return (sub, kv[key])
    return (None, None)


def tree_upsert_node(tree: dict, node: dict, debug: bool = False, unique_scope: str = "layer"):
    layer = node.get("layer")
    sub = normalize_subcategory(node.get("subcategory", ""))
    key = normalize_key(node.get("key", ""))
    q = (node.get("question") or "").strip()
    if layer not in ROOT_LAYERS or not sub or not key or not q:
        return {"action": "drop", "reason": "invalid_fields", "node": node}
    if unique_scope not in ("layer", "subcategory"):
        unique_scope = "layer"
    if unique_scope == "layer":
        exist_sub, exist_q = tree_find_key(tree, layer, key)
        if exist_sub is not None:
            best_q = choose_better_question(exist_q, q)
            tree[layer][exist_sub][key] = best_q
            return {"action": "update", "layer": layer, "key": key, "subcategory": exist_sub, "question_old": exist_q, "question_new": best_q}
    else:
        tree.setdefault(layer, {})
        tree[layer].setdefault(sub, {})
        exist_q = tree[layer][sub].get(key)
        if exist_q is not None:
            best_q = choose_better_question(exist_q, q)
            tree[layer][sub][key] = best_q
            return {"action": "update", "layer": layer, "key": key, "subcategory": sub, "question_old": exist_q, "question_new": best_q}
    tree[layer].setdefault(sub, {})
    tree[layer][sub][key] = q
    return {"action": "insert", "layer": layer, "subcategory": sub, "key": key}


def sort_tree(tree: dict):
    for layer in tree:
        for sub in list(tree[layer].keys()):
            tree[layer][sub] = dict(sorted(tree[layer][sub].items(), key=lambda kv: kv[0]))
        tree[layer] = dict(sorted(tree[layer].items(), key=lambda kv: kv[0]))
    return tree


def load_framework_tree(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {layer: {} for layer in ROOT_LAYERS}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {layer: {} for layer in ROOT_LAYERS}
    tree = {layer: {} for layer in ROOT_LAYERS}
    for layer in ROOT_LAYERS:
        subs = data.get(layer, {})
        if isinstance(subs, dict):
            for sub in subs.keys():
                sub_norm = normalize_subcategory(sub)
                if sub_norm:
                    tree[layer].setdefault(sub_norm, {})
    return tree


def cluster_within_subcategories(tree: dict, model_config: dict, embedding_model: str, distance_threshold: float, use_llm_only_canonicalize: bool = False, debug: bool = False):
    out = {layer: {} for layer in ROOT_LAYERS}
    freq_out = {layer: {} for layer in ROOT_LAYERS}
    for layer in ROOT_LAYERS:
        layer_subs = list((tree.get(layer, {}) or {}).items())
        if not layer_subs:
            continue
        pbar = tqdm(layer_subs, desc=f"[2/2] 子类规整-" + layer, unit="子类")
        for sub, kv in pbar:
            if not kv:
                out[layer][sub] = {}
                freq_out[layer][sub] = {}
                continue
            nodes = [{"layer": layer, "subcategory": sub, "key": k, "question": q} for k, q in kv.items()]
            if use_llm_only_canonicalize:
                merged = llm_canonicalize_subcategory(nodes, layer=layer, sub=sub, model_config=model_config, debug_id=f"canon_subcat.{layer}.{sub}")
                merged_freq = {k: 1 for k in merged}
            else:
                clusters = cluster_nodes(nodes, model_config, embedding_model, distance_threshold, f"final_{layer}_{sub}")
                merged = {}
                merged_freq = {}
                for cid, cluster_nodes_ in clusters.items():
                    canon = llm_canonicalize_cluster(cluster_nodes_[:30], model_config, f"final.{layer}.{sub}.cluster_{cid}", skeleton=None)
                    ok, _ = node_ok(canon)
                    if not ok:
                        continue
                    key = normalize_key(canon.get("key", ""))
                    q = (canon.get("question") or "").strip()
                    if not key or not q:
                        continue
                    existing = merged.get(key)
                    merged[key] = choose_better_question(existing or "", q)
                    merged_freq[key] = merged_freq.get(key, 0) + len(cluster_nodes_)
            out[layer][sub] = dict(sorted(merged.items(), key=lambda x: x[0]))
            freq_out[layer][sub] = merged_freq
        pbar.close()
    return sort_tree(out), freq_out


# -------------------- main --------------------
def build_typo_tree(
    train_path: str | None = None,
    out_path: str | None = None,
    framework_path: str | None = None,
    layer2_output_path: str | Path | None = None,
    pre_cluster_output_path: str | Path | None = None,
    use_layer2_extraction: bool = False,
    use_layer3_extraction: bool = True,
    do_clustering: bool = True,
    api_key: str = "null",
    base_url: str | None = None,
    model_name: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: float = 300.0,
    embedding_model: str = "text-embedding-3-small",
    use_local_embedding: bool = True,
    local_embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    embedding_device: str | None = None,
    final_within_sub_threshold: float = 0.38,
):
    """
    构建 Typo Tree。输入 train_new.jsonl，输出 Typo_Tree 保存至 output/save_tree。
    """
    _train_path = train_path or str(_DATA_DIR / "train_new.jsonl")
    _out_path = out_path or str(_SAVE_TREE_DIR / "Typo_Tree.json")
    _framework_path = framework_path or str(_DATA_DIR / "LLMTree_struction.json")
    _layer2_output_path = str(layer2_output_path) if layer2_output_path else ""
    _pre_cluster_path = Path(pre_cluster_output_path) if pre_cluster_output_path else (_SAVE_TREE_DIR / "Typo_Tree_incremental.json")

    _SAVE_TREE_DIR.mkdir(parents=True, exist_ok=True)

    model_config = {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "debug": False,
        "debug_dir": str(_SAVE_TREE_DIR / "llm_debug"),
        "embedding_model": embedding_model,
        "use_local_embedding": use_local_embedding,
        "local_embedding_model": local_embedding_model,
        "embedding_device": embedding_device,
    }

    print("[配置] use_layer2_extraction =", use_layer2_extraction)
    print("[配置] use_layer3_extraction =", use_layer3_extraction)
    print("[配置] 输出目录 =", _SAVE_TREE_DIR)
    print("[配置] 最终输出 =", _out_path)

    # 步骤 1：Layer2
    _layer2_path = Path(_layer2_output_path) if _layer2_output_path else None
    run_layer2 = use_layer2_extraction and (_layer2_path is None or not _layer2_path.exists())
    if use_layer2_extraction and not run_layer2 and _layer2_path and _layer2_path.exists():
        print(f"[Layer2] 跳过：目标文件已存在 {_layer2_path}")

    if run_layer2:
        skeleton_layer2 = {layer: set() for layer in ROOT_LAYERS}
        with open(_train_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())
        pbar_l2 = tqdm(total=total_lines or 1, desc="[Layer2] 从 instruction 提炼核心功能点", unit="条")
        for i, row in enumerate(read_jsonl(_train_path)):
            pbar_l2.update(1)
            rid = row.get("id", f"row_{i}")
            instr = row.get("instruction", "") or ""
            current_skeleton = {layer: sorted(skeleton_layer2[layer]) for layer in ROOT_LAYERS}
            current_file_content = json.dumps(current_skeleton, ensure_ascii=False, indent=2)
            expand_existing, features = llm_extract_layer2_core_features(instr, model_config, rid, current_file_content)
            for item in expand_existing:
                layer = item.get("layer")
                current = normalize_subcategory(item.get("current", ""))
                suggested = normalize_subcategory(item.get("suggested", ""))
                if layer in ROOT_LAYERS and current and suggested and current != suggested and len(suggested) <= MAX_SUBCATEGORY_NAME_LEN:
                    if current in skeleton_layer2[layer]:
                        skeleton_layer2[layer].discard(current)
                        skeleton_layer2[layer].add(suggested)
            for f in features:
                layer = f.get("layer")
                sub = normalize_subcategory(f.get("subcategory", ""))
                if layer in ROOT_LAYERS and sub and len(sub) <= MAX_SUBCATEGORY_NAME_LEN and sub not in skeleton_layer2[layer]:
                    skeleton_layer2[layer].add(sub)
            if _layer2_output_path:
                skeleton = {layer: sorted(skeleton_layer2[layer]) for layer in ROOT_LAYERS}
                write_skeleton_to_path(skeleton, _layer2_output_path)
        pbar_l2.close()
        skeleton = {layer: sorted(skeleton_layer2[layer]) for layer in ROOT_LAYERS}
        tree = skeleton_to_framework_tree(skeleton)
        if _layer2_output_path:
            write_skeleton_to_path(skeleton, _layer2_output_path)
        _framework_path = _layer2_output_path
    else:
        tree = load_framework_tree(_framework_path)
        skeleton = {layer: sorted(tree[layer].keys()) for layer in ROOT_LAYERS}

    # 步骤 2：Layer3
    if not use_layer3_extraction:
        tree = sort_tree(tree)
        Path(_out_path).write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[直接结束] 已写入: {_out_path}")
        return

    run_layer3 = not _pre_cluster_path.exists()
    if not run_layer3:
        print(f"[Layer3] 跳过：目标文件已存在 {_pre_cluster_path}")
        tree = json.loads(_pre_cluster_path.read_text(encoding="utf-8"))
    else:
        tree = load_framework_tree(_framework_path)
        skeleton = {layer: sorted(tree[layer].keys()) for layer in ROOT_LAYERS}
        with open(_train_path, "r", encoding="utf-8") as f:
            total_extract = sum(1 for line in f if line.strip())
        pbar = tqdm(total=total_extract, desc="[1/2] 增量构建", unit="条")
        for i, row in enumerate(read_jsonl(_train_path)):
            pbar.update(1)
            rid = row.get("id", f"row_{i}")
            instr = row.get("instruction", "") or ""
            relevant_dimensions = llm_extract_layer3_details(instr, skeleton, model_config, rid)
            for dim in relevant_dimensions:
                layer = dim.get("layer")
                sub = normalize_subcategory(dim.get("subcategory", ""))
                details = dim.get("details") or []
                if layer not in ROOT_LAYERS or not sub:
                    continue
                tree.setdefault(layer, {})
                tree[layer].setdefault(sub, {})
                for d in details:
                    key = normalize_key(d.get("key", ""))
                    q = (d.get("question") or "").strip()
                    if key and q:
                        tree_upsert_node(tree, {"layer": layer, "subcategory": sub, "key": key, "question": q}, unique_scope="subcategory")
        pbar.close()
        tree = sort_tree(tree)
        _pre_cluster_path.parent.mkdir(parents=True, exist_ok=True)
        _pre_cluster_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[聚类前输出] 已写入: {_pre_cluster_path}")

    n_after_upsert = sum(len(tree.get(l, {}).get(s, {})) for l in ROOT_LAYERS for s in tree.get(l, {}))
    if n_after_upsert == 0:
        Path(_out_path).write_text(json.dumps({l: {} for l in ROOT_LAYERS}, indent=2), encoding="utf-8")
        print(f"已写入空树: {_out_path}")
        return

    if not do_clustering:
        Path(_out_path).write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[直接结束] 已写入: {_out_path}")
        return

    if Path(_out_path).exists():
        print(f"[聚类] 跳过：目标文件已存在 {_out_path}")
        return
    if not _pre_cluster_path.exists():
        print(f"[聚类] 跳过：聚类输入文件不存在 {_pre_cluster_path}")
        return
    tree2, _ = cluster_within_subcategories(tree, model_config, embedding_model, final_within_sub_threshold, use_llm_only_canonicalize=True)
    Path(_out_path).write_text(json.dumps(tree2, ensure_ascii=False, indent=2), encoding="utf-8")
    n_final = sum(len(tree2[l][s]) for l in ROOT_LAYERS for s in tree2[l])
    print(f"已写入: {_out_path} (节点数={n_final})")
