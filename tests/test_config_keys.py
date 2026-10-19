"""Every config key the code reads must exist in the REAL config.yaml (a reverted file once lost keys silently)."""
import ast

import yaml

from csmr.config import ROOT
from csmr.pipeline import dependency, evidence

SOURCES = [*ROOT.glob("csmr/**/*.py"), *ROOT.glob("baselines/**/*.py"), *ROOT.glob("eval/**/*.py"),
           *ROOT.glob("examples/*.py"), ROOT / "run.py"]
LLM_SECTION_FUNCS = {"_complete", "_retry_wait"}  # llm.py: `cfg = cfg["llm"]` happens inside these


def _chain(node):
    keys = []
    while isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        keys.append(node.slice.value)
        node = node.value
    return keys[::-1], node


def _is_cfg(node) -> bool:
    return (isinstance(node, ast.Name) and node.id == "cfg") or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_config")


def _reads(path) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()

    def visit(node, func=None):
        func = node.name if isinstance(node, ast.FunctionDef) else func
        prefix = ("llm",) if path.name == "llm.py" and func in LLM_SECTION_FUNCS else ()
        keys = ()
        if isinstance(node, ast.Subscript):
            keys, base = _chain(node)
            keys = tuple(keys) if keys and _is_cfg(base) else ()
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
              and _is_cfg(node.func.value) and node.args and isinstance(node.args[0], ast.Constant)):
            keys = (node.args[0].value,)
        if keys and not (prefix and keys == ("llm",)):  # skip the `cfg = cfg["llm"]` line itself
            found.add(prefix + keys)
        for child in ast.iter_child_nodes(node):
            visit(child, func)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def test_every_key_the_code_reads_is_in_config_yaml():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    reads = {k: p for p in SOURCES for k in _reads(p)}
    assert ("llm", "quota_patterns") in reads and ("extract_context_k",) in reads  # the scanner sees what it should
    missing = []
    for keys, path in reads.items():
        node = cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                missing.append(f"{'.'.join(keys)}  (read in {path.relative_to(ROOT)})")
                break
            node = node[k]
    assert not missing, "keys read by the code but missing from config.yaml:\n" + "\n".join(sorted(missing))


def test_classifier_weights_cover_every_feature():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert set(dependency.FEATURES) <= set(cfg["dependency_weights"])
    assert set(evidence.FEATURES) <= set(cfg["evidence_weights"])
