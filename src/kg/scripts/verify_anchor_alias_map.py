import os
import runpy
from types import SimpleNamespace


def main() -> int:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_py = os.path.join(base_dir, "pipeline", "main.py")
    mod = runpy.run_path(main_py, run_name="kg_main_verify")

    build_alias = mod.get("_build_alias_map_by_vectors")
    if build_alias is None:
        raise RuntimeError("missing _build_alias_map_by_vectors")

    names = ["A", "B", "C", "D"]
    vecs = [
        [0.9063078, -0.42261827],
        [1.0, 0.0],
        [0.9063078, 0.42261827],
        [0.64278764, 0.76604444],
    ]

    def _fake_embed_texts(texts, dimensions):
        idx = {n: i for i, n in enumerate(names)}
        out = []
        for t in texts:
            out.append(vecs[idx[t]])
        return out

    build_alias.__globals__["_embed_texts"] = _fake_embed_texts

    entities = [SimpleNamespace(name=n, entity_type="组织机构") for n in names]
    alias_map = build_alias(entities, dimensions=2, threshold=0.86)

    assert isinstance(alias_map, dict)
    assert len(alias_map) == 2
    assert ("A" in alias_map) != ("D" in alias_map)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
