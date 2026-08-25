"""
Phase 4, Step 4.3 - interactive visualisation of the knowledge graph.

WHY THIS MATTERS BEYOND LOOKING NICE
------------------------------------
Two real uses, only one of which is the demo:

1. DEBUGGING. A graph is an abstract object until you look at it. Fifteen minutes
   with this HTML file tells you things no stats table will: that one hub node has
   swallowed half the corpus through bad entity merging, that a whole paper sits
   disconnected because its entities never matched anything, that "outperforms"
   edges point the wrong way. All of those are invisible in a node/edge count.

2. THE DEMO. "I built a knowledge graph" is a claim. An interactive graph you can
   drag, zoom and click, with real paper entities in it, is evidence.

HOW NODES ARE TYPED (and why not with NER)
------------------------------------------
Colour-coding needs entity types, and notebooks/02_ner_demo.py showed spaCy is
unreliable on this text - it tagged "Swin Transformer" as an organisation and
missed DETR entirely. So we infer type from the GRAPH'S OWN STRUCTURE instead:

    the object of a "trained_on" edge        is a DATASET
    the object of an "achieves" edge         is a METRIC
    the subject of "outperforms"/"achieves"  is a MODEL
    the subject of "proposes"                is a PERSON

An entity's type is revealed by the relations it participates in. That is
information the graph has and a general-purpose NER model does not, and it is
more accurate here precisely because it is domain-specific.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.knowledge_graph import KnowledgeGraph  # noqa: E402
from src.utils import GRAPH_VIZ_PATH, banner, ensure_dirs, setup_console  # noqa: E402

TYPE_COLOURS = {
    "PERSON":     "#4A9EFF",   # blue
    "MODEL":      "#4CD97B",   # green
    "DATASET":    "#FFB84D",   # orange
    "METRIC":     "#FF6B9D",   # pink
    "ORG":        "#B388FF",   # purple
    "CONCEPT":    "#9AA5B1",   # grey
}

# Relations that reveal what their subject/object IS.
_OBJECT_TYPE_BY_RELATION = {
    "trained_on": "DATASET",
    "evaluated_on": "DATASET",
    "achieves": "METRIC",
}
_SUBJECT_TYPE_BY_RELATION = {
    "outperforms": "MODEL",
    "underperforms": "MODEL",
    "achieves": "MODEL",
    "trained_on": "MODEL",
    "evaluated_on": "MODEL",
    "improves_on": "MODEL",
    "extends": "MODEL",
    "proposes": "PERSON",
    "introduces": "PERSON",
}

_PERSON_RE = re.compile(r"\bet al\.?$|^[A-Z]\.\s*[A-Z][a-z]+", re.IGNORECASE)
_ORG_WORDS = {"google", "microsoft", "facebook", "meta", "openai", "nvidia",
              "university", "research", "institute", "lab", "labs", "brain",
              "deepmind", "berkeley", "stanford", "mit"}
_METRIC_RE = re.compile(r"%|\bmap\b|\bap\b|accuracy|bleu|fps|top-?\d|score|error rate",
                        re.IGNORECASE)
_MODEL_RE = re.compile(r"r-cnn|net\b|yolo|transformer|bert|gpt|vgg|detr|swin|vit\b",
                       re.IGNORECASE)
_DATASET_RE = re.compile(r"coco|imagenet|pascal|voc|cityscapes|jft|wmt|kitti|"
                         r"\bdataset\b|\bbenchmark\b", re.IGNORECASE)


def infer_entity_type(kg: KnowledgeGraph, node: str) -> str:
    """
    Guess an entity's type from the relations it takes part in, then from its name.

    Structural evidence wins over string patterns because it is corpus-specific:
    whatever sits on the receiving end of "trained_on" in THESE papers is a
    dataset, whether or not its name looks like one.
    """
    votes: Counter = Counter()

    for _, _, data in kg.graph.out_edges(node, data=True):
        entity_type = _SUBJECT_TYPE_BY_RELATION.get(data["relation"])
        if entity_type:
            votes[entity_type] += 2          # structural evidence, weighted higher

    for _, _, data in kg.graph.in_edges(node, data=True):
        entity_type = _OBJECT_TYPE_BY_RELATION.get(data["relation"])
        if entity_type:
            votes[entity_type] += 2
        if data["relation"] in {"proposes", "introduces", "developer", "designed by",
                                "creator", "discoverer or inventor"}:
            votes["MODEL"] += 1              # things that get proposed are usually models

    name = kg.display_name(node)
    if _PERSON_RE.search(name):
        votes["PERSON"] += 3                 # "X et al." is near-conclusive
    if any(word in name.lower().split() for word in _ORG_WORDS):
        votes["ORG"] += 2
    if _METRIC_RE.search(name):
        votes["METRIC"] += 2
    if _DATASET_RE.search(name):
        votes["DATASET"] += 2
    if _MODEL_RE.search(name):
        votes["MODEL"] += 2

    return votes.most_common(1)[0][0] if votes else "CONCEPT"


def _legend_html() -> str:
    swatches = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<span style="width:14px;height:14px;border-radius:50%;background:{colour};'
        f'display:inline-block"></span><span>{name}</span></div>'
        for name, colour in TYPE_COLOURS.items()
    )
    return f"""
<div style="position:fixed;top:14px;left:14px;z-index:999;background:rgba(20,22,28,.92);
     color:#e6e9ef;padding:14px 16px;border-radius:10px;font:13px/1.4 system-ui,sans-serif;
     box-shadow:0 4px 20px rgba(0,0,0,.4);max-width:260px">
  <div style="font-weight:600;margin-bottom:8px">Entity types</div>
  {swatches}
  <div style="margin-top:10px;padding-top:9px;border-top:1px solid #333;color:#9aa5b1">
    Node size = number of connections.<br>
    Hover a node for its aliases and sources.<br>
    Drag to rearrange, scroll to zoom.
  </div>
</div>
"""


def build_visualisation(
    kg: KnowledgeGraph,
    entity: str | None = None,
    hops: int = 2,
    max_nodes: int = 80,
    output: Path = GRAPH_VIZ_PATH,
    height: str = "820px",
) -> Path:
    """Render an interactive HTML graph, either the whole graph or a neighbourhood."""
    from pyvis.network import Network

    ensure_dirs()

    if entity:
        subgraph = kg.get_subgraph(entity, hops=hops, max_nodes=max_nodes)
        if subgraph.number_of_nodes() == 0:
            raise SystemExit(f"Entity {entity!r} not found in the graph.")
        title = f"{kg.display_name(kg.find_entity(entity))} - {hops}-hop neighbourhood"
    else:
        # Whole graph would be unreadable; show the most connected core.
        degrees = dict(kg.graph.degree())
        top = [n for n, _ in sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:max_nodes]]
        subgraph = kg.graph.subgraph(top).copy()
        title = f"Knowledge graph - {max_nodes} most connected entities"

    network = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#12141a",
        font_color="#e6e9ef",
        # in_line embeds vis.js in the file itself, so the HTML works offline and
        # can be dropped straight into Streamlit in Phase 7.
        cdn_resources="in_line",
    )

    degrees = dict(subgraph.degree())

    for node in subgraph.nodes():
        data = kg.graph.nodes[node]
        entity_type = infer_entity_type(kg, node)
        degree = degrees.get(node, 1)

        aliases = sorted(data.get("aliases", ()))[:6]
        sources = sorted(data.get("sources", ()))
        tooltip = (
            f"{data.get('name', node)}\n"
            f"type: {entity_type}\n"
            f"connections: {degree}\n"
            f"mentions: {data.get('mentions', 0)}\n"
            f"papers: {', '.join(sources) if sources else '-'}\n"
            f"aliases: {', '.join(aliases) if aliases else '-'}"
        )

        network.add_node(
            node,
            label=data.get("name", node),
            title=tooltip,
            color=TYPE_COLOURS[entity_type],
            # sqrt keeps one 200-degree hub from dwarfing everything else.
            size=12 + (degree ** 0.5) * 5,
            font={"size": 15, "color": "#e6e9ef"},
        )

    # Collapse parallel edges with the same relation so the picture stays readable,
    # but keep the count so nothing is silently hidden.
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for source, target, data in subgraph.edges(data=True):
        grouped[(source, target, data["relation"])].append(data)

    for (source, target, relation), edges in grouped.items():
        citations = {f"{e.get('source_file', '?')} p.{e.get('page', '?')}" for e in edges}
        sentence = next((e.get("sentence", "") for e in edges if e.get("sentence")), "")
        label = relation if len(edges) == 1 else f"{relation} (x{len(edges)})"
        network.add_edge(
            source, target,
            label=label,
            title=f"{relation}\n{'; '.join(sorted(citations))}\n\n{sentence[:220]}",
            color={"color": "#5a6472", "highlight": "#4A9EFF"},
            font={"size": 10, "color": "#9aa5b1", "strokeWidth": 0},
            arrows="to",
        )

    # Force-directed layout. These constants are tuned for a readable spread at
    # this scale: springLength keeps labels from overlapping, gravitationalConstant
    # controls how tightly clusters pull together.
    network.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -55,
          "centralGravity": 0.012,
          "springLength": 130,
          "springConstant": 0.09,
          "damping": 0.45
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 220},
        "minVelocity": 0.75
      },
      "interaction": {"hover": true, "tooltipDelay": 120, "navigationButtons": true},
      "edges": {"smooth": {"type": "continuous"}, "width": 1.2}
    }
    """)

    # NOT network.write_html(): pyvis opens the file with the platform default
    # encoding, which on Windows is cp1252, and dies on the first non-ASCII
    # character in a node label or tooltip - and this corpus is full of them
    # (Greek letters, en-dashes, accented author names). Generating the HTML and
    # writing it ourselves as UTF-8 sidesteps that entirely.
    html = network.generate_html(notebook=False)

    # pyvis has no legend support, so inject one directly into the page.
    html = html.replace("<body>", f"<body>{_legend_html()}", 1)
    html = html.replace("</head>", f"<title>{title}</title></head>", 1)
    output.write_text(html, encoding="utf-8")

    return output


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Visualise the knowledge graph")
    parser.add_argument("--entity", help="centre the view on this entity")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=80)
    parser.add_argument("--output", type=Path, default=GRAPH_VIZ_PATH)
    args = parser.parse_args()

    kg = KnowledgeGraph.load()
    stats = kg.stats()
    print(f"Loaded graph: {stats['nodes']:,} nodes, {stats['edges']:,} edges")

    # Default to the best-connected entity - the most interesting neighbourhood.
    entity = args.entity
    if entity is None and stats["top_nodes"]:
        entity = kg.display_name(stats["top_nodes"][0][0])
        print(f"No --entity given; centring on the hub node: {entity!r}")

    path = build_visualisation(
        kg, entity=entity, hops=args.hops, max_nodes=args.max_nodes, output=args.output
    )

    print(banner("VISUALISATION WRITTEN"))
    print(f"  {path}")
    print(f"  {path.stat().st_size / 1024:.0f} KB, self-contained (works offline)")

    types = Counter(infer_entity_type(kg, n) for n in kg.graph.nodes())
    print("\n  entity types across the whole graph:")
    for entity_type, count in types.most_common():
        print(f"     {entity_type:<10} {count:>6}")

    print("\n  Open it in a browser:")
    print(f"     start {path}")


if __name__ == "__main__":
    main()
