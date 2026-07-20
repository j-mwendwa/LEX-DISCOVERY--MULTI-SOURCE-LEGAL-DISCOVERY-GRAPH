"""
scripts/visualise_graph.py — Generate a Mermaid diagram of the discovery pipeline.
"""
from src.graph.graph import build_graph

app = build_graph()

try:
    graph = app.get_graph()
    mermaid = graph.draw_mermaid()
    print("```mermaid")
    print(mermaid)
    print("```")

    # Also save to file
    with open("docs/graph_diagram.md", "w") as f:
        f.write("# LEX-DISCOVERY Pipeline Diagram\n\n```mermaid\n")
        f.write(mermaid)
        f.write("\n```\n")

    print("\n✅ Diagram saved to docs/graph_diagram.md")
except Exception as exc:
    print(f"⚠ Could not generate diagram: {exc}")
    print("Install graphviz: pip install pygraphviz")
