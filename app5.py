import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
from aizynthfinder.aizynthfinder import AiZynthFinder

"""
========================================================
RETROSYNTHESIS DASHBOARD — FINAL VERSION v2
========================================================
Fixes:
- restored recap table (global comparison)
- shows STARTING MATERIALS instead of product
- unified route cards preserved
- stable scoring + selection system
"""


# =====================================================
# INIT
# =====================================================
finder = AiZynthFinder(configfile="config.yml")
finder.stock.select("zinc")
finder.expansion_policy.select("uspto")
finder.filter_policy.select("uspto")


def set_config(finder):
    try:
        finder.config.search.max_expansion = 200
        finder.config.search.time_limit = 120
        finder.config.search.max_iterations = 500
    except:
        pass

set_config(finder)


# =====================================================
# HELPERS
# =====================================================
def safe_smiles(smi):
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    return smi if mol else None


def get_tree(route):
    return route["reaction_tree"] if isinstance(route, dict) else route.reaction_tree


def get_root_smiles(tree):
    if isinstance(tree, dict):
        return safe_smiles(tree.get("smiles"))
    return safe_smiles(getattr(tree.root, "smiles", None))


def get_leaves(tree):
    """Starting materials (leaf nodes)"""

    if isinstance(tree, dict):

        def walk(n):
            smi = safe_smiles(n.get("smiles"))
            if not smi:
                return []

            children = n.get("children", [])
            if not children:
                return [smi]

            out = []
            for c in children:
                out.extend(walk(c))
            return out

        return walk(tree)

    return [
        safe_smiles(getattr(l, "smiles", None))
        for l in getattr(tree, "leafs", lambda: [])()
        if safe_smiles(getattr(l, "smiles", None))
    ]


# =====================================================
# SEARCH
# =====================================================
def run_search(target):
    finder.target_smiles = target
    finder.tree_search()
    finder.build_routes()
    return getattr(finder, "routes", [])


# =====================================================
# METRICS
# =====================================================
def compute_metrics(route):
    tree = get_tree(route)

    steps = len(get_leaves(tree))

    return {
        "yield": 1 / (1 + steps),
        "toxicity": max(0, 1 - 0.1 * steps),
        "e_factor": 1 / (1 + 0.2 * steps),
        "steps": 1 / max(steps, 1)
    }


# =====================================================
# GRAPH
# =====================================================
def build_graph(tree):
    G = nx.DiGraph()

    def walk(n, parent=None):
        smi = n.get("smiles") if isinstance(n, dict) else getattr(n, "smiles", None)
        if not smi:
            return

        G.add_node(smi)

        if parent:
            G.add_edge(parent, smi)

        children = n.get("children", []) if isinstance(n, dict) else []
        for c in children:
            walk(c, smi)

    walk(tree)
    return G


def plot_graph(G):
    if len(G.nodes) == 0:
        st.warning("No network available")
        return

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, labels = [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        labels.append(n)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=10)
    ))

    fig.update_layout(title="Reaction Network", showlegend=False)
    st.plotly_chart(fig)


# =====================================================
# UI
# =====================================================
st.title("🧬 Retrosynthesis Explorer (Final v2)")

target = st.text_input(
    "Target SMILES",
    "CC(=O)OC1=CC=CC=C1C(=O)O"
)


# =====================================================
# USER CRITERIA
# =====================================================
st.subheader("Select criteria")

use_yield = st.checkbox("Yield")
use_tox = st.checkbox("Toxicity")
use_ef = st.checkbox("E-factor")
use_steps = st.checkbox("Steps")

selected = []
if use_yield: selected.append("yield")
if use_tox: selected.append("toxicity")
if use_ef: selected.append("e_factor")
if use_steps: selected.append("steps")


# =====================================================
# RUN
# =====================================================
if st.button("Run Analysis"):

    if not selected:
        st.warning("Select at least one criterion")
        st.stop()

    routes = run_search(target)

    if not routes:
        st.error("No routes found")
        st.stop()


    # =================================================
    # SCORE ROUTES
    # =================================================
    results = []

    for r in routes:
        m = compute_metrics(r)
        score = sum(m[c] for c in selected)
        results.append((score, r, m))

    results.sort(reverse=True, key=lambda x: x[0])


    # =================================================
    # 📊 RECAP TABLE (RESTORED)
    # =================================================
    st.subheader("📊 Route Comparison Table")

    recap = []
    for score, _, m in results:
        recap.append({
            "score": round(score, 3),
            "yield": round(m["yield"], 3),
            "toxicity": round(m["toxicity"], 3),
            "e_factor": round(m["e_factor"], 3),
            "steps": round(m["steps"], 3),
            "criteria_used": ",".join(selected)
        })

    st.dataframe(pd.DataFrame(recap))


    # =================================================
    # 🧬 ROUTE CARDS
    # =================================================
    st.subheader("🧬 Detailed Routes")

    def show_route(i, score, route, metrics):

        tree = get_tree(route)

        # 🔥 FIX: STARTING MATERIALS (NOT PRODUCT)
        starting_materials = get_leaves(tree)

        with st.expander(f"Route {i+1} | Score {round(score,3)}", expanded=(i == 0)):

            # ---------------------
            # METRICS
            # ---------------------
            st.subheader("📊 Metrics")
            st.json(metrics)

            # ---------------------
            # STARTING MATERIALS
            # ---------------------
            st.subheader("🧪 Starting Materials")
            for smi in starting_materials:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    st.image(Draw.MolToImage(mol), caption=smi)

            # ---------------------
            # NETWORK
            # ---------------------
            st.subheader("🔬 Reaction Network")
            G = build_graph(tree)
            plot_graph(G)


    for i, (score, route, metrics) in enumerate(results):
        show_route(i, score, route, metrics)