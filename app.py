"""
app_streamlit.py — Interface Streamlit pour le pipeline de sélection de routes de synthèse.

Utilisation :
    streamlit run app_streamlit.py

Prérequis : tous les fichiers du projet dans le même dossier (voir README).
"""

import streamlit as st
import json
import os
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO

# ---------------------------------------------------------------------------
# Import du pipeline principal
# On importe depuis le fichier pipeline.py (à adapter si votre fichier a un autre nom)
# ---------------------------------------------------------------------------
try:
    from pipeline import (
        find_best_routes,
        CRITERIA_REGISTRY,
        load_reaction_dataset,
        load_toxicity_dataset,
        run_aizynthfinder,
        adapt_route,
        filter_routes,
        rank_weighted,
        rank_borda,
        SCORING_METHOD,
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RetroRoute — Sélection de routes de synthèse",
    page_icon="⚗️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS minimaliste pour améliorer la lisibilité
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.route-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #4c8bf5;
}
.rank-badge {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4c8bf5;
}
.score-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2d6a4f;
}
.step-box {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
}
.warning-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Barre latérale — paramètres utilisateur
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚗️ RetroRoute")
    st.caption("Sélection de routes de synthèse rétrosynthétique")
    st.divider()

    # --- Molécule cible ---
    st.subheader("🎯 Molécule cible")
    target_smiles_input = st.text_input(
        "SMILES de la molécule",
        value="OC1=CC2=C(C=C1)[C@@H]1[C@H]3C[C@@H](O)C=C[C@@H]3N(C)CC1=C2",
        help="Entrez le SMILES de la molécule à synthétiser. Exemple : morphine."
    )

    # Validation du SMILES en temps réel
    mol_valid = None
    if target_smiles_input:
        mol_valid = Chem.MolFromSmiles(target_smiles_input)
        if mol_valid:
            st.success("✓ SMILES valide")
        else:
            st.error("✗ SMILES invalide — vérifiez la syntaxe")

    st.divider()

    # --- Critères de sélection ---
    st.subheader("📊 Critères de sélection")
    st.caption("Choisissez 2 à 5 critères et ordonnez-les par priorité décroissante.")

    available_criteria = list(CRITERIA_REGISTRY.keys()) if PIPELINE_AVAILABLE else [
        "steps", "yield", "e_factor", "atom_economy", "toxicity"
    ]
    criteria_labels = {
        "steps": "Nombre d'étapes (moins = mieux)",
        "yield": "Rendement global",
        "e_factor": "E-factor (déchets)",
        "atom_economy": "Économie atomique",
        "toxicity": "Sécurité / toxicité",
    }

    selected_criteria = []
    st.markdown("**Priorité 1** (la plus importante)")
    c1 = st.selectbox("", available_criteria, index=0, key="c1",
                       format_func=lambda x: criteria_labels.get(x, x))
    selected_criteria.append(c1)

    st.markdown("**Priorité 2**")
    remaining2 = [c for c in available_criteria if c != c1]
    c2 = st.selectbox("", remaining2, index=0, key="c2",
                       format_func=lambda x: criteria_labels.get(x, x))
    selected_criteria.append(c2)

    add_third = st.checkbox("Ajouter un 3ᵉ critère", value=True)
    if add_third:
        remaining3 = [c for c in available_criteria if c not in selected_criteria]
        c3 = st.selectbox("**Priorité 3**", remaining3, index=0, key="c3",
                           format_func=lambda x: criteria_labels.get(x, x))
        selected_criteria.append(c3)

    add_fourth = st.checkbox("Ajouter un 4ᵉ critère", value=False)
    if add_fourth and add_third:
        remaining4 = [c for c in available_criteria if c not in selected_criteria]
        if remaining4:
            c4 = st.selectbox("**Priorité 4**", remaining4, index=0, key="c4",
                               format_func=lambda x: criteria_labels.get(x, x))
            selected_criteria.append(c4)

    st.divider()

    # --- Méthode de scoring ---
    st.subheader("⚖️ Méthode de classement")
    method = st.radio(
        "",
        options=["weighted", "borda"],
        index=0 if SCORING_METHOD == "weighted" else 1,
        format_func=lambda x: {
            "weighted": "Somme pondérée — priorité forte au critère 1",
            "borda": "Borda Count — favorise le meilleur compromis"
        }[x],
        help=(
            "**Somme pondérée** : le critère 1 a un poids ~4× supérieur au critère 2. "
            "Rapide et intuitif.\n\n"
            "**Borda Count** : on classe les routes pour chaque critère séparément, "
            "puis on cumule les rangs pondérés. Une route bonne partout peut battre "
            "une route excellente sur un seul critère."
        )
    )

    st.divider()

    # --- Paramètres avancés ---
    with st.expander("⚙️ Paramètres avancés"):
        top_n = st.slider("Nombre de routes à afficher", 1, 10, 3)
        dataset_path = st.text_input("Chemin reaction_dataset.json", value="reaction-dataset.json")
        toxicity_path = st.text_input("Chemin toxicity_dataset.json", value="toxicity_dataset.json")
        config_path = st.text_input("Chemin config.yml (AiZynthFinder)", value="config.yml")

    st.divider()
    run_button = st.button("🔍 Lancer l'analyse", type="primary", use_container_width=True,
                            disabled=(mol_valid is None))


# ---------------------------------------------------------------------------
# Zone principale
# ---------------------------------------------------------------------------
st.title("Sélection de routes de synthèse rétrosynthétique")

# Vérification des imports
if not PIPELINE_AVAILABLE:
    st.error(f"❌ Impossible d'importer le pipeline : `{IMPORT_ERROR}`")
    st.info("Vérifiez que `pipeline.py` est dans le même dossier que `app_streamlit.py`.")
    st.stop()

# Aperçu de la molécule
if mol_valid:
    col_mol, col_info = st.columns([1, 2])
    with col_mol:
        st.subheader("Molécule cible")
        img = Draw.MolToImage(mol_valid, size=(280, 200))
        buf = BytesIO()
        img.save(buf, format="PNG")
        st.image(buf.getvalue(), caption=f"SMILES : {target_smiles_input[:60]}{'…' if len(target_smiles_input) > 60 else ''}")

    with col_info:
        st.subheader("Paramètres sélectionnés")
        st.markdown(f"**Critères** (ordre de priorité) :")
        for i, c in enumerate(selected_criteria, 1):
            weight_hint = round(1 / (i ** 2), 3)
            st.markdown(f"  {i}. `{c}` — {criteria_labels.get(c, c)} *(poids relatif ≈ {weight_hint})*")
        st.markdown(f"**Méthode :** `{method}`")
        st.markdown(f"**Top N :** {top_n} routes")

# ---------------------------------------------------------------------------
# Lancement du pipeline
# ---------------------------------------------------------------------------
if run_button and mol_valid:
    st.divider()
    st.subheader("🔄 Analyse en cours...")

    progress = st.progress(0, text="Chargement des datasets...")
    log_area = st.empty()

    try:
        # Étape 1 — Datasets
        progress.progress(10, text="Chargement des datasets...")
        dataset = load_reaction_dataset(dataset_path)
        tox_index = load_toxicity_dataset(toxicity_path)
        log_area.success(f"✓ Dataset chargé : {len(dataset['all'])} réactions | {len(tox_index)} composés toxicité")

        # Étape 2 — AiZynthFinder
        progress.progress(30, text="Recherche des routes (AiZynthFinder)...")
        raw_routes = run_aizynthfinder(target_smiles_input, config_path)
        routes = [adapt_route(r) for r in raw_routes]
        log_area.info(f"ℹ️ {len(routes)} routes trouvées par AiZynthFinder")

        # Étape 3 — Filtrage
        progress.progress(60, text="Filtrage par le dataset...")
        valid_routes = filter_routes(routes, dataset)

        if not valid_routes:
            progress.progress(100, text="Terminé")
            log_area.empty()
            st.markdown("""
<div class="warning-box">
<strong>⚠️ Aucune route n'a passé le filtrage.</strong><br>
Cela signifie qu'aucune étape proposée par AiZynthFinder ne correspond
à une réaction répertoriée dans votre dataset.<br><br>
<strong>Solutions :</strong>
<ul>
<li>Enrichissez <code>reaction-dataset.json</code> avec des réactions supplémentaires.</li>
<li>Vérifiez que les SMILES du dataset correspondent à ceux qu'AiZynthFinder utilise.</li>
<li>Si vous êtes en <code>MOCK_MODE=True</code>, vérifiez les SMILES des routes simulées.</li>
</ul>
</div>
""", unsafe_allow_html=True)
            st.stop()

        # Étape 4 — Scoring
        progress.progress(85, text="Scoring et classement...")
        if method == "borda":
            top_routes = rank_borda(valid_routes, selected_criteria, tox_index)[:top_n]
        else:
            top_routes = rank_weighted(valid_routes, selected_criteria, tox_index)[:top_n]

        progress.progress(100, text="Analyse terminée ✓")
        log_area.empty()

        # ---------------------------------------------------------------------------
        # Affichage des résultats
        # ---------------------------------------------------------------------------
        st.divider()
        n_filtered = len(valid_routes)
        st.success(f"✅ {n_filtered}/{len(routes)} routes validées — affichage du top {min(top_n, n_filtered)}")

        for rank, (score, details, route) in enumerate(top_routes, 1):
            route_name = route.get("route_name", f"Route {rank}")
            n_steps = len(route.get("steps", []))

            with st.expander(f"🥇 Rang {rank} — {route_name}   |   Score : {score:.4f}   |   {n_steps} étape(s)", expanded=(rank == 1)):

                # Score global et méthode
                col_score, col_method = st.columns([1, 2])
                with col_score:
                    st.markdown(f'<p class="score-value">{score:.4f}</p>', unsafe_allow_html=True)
                    st.caption(f"Score global ({method})")
                with col_method:
                    st.caption(f"Méthode : **{method}** | Critères : {', '.join(selected_criteria)}")

                st.markdown("---")

                # Tableau des critères
                st.markdown("**Détail par critère**")

                crit_cols = st.columns(len(selected_criteria))
                for i, c in enumerate(selected_criteria):
                    d = details.get(c, {})
                    with crit_cols[i]:
                        raw_val = d.get("raw", 0)
                        label = criteria_labels.get(c, c)
                        st.metric(
                            label=label,
                            value=f"{raw_val:.3f}",
                            help=f"Poids : {d.get('weight', d.get('weighted_pts', '?'))}"
                                 + (f" | Rang : {d.get('rank','?')}" if method == "borda" else
                                    f" | Contribution : {d.get('weighted', 0):.4f}")
                        )

                st.markdown("---")

                # Étapes de synthèse
                st.markdown("**Étapes de synthèse**")
                dataset_steps = route.get("dataset_steps", [])
                if dataset_steps:
                    for i, step in enumerate(dataset_steps, 1):
                        name_step = step.get("reaction_name", "Étape inconnue")
                        rtype = step.get("reaction_type", "?")
                        yld = step.get("yield_percent", "?")
                        conditions = step.get("conditions", {})
                        solvent = conditions.get("solvent", "?")
                        temp = conditions.get("temperature_c", "?")
                        reactants = step.get("reactants_smiles", [])

                        st.markdown(f"""
<div class="step-box">
<strong>Étape {i} — {name_step}</strong> <em>({rtype})</em><br>
🧪 Réactifs : <code>{' + '.join(reactants) if reactants else '—'}</code><br>
📈 Rendement : <strong>{yld}%</strong> &nbsp;|&nbsp;
🧴 Solvant : <strong>{solvent}</strong> &nbsp;|&nbsp;
🌡️ Température : <strong>{temp}°C</strong>
</div>
""", unsafe_allow_html=True)
                else:
                    st.info("Aucun détail d'étape disponible dans le dataset.")

                # Rendement global calculé
                if dataset_steps:
                    global_yield = 1.0
                    for s in dataset_steps:
                        global_yield *= s.get("yield_percent", 50) / 100.0
                    st.markdown(f"**Rendement global estimé :** {global_yield * 100:.1f}%")

        # ---------------------------------------------------------------------------
        # Export JSON
        # ---------------------------------------------------------------------------
        st.divider()
        st.subheader("📥 Exporter les résultats")

        export_data = []
        for rank, (score, details, route) in enumerate(top_routes, 1):
            export_data.append({
                "rank": rank,
                "score": score,
                "route_name": route.get("route_name", f"Route {rank}"),
                "n_steps": len(route.get("steps", [])),
                "criteria_details": details,
                "dataset_steps": route.get("dataset_steps", []),
            })

        export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Télécharger les résultats (JSON)",
            data=export_json,
            file_name=f"routes_{target_smiles_input[:20].replace('/', '_')}.json",
            mime="application/json",
        )

    except FileNotFoundError as e:
        progress.empty()
        st.error(f"❌ Fichier introuvable : {e}")
        st.info("Vérifiez les chemins dans les paramètres avancés de la barre latérale.")

    except ValueError as e:
        progress.empty()
        st.error(f"❌ Erreur de paramètre : {e}")

    except Exception as e:
        progress.empty()
        st.error(f"❌ Erreur inattendue : {e}")
        with st.expander("Détails techniques"):
            import traceback
            st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# État initial (avant de cliquer sur le bouton)
# ---------------------------------------------------------------------------
elif not run_button:
    st.info("👈 Configurez les paramètres dans la barre latérale, puis cliquez sur **Lancer l'analyse**.")

    with st.expander("ℹ️ Comment ça marche ?"):
        st.markdown("""
**Pipeline en 4 étapes :**

1. **Chargement** des datasets (`reaction-dataset.json` et `toxicity_dataset.json`)
2. **Recherche** rétrosynthétique via AiZynthFinder (ou simulation en `MOCK_MODE`)
3. **Filtrage** : on ne conserve que les routes dont *toutes* les étapes sont couvertes par le dataset
4. **Scoring & classement** selon vos critères et votre méthode de pondération

**Critères disponibles :**
- `steps` : nombre d'étapes (moins = mieux)
- `yield` : rendement global (produit des rendements)
- `e_factor` : E-factor estimé (moins de déchets = mieux)
- `atom_economy` : économie atomique (chimie verte)
- `toxicity` : sécurité des réactifs et solvants

**Méthodes de classement :**
- *Somme pondérée* : simple, le critère 1 domine
- *Borda Count* : favorise les routes équilibrées sur tous les critères
        """)