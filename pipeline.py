import json
import os
from rdkit import Chem
from rdkit.Chem import Descriptors


# Si True, on simule AiZynthFinder avec des routes fictives pour tester le code
# sans avoir à configurer les modèles et la base de données.
# Passer à False quand config.yml et les modèles sont en place.
MOCK_MODE = True

SCORING_METHOD = "weighted"   # "weighted" ou "borda" — voir explications en section 6

if not MOCK_MODE:
    from aizynthfinder.aizynthfinder import AiZynthFinder


# On charge deux fichiers : le dataset de réactions (minimal) et le dataset
# de toxicité (par composé). Les deux sont indexés en mémoire pour que les
# recherches pendant le filtrage et le scoring soient rapides.

def load_reaction_dataset(path: str) -> dict:
    """
    Charge le dataset de réactions et construit deux index :
      - by_product : SMILES canonique du produit → liste de réactions
      - by_route   : route_id → liste des étapes de cette route
    """
    print(f"[Dataset] Lecture de {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Vérifiez que reaction_dataset.json est dans le même dossier."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    reactions = raw.get("reactions", [])
    print(f"[Dataset] {len(reactions)} réactions chargées")

    by_product = {}
    for rxn in reactions:
        key = to_canonical(rxn["product_smiles"])  # SMILES normalisé pour comparaison fiable
        by_product.setdefault(key, []).append(rxn)

    by_route = {}
    for rxn in reactions:
        rid = rxn.get("route_id", "inconnu")
        by_route.setdefault(rid, []).append(rxn)

    return {"by_product": by_product, "by_route": by_route, "all": reactions}


def load_toxicity_dataset(path: str) -> dict:
    """
    Charge le dataset de toxicité et l'indexe par SMILES canonique.
    Retourne un dict  smiles_canonique → données du composé
    """
    print(f"[Toxicité] Lecture de {path}")

    if not os.path.exists(path):
        print(f"[Toxicité] Fichier {path} absent — les scores de toxicité seront estimés à 0.5")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    index = {}
    for compound in raw.get("compounds", []):
        key = to_canonical(compound["smiles"])
        index[key] = compound

    print(f"[Toxicité] {len(index)} composés indexés")
    return index


# pour eviter les erreurs
def to_canonical(smiles: str) -> str:
    """
    Convertit un SMILES en forme canonique via RDKit.
    Deux SMILES qui représentent la même molécule mais écrits différemment
    donnent la même forme canonique — ce qui permet de les comparer.
    En cas d'échec (SMILES invalide), on retourne la chaîne d'origine.
    """
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


"lien entre le dataset et aizynthfinder"
# Toute interaction avec AiZynthFinder est isolée ici.
# Si l'API ou la version change, seule cette section est à modifier.

def run_aizynthfinder(target_smiles: str, config_path: str = "config.yml") -> list:
    """
    Lance la recherche rétrosynthétique pour le SMILES cible donné.
    Retourne une liste de routes brutes (objets ou dicts selon la version).
    """
    if MOCK_MODE:
        print(f"[AiZynthFinder] Simulation activée — cible : {target_smiles}")
        return _mock_routes()

    print(f"[AiZynthFinder] Démarrage avec {config_path}")
    finder = AiZynthFinder(configfile=config_path)
    finder.stock.select("zinc")             # stock de molécules commerciales disponibles
    finder.expansion_policy.select("uspto") # modèle de rétrosynthèse entraîné sur USPTO
    finder.filter_policy.select("uspto")    # filtre pour éliminer les réactions peu plausibles

    finder.target_smiles = target_smiles
    print("[AiZynthFinder] Recherche en cours...")
    finder.tree_search()   # exploration de l'arbre rétrosynthétique (MCTS)
    finder.build_routes()  # conversion des chemins de l'arbre en routes exploitables

    print(f"[AiZynthFinder] {len(finder.routes)} routes trouvées")
    return list(finder.routes)


def _mock_routes() -> list:
    """
    Routes simulées pour développer et tester sans AiZynthFinder.
    Contiennent intentionnellement des étapes connues et inconnues du dataset
    pour vérifier que le filtrage fonctionne correctement.
    """
    return [
        {
            "route_id": "mock-rice",
            "route_name": "Rice biomimétique (simulé)",
            "steps": [
                {"reactants": ["COC1=CC=C2CC(NC)CCC2=C1O"], "product": "COC1=CC2=C(C=C1)CC1CC(=CC1=C2)O"},
                {"reactants": ["COC1=CC2=C(C=C1)CC1CC(=CC1=C2)O", "[Na+].[BH4-]"], "product": "COC1=CC2=C(C=C1)CC1CC(CC1=C2)O"},
                {"reactants": ["COC1=CC2=C(C=C1)CC1CC(CC1=C2)O"], "product": "COC1=CC2=C(C=C1)[C@@H]1[C@H]3CC=C[C@@H]([C@H]3OC=C2)[N]1C"},
                {"reactants": ["COC1=CC2=C(C=C1)[C@@H]1[C@H]3CC=C[C@@H]([C@H]3OC=C2)[N]1C"], "product": "COC1=CC2=C(C=C1)[C@@H]1[C@H]3C[C@@H](O)C=C[C@@H]3N(C)CC1=C2"},
                {"reactants": ["COC1=CC2=C(C=C1)[C@@H]1[C@H]3C[C@@H](O)C=C[C@@H]3N(C)CC1=C2", "BrBr.BBr3"], "product": "OC1=CC2=C(C=C1)[C@@H]1[C@H]3C[C@@H](O)C=C[C@@H]3N(C)CC1=C2"},
            ]
        },
        {
            "route_id": "mock-gates",
            "route_name": "Gates 1952 (simulé)",
            "steps": [
                {"reactants": ["C1=CC2=CC=CC=C2C=C1", "C=CC(=O)OCC"], "product": "O=C(OCC)C1CC2=C(C=C2)CC1"},
                {"reactants": ["O=C1CC2=CC=C(O)C=C2CC1=O", "[Na+].[BH4-]"], "product": "OC1CC2=CC=C(O)C=C2CC1=O"},
                {"reactants": ["O=C1CC2=CC=C(OC)C=C2CC1", "CN"], "product": "CNC1CC2=CC=C(OC)C=C2CC1"},
            ]
        },
        {
            "route_id": "mock-trost",
            "route_name": "Trost 2002 (simulé — étape inconnue)",
            "steps": [
                {"reactants": ["O=C1OCC=CC1=CC1=CC=C(O)C=C1"], "product": "O=C1OC[C@@H]2CC3=CC=C(O)C=C3[C@H]12"},
                {"reactants": ["REACTIF_INCONNU"], "product": "PRODUIT_HORS_DATASET"},  # sera rejetée
            ]
        },
        {
            "route_id": "mock-fukuyama",
            "route_name": "Fukuyama 2017 (simulé — hors dataset)",
            "steps": [
                {"reactants": ["COC1=CC(=O)CCC1=O", "CC(=O)OCC"], "product": "COC1=CC2(CC(O)=O)CCCCC2=C1"},
                {"reactants": ["AUTRE_INCONNU"], "product": "PRODUIT_INCONNU"},  # sera rejetée
            ]
        },
    ]



# AiZynthFinder peut renvoyer des objets Python ou des dicts selon la version
# Cet adaptateur normalise tout en un format uniforme avant de continuer

def adapt_route(route) -> dict:
    """
    Transforme une route brute d'AiZynthFinder en dict normalisé.
    Format de sortie : { route_id, route_name, steps: [{reactants, product}] }
    """
    if isinstance(route, dict) and "steps" in route:
        return route  # déjà au bon format (cas mock ou dict AiZynthFinder récent)

    # Extraction depuis un objet AiZynthFinder ou un dict avec reaction_tree
    tree = route.get("reaction_tree", {}) if isinstance(route, dict) else route.reaction_tree
    steps = _walk_tree(tree)

    return {
        "route_id":   _get(route, "route_id", "inconnu"),
        "route_name": _get(route, "route_name", "Route AiZynthFinder"),
        "steps": steps,
        "raw": route
    }


def _get(obj, key, default=None):
    """Lecture d'un attribut depuis un dict ou un objet indifféremment."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _walk_tree(tree) -> list:
    """
    Parcourt l'arbre rétrosynthétique et extrait chaque étape
    sous la forme {reactants: [...], product: "..."}.
    Supporte les arbres en format dict et les objets AiZynthFinder.
    """
    steps = []

    if isinstance(tree, dict):
        def descend(node):
            if not node:
                return
            smiles = node.get("smiles", "")
            children = node.get("children", [])
            if children:
                steps.append({
                    "reactants": [c.get("smiles", "") for c in children],
                    "product": smiles
                })
                for child in children:
                    descend(child)
        descend(tree)

    else:  # objet AiZynthFinder avec la méthode reactions()
        try:
            for rxn in tree.reactions():
                steps.append({
                    "reactants": [m.smiles for m in rxn.reactants],
                    "product": rxn.mol.smiles
                })
        except AttributeError:
            pass

    return steps



# On ne conserve que les routes dont CHAQUE étape est répertoriée dans
# reaction_dataset.json. Une route avec une seule étape manquante est rejetée
# car on n'aurait pas les données de rendement nécessaires pour la noter.

def filter_routes(routes: list, dataset: dict) -> list:
    """
    Filtre les routes candidates selon la couverture du dataset.
    Enrichit chaque route validée avec les données dataset de chaque étape.

    Retourne une liste de routes enrichies avec la clé "dataset_steps".
    """
    known = dataset["by_product"]  # index  SMILES canonique → réactions
    validated = []

    for route in routes:
        steps = route.get("steps", [])
        if not steps:
            print(f"  [Filtrage] '{route.get('route_name')}' — ignorée (aucune étape)")
            continue

        data_steps = []
        all_found = True

        for step in steps:
            canon = to_canonical(step.get("product", ""))
            if canon in known:
                data_steps.append(known[canon][0])  # on prend la première correspondance
            else:
                print(f"  [Filtrage] Étape absente du dataset : {step.get('product','?')[:45]}")
                all_found = False
                break

        if all_found:
            enriched = dict(route)
            enriched["dataset_steps"] = data_steps  # on attache les données pour le scoring
            validated.append(enriched)
            print(f"  [Filtrage] ✓ '{route.get('route_name')}' — {len(steps)} étapes validées")
        else:
            print(f"  [Filtrage] ✗ '{route.get('route_name')}' — rejetée")

    print(f"\n  {len(validated)}/{len(routes)} routes conservées après filtrage")
    return validated


#CALCUL DES MÉTRIQUES DEPUIS LES SMILES (pas besoin du dataset pour ça)

def calc_atom_economy(reactants_smiles: list, product_smiles: str) -> float:
    """
    Économie atomique = masse molaire du produit / somme des masses des réactifs.
    Mesure l'efficacité d'utilisation des atomes dans la réaction.
    Résultat entre 0 et 1 (1 = tous les atomes des réactifs se retrouvent dans le produit).
    """
    prod_mol = Chem.MolFromSmiles(product_smiles)
    if prod_mol is None:
        return 0.0

    prod_mw = Descriptors.MolWt(prod_mol)
    total_reactant_mw = 0.0

    for smi in reactants_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            total_reactant_mw += Descriptors.MolWt(mol)

    if total_reactant_mw == 0:
        return 0.0

    ae = prod_mw / total_reactant_mw  # rapport entre 0 et (en théorie) 1
    return min(ae, 1.0)               # on plafonne à 1.0 au cas où


def calc_e_factor(reactants_smiles: list, product_smiles: str, yield_fraction: float) -> float:
    """
    E-factor estimé = masse des déchets / masse du produit obtenu.
    Déchets = réactifs non incorporés dans le produit + pertes dues au rendement.
    Un E-factor élevé signifie beaucoup de déchets par gramme de produit.
    On retourne ensuite l'inverse normalisé pour que 1.0 soit le meilleur score.
    """
    prod_mol = Chem.MolFromSmiles(product_smiles)
    if prod_mol is None:
        return 0.5  # valeur neutre si le SMILES est invalide

    prod_mw = Descriptors.MolWt(prod_mol)
    total_mw = sum(
        Descriptors.MolWt(Chem.MolFromSmiles(s))
        for s in reactants_smiles
        if Chem.MolFromSmiles(s) is not None
    )

    product_obtained = prod_mw * max(yield_fraction, 0.01)  # masse effective de produit
    waste = total_mw - product_obtained                     # ce qui n'est pas dans le produit
    e_factor = max(waste, 0) / product_obtained

    return 1.0 / (1.0 + e_factor)  # on inverse : plus c'est bas, meilleur est le score


def calc_toxicity_score(reactants_smiles: list, solvent: str, tox_index: dict) -> float:
    """
    Score de sécurité moyen pour tous les composés d'une étape (réactifs + solvant).
    Les hazard_scores viennent du toxicity_dataset.json.
    Si un composé est absent du dataset de toxicité, on lui attribue 0.5 (prudence).
    On retourne l'inverse : 1.0 = aucun danger, 0.0 = très dangereux.
    """
    smiles_to_check = list(reactants_smiles)

    # Conversion du nom de solvant en SMILES si possible
    solvent_smiles_map = {
        "ethanol": "CCO", "methanol": "CO", "THF": "C1CCOC1",
        "DCM": "ClCCl", "DMF": "CN(C)C=O", "acetonitrile": "CC#N",
        "toluene": "Cc1ccccc1", "acide acetique": "CC(=O)O",
        "acide formique": "OC=O", "eau": "O", "aucun": None
    }

    solv_smi = solvent_smiles_map.get(solvent)
    if solv_smi:
        smiles_to_check.append(solv_smi)

    scores = []
    for smi in smiles_to_check:
        canon = to_canonical(smi)
        if canon in tox_index:
            scores.append(tox_index[canon]["hazard_score"])
        else:
            scores.append(0.5)  # composé inconnu → score de précaution par défaut

    if not scores:
        return 0.5

    avg_hazard = sum(scores) / len(scores)
    return 1.0 - avg_hazard  # on inverse : 0 de danger → score de 1.0


# REGISTRE DES CRITÈRES
# Chaque critère est une entrée dans ce dictionnaire.
# Pour ajouter un critère :
#   1. Écrire une fonction compute_X(route_data, tox_index) → float (entre 0 et 1)
#   2. L'ajouter dans CRITERIA_REGISTRY ci-dessous
#   3. Ajouter la clé correspondante dans reaction_dataset.json si besoin

def compute_atom_economy(route_data: dict, tox_index: dict) -> float:
    """
    Économie atomique moyenne sur toutes les étapes de la route.
    Calculée depuis les SMILES, pas depuis le dataset.
    """
    steps = route_data.get("dataset_steps", [])
    scores = []
    for s in steps:
        ae = calc_atom_economy(s.get("reactants_smiles", []), s.get("product_smiles", ""))
        scores.append(ae)
    return sum(scores) / len(scores) if scores else 0.0


def compute_steps(route_data: dict, tox_index: dict) -> float:
    """
    Score basé sur la longueur de la route.
    1/n_étapes : une route en 2 étapes donne 0.5, en 5 étapes donne 0.2, etc.
    """
    n = len(route_data.get("steps", []))
    return 1.0 / max(n, 1)


def compute_yield(route_data: dict, tox_index: dict) -> float:
    """
    Rendement global = produit des rendements de toutes les étapes.
    Ex : 80% × 95% × 98% × 80% × 86% ≈ 50.5%
    Valeur entre 0 et 1.
    """
    steps = route_data.get("dataset_steps", [])
    if not steps:
        return 0.0
    result = 1.0
    for s in steps:
        result *= s.get("yield_percent", 50) / 100.0  # on multiplie les rendements
    return result


def compute_e_factor(route_data: dict, tox_index: dict) -> float:
    """
    E-factor moyen sur toutes les étapes, calculé depuis les SMILES.
    Retourne un score entre 0 et 1 (1 = peu de déchets).
    """
    steps = route_data.get("dataset_steps", [])
    scores = []
    for s in steps:
        y = s.get("yield_percent", 50) / 100.0
        ef_score = calc_e_factor(
            s.get("reactants_smiles", []),
            s.get("product_smiles", ""),
            y
        )
        scores.append(ef_score)
    return sum(scores) / len(scores) if scores else 0.0


def compute_toxicity(route_data: dict, tox_index: dict) -> float:
    """
    Score de sécurité moyen sur toutes les étapes de la route.
    Consulte le toxicity_dataset pour chaque réactif et solvant.
    Retourne un score entre 0 et 1 (1 = aucun danger).
    """
    steps = route_data.get("dataset_steps", [])
    scores = []
    for s in steps:
        score = calc_toxicity_score(
            s.get("reactants_smiles", []),
            s.get("conditions", {}).get("solvent", "inconnu"),
            tox_index
        )
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.5


# Registre central : associe un nom de critère à sa fonction de calcul
CRITERIA_REGISTRY = {
    "atom_economy": {
        "fn": compute_atom_economy,
        "description": "Économie atomique (chimie verte)"
    },
    "steps": {
        "fn": compute_steps,
        "description": "Nombre d'étapes (moins = mieux)"
    },
    "yield": {
        "fn": compute_yield,
        "description": "Rendement global estimé"
    },
    "e_factor": {
        "fn": compute_e_factor,
        "description": "E-factor (déchets, moins = mieux)"
    },
    "toxicity": {
        "fn": compute_toxicity,
        "description": "Sécurité (faible toxicité)"
    },
}


# CLASSEMENT

# Deux méthodes disponibles, choisies via la variable SCORING_METHOD en haut.
#
# --- MÉTHODE 1 : WEIGHTED (somme pondérée) ---
# Chaque critère reçoit un poids décroissant selon sa position dans la liste :
#   priorité 1 → poids 1/1² = 1.0
#   priorité 2 → poids 1/2² = 0.25
#   priorité 3 → poids 1/3² = 0.11
# Le score est la somme pondérée de tous les critères, normalisée.
# Avantage : simple. Inconvénient : le critère 1 peut écraser les autres.
#
# --- MÉTHODE 2 : BORDA COUNT ---
# Pour chaque critère, on classe les routes entre elles et on attribue des points :
#   - meilleure route → (N-1) points    (N = nombre de routes)
#   - 2ème route      → (N-2) points
#   - ...
# Ces points sont ensuite pondérés selon la priorité du critère.
# Avantage : une route moyenne sur tous les critères peut battre une route
# excellente sur un seul critère mais mauvaise sur les autres.
# C'est la méthode qui favorise le "meilleur compromis".

def compute_weights(criteria: list) -> dict:
    """Poids décroissants en 1/i² pour chaque critère selon sa position."""
    raw = {c: 1.0 / (i + 1) ** 2 for i, c in enumerate(criteria)}
    total = sum(raw.values())
    return {c: w / total for c, w in raw.items()}  # normalisé pour que la somme = 1


def rank_weighted(routes: list, criteria: list, tox_index: dict) -> list:
    """
    Classement par somme pondérée.
    Calcule un score entre 0 et 1 pour chaque route puis les trie.
    """
    weights = compute_weights(criteria)
    scored = []

    for route in routes:
        details = {}
        total = 0.0

        for c in criteria:
            fn = CRITERIA_REGISTRY[c]["fn"]
            raw = fn(route, tox_index)
            w = weights[c]
            details[c] = {"raw": round(raw, 4), "weight": round(w, 4), "weighted": round(raw * w, 4)}
            total += raw * w

        scored.append((round(total, 4), details, route))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


def rank_borda(routes: list, criteria: list, tox_index: dict) -> list:
    """
    Classement Borda Count pondéré.

    Principe : pour chaque critère, on attribue des points de position
    (N-1 à la 1ère, N-2 à la 2ème, etc.), puis on pondère ces points
    selon la priorité du critère. La route avec le plus de points totaux gagne.

    Avantage par rapport à la somme pondérée : une route constamment bonne
    sur tous les critères peut battre une route exceptionnelle sur un seul
    critère mais médiocre sur les autres.
    """
    weights = compute_weights(criteria)
    n = len(routes)

    # Étape 1 : calculer le score brut de chaque critère pour chaque route
    raw_scores = []
    all_details = []
    for route in routes:
        details = {}
        for c in criteria:
            fn = CRITERIA_REGISTRY[c]["fn"]
            details[c] = round(fn(route, tox_index), 4)
        raw_scores.append(details)
        all_details.append({})

    # Étape 2 : transformer les scores en points de position pour chaque critère
    for c in criteria:
        # On trie les routes selon ce critère (du meilleur au moins bon)
        order = sorted(range(n), key=lambda i: raw_scores[i][c], reverse=True)

        for rank_pos, route_idx in enumerate(order):
            borda_pts = (n - 1) - rank_pos               # meilleure route → n-1 points
            weighted_pts = borda_pts * weights[c]         # pondéré par la priorité du critère
            all_details[route_idx][c] = {
                "raw": raw_scores[route_idx][c],
                "rank": rank_pos + 1,                     # position dans ce critère (1 = meilleur)
                "borda_pts": borda_pts,
                "weighted_pts": round(weighted_pts, 4)
            }

    # Étape 3 : sommer les points pondérés pour chaque route
    scored = []
    for i, route in enumerate(routes):
        total_pts = sum(all_details[i][c]["weighted_pts"] for c in criteria)
        scored.append((round(total_pts, 4), all_details[i], route))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


# AFFICHAGE DES RÉSULTATS

def display_results(top_routes: list, criteria: list, method: str):
    """Affiche les résultats dans le terminal. Remplaceable par une interface web."""

    print("\n" + "=" * 62)
    print(f"  MEILLEURES ROUTES — méthode : {method.upper()}")
    print("=" * 62)

    if not top_routes:
        print("  Aucune route n'a passé le filtrage.")
        print("  → Enrichissez le dataset ou vérifiez les SMILES.")
        return

    for rank, (score, details, route) in enumerate(top_routes, 1):
        name = route.get("route_name", f"Route {rank}")
        n_steps = len(route.get("steps", []))

        print(f"\n{'─' * 55}")
        print(f"  Rang {rank} — {name}")
        print(f"{'─' * 55}")
        print(f"  Score total : {score:.4f}   |   Étapes : {n_steps}")
        print(f"\n  Détail par critère :")

        for c in criteria:
            desc = CRITERIA_REGISTRY[c]["description"]
            d = details.get(c, {})
            if method == "borda":
                print(f"    • {desc:<38} score={d.get('raw',0):.3f}  rang={d.get('rank','?')}  pts={d.get('weighted_pts',0):.3f}")
            else:
                print(f"    • {desc:<38} score={d.get('raw',0):.3f}  ×{d.get('weight',0):.3f}  ={d.get('weighted',0):.4f}")

        print(f"\n  Étapes de synthèse :")
        for i, step in enumerate(route.get("dataset_steps", []), 1):
            name_step = step.get("reaction_name", "?")
            rtype = step.get("reaction_type", "?")
            yld = step.get("yield_percent", "?")
            solv = step.get("conditions", {}).get("solvent", "?")
            print(f"    {i}. {name_step} ({rtype})")
            print(f"       Rendement : {yld}%   Solvant : {solv}")

    print(f"\n{'=' * 62}\n")


# POINT D'ENTRÉE PRINCIPAL

def find_best_routes(
    target_smiles: str,
    criteria_priority: list,
    dataset_path: str = "reaction_dataset.json",
    toxicity_path: str = "toxicity_dataset.json",
    config_path: str = "config.yml",
    top_n: int = 3,
    method: str = SCORING_METHOD
) -> list:
    """
    Fonction principale — seul point d'entrée à appeler depuis l'extérieur.

    Paramètres :
      target_smiles     : SMILES de la molécule cible à synthétiser
      criteria_priority : liste de critères dans l'ordre de priorité décroissante
                          Disponibles : "atom_economy", "steps", "yield",
                                        "e_factor", "toxicity"
      dataset_path      : chemin vers reaction_dataset.json
      toxicity_path     : chemin vers toxicity_dataset.json
      config_path       : chemin vers config.yml d'AiZynthFinder
      top_n             : nombre de routes à retourner (3 par défaut)
      method            : "weighted" (somme pondérée) ou "borda" (meilleur compromis)

    Retourne la liste des meilleures routes avec leurs scores.
    """
    print("\n[Pipeline] Démarrage")
    print(f"  Cible    : {target_smiles}")
    print(f"  Critères : {criteria_priority}")
    print(f"  Méthode  : {method}")

    # Vérification que tous les critères demandés existent bien dans le registre
    unknown = [c for c in criteria_priority if c not in CRITERIA_REGISTRY]
    if unknown:
        raise ValueError(
            f"Critères inconnus : {unknown}\n"
            f"Disponibles : {list(CRITERIA_REGISTRY.keys())}"
        )

    # Chargement des deux datasets
    print("\n[Étape 1/4] Chargement des datasets...")
    dataset = load_reaction_dataset(dataset_path)
    tox_index = load_toxicity_dataset(toxicity_path)

    # Recherche rétrosynthétique
    print("\n[Étape 2/4] Recherche des routes (AiZynthFinder)...")
    raw_routes = run_aizynthfinder(target_smiles, config_path)
    routes = [adapt_route(r) for r in raw_routes]  # normalisation du format
    print(f"  {len(routes)} routes récupérées")

    # Filtrage : on ne garde que les routes couvertes par le dataset
    print("\n[Étape 3/4] Filtrage par le dataset...")
    valid_routes = filter_routes(routes, dataset)

    if not valid_routes:
        print("\n  Aucune route validée — pipeline arrêté.")
        return []

    # Calcul des scores et classement
    print("\n[Étape 4/4] Scoring et classement...")
    if method == "borda":
        top_routes = rank_borda(valid_routes, criteria_priority, tox_index)[:top_n]
    else:
        top_routes = rank_weighted(valid_routes, criteria_priority, tox_index)[:top_n]

    display_results(top_routes, criteria_priority, method)
    return top_routes
