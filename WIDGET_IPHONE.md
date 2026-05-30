# 📱 Widget iPhone — récap du portefeuille

Affiche le total + la répartition (Silver / PEA / Cash / Crypto) + le signal
d'achat XAG directement sur ton **écran d'accueil** et ton **écran de
verrouillage**, comme le widget Mercedes. Le widget se rafraîchit tout seul
(cadence décidée par iOS, ~15-30 min).

## Pourquoi pas directement la PWA ?

iOS **n'autorise aucun widget depuis une PWA ou un site web** (limite d'Apple).
La seule voie sans développer une app native est l'app gratuite **Scriptable**,
qui exécute un petit script JavaScript et l'affiche comme widget.

## Comment ça marche

```
App Streamlit (onglet Dashboard)
   │  à chaque ouverture → pousse le récap
   ▼
Gist secret GitHub  (widget.json, URL non devinable, pas de token requis)
   ▲
   │  lecture de l'URL raw
Widget Scriptable (iPhone, accueil + verrouillage)
```

- Le gist est **secret** : non listé, accessible uniquement via son URL (un
  hash de 32 caractères impossible à deviner). Aucun token n'est stocké sur
  l'iPhone.
- Les données sont celles de ton **dernier passage dans l'app** (pas de
  recalcul live en arrière-plan).

## Installation (à faire une seule fois)

### 1. Côté app
- Ouvre l'app → onglet **📊 Dashboard** → déplie **📱 Widget iPhone**.
- Copie l'**URL** affichée (commence par `https://gist.githubusercontent.com/...`).
- Si tu vois une erreur de scope `gist` : régénère ton `GITHUB_TOKEN` sur
  GitHub en cochant la case **gist**, puis remplace-le dans les secrets
  Streamlit.

### 2. Côté iPhone
1. Installe **Scriptable** (App Store, gratuit).
2. Ouvre Scriptable → **+** → colle le contenu de [`scriptable_widget.js`](scriptable_widget.js).
3. Remplace `WIDGET_URL = "REMPLACE_PAR_TON_URL"` par l'URL copiée.
4. Renomme le script (ex. **Patrimoine**) puis valide (✓).
5. Ajoute le widget :
   - **Écran d'accueil** : appui long sur le fond → **+** → **Scriptable** →
     taille **Moyen** (recommandé) → ajoute → appui long sur le widget →
     *Modifier le widget* → **Script** = Patrimoine.
   - **Écran de verrouillage** : appui long sur l'écran verrouillé →
     *Personnaliser* → écran verrouillé → zone widgets → **+** →
     **Scriptable** → taille **Rectangulaire** → choisis le script.

## Tailles supportées

| Famille                       | Contenu |
|-------------------------------|---------|
| Petit (accueil)               | Total + variation du jour |
| Moyen / Grand (accueil)       | Total + variation + signal + Silver/PEA/Cash/Crypto |
| Rectangulaire (verrouillage)  | Total + variation + signal (compact) |
| Inline (au-dessus de l'heure) | Total + variation sur une ligne |

## Rafraîchissement

- Le script demande un refresh ~15 min, mais **iOS impose sa propre cadence**
  (souvent 15-30 min, parfois plus). C'est une limite système, identique pour
  tous les widgets tiers.
- Pour forcer une mise à jour des **données** (pas seulement de l'affichage) :
  ouvre l'app, onglet Dashboard, bouton **🔄 Mettre à jour le widget
  maintenant**.

## Confidentialité

- Le gist est secret (non indexé, non listé sur ton profil).
- Il ne contient que des montants agrégés, aucune donnée d'identification ni
  clé d'API.
- Aucun token GitHub n'est présent dans le script Scriptable.
