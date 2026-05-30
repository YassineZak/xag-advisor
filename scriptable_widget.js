// ============================================================================
//  XAG Advisor — Widget portefeuille pour iPhone (Scriptable)
// ----------------------------------------------------------------------------
//  Installation (1 fois) :
//   1. Installe l'app « Scriptable » (gratuite, App Store).
//   2. Ouvre Scriptable → « + » → colle tout ce fichier dans le script.
//   3. Renseigne WIDGET_URL ci-dessous avec l'URL affichée dans l'app
//      (onglet 📊 Dashboard → « 📱 Widget iPhone »).
//   4. Renomme le script (ex. « Patrimoine »), tape ✓.
//   5. Écran d'accueil OU verrouillage → appui long → « + » → Scriptable →
//      choisis la taille (Moyen recommandé sur l'accueil ;
//      « Rectangulaire » sur l'écran de verrouillage) → édite le widget →
//      Script = « Patrimoine ».
//
//  Le widget se rafraîchit automatiquement (iOS décide de la cadence réelle,
//  ~toutes les 15-30 min). Les données reflètent ton dernier passage dans l'app.
// ============================================================================

const WIDGET_URL = "REMPLACE_PAR_TON_URL";

const C = {
  bg1:   new Color("#1e1e2e"),
  bg2:   new Color("#16213e"),
  gold:  new Color("#fbbf24"),
  muted: new Color("#94a3b8"),
  faint: new Color("#64748b"),
  white: new Color("#e2e8f0"),
  green: new Color("#22c55e"),
  red:   new Color("#ef4444"),
};

// ── Données ─────────────────────────────────────────────────────────────────
async function fetchData() {
  const req = new Request(WIDGET_URL);
  req.headers = { "Cache-Control": "no-cache" };
  return await req.loadJSON();
}

function fmtEur(v, dec = 2) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return Number(v).toLocaleString("fr-FR", {
    minimumFractionDigits: dec, maximumFractionDigits: dec,
  }) + " €";
}

function isUp(pct) {
  return !(pct && String(pct).trim().startsWith("-"));
}

// ── Widget accueil (small / medium / large) ──────────────────────────────────
function poche(stack, emoji, label, value) {
  const row = stack.addStack();
  row.layoutHorizontally();
  row.centerAlignContent();
  const e = row.addText(emoji + " ");
  e.font = Font.systemFont(12);
  const l = row.addText(label);
  l.font = Font.mediumSystemFont(12);
  l.textColor = C.muted;
  row.addSpacer();
  const v = row.addText(fmtEur(value));
  v.font = Font.boldSystemFont(12);
  v.textColor = C.white;
}

function buildHome(data, family) {
  const w = new ListWidget();
  const grad = new LinearGradient();
  grad.colors = [C.bg1, C.bg2];
  grad.locations = [0, 1];
  w.backgroundGradient = grad;
  w.setPadding(14, 16, 14, 16);

  // En-tête : libellé + heure de mise à jour
  const head = w.addStack();
  head.layoutHorizontally();
  const ht = head.addText("PATRIMOINE");
  ht.font = Font.semiboldSystemFont(9);
  ht.textColor = C.faint;
  head.addSpacer();
  const upd = head.addText("↻ " + (data.updated_at || ""));
  upd.font = Font.systemFont(9);
  upd.textColor = C.faint;

  w.addSpacer(5);

  // Total
  const total = w.addText(fmtEur(data.total));
  total.font = Font.boldSystemFont(family === "small" ? 22 : 27);
  total.textColor = C.gold;

  // Variation jour + signal
  const sub = w.addStack();
  sub.layoutHorizontally();
  sub.centerAlignContent();
  if (data.day_pct && data.day_pct !== "—") {
    const up = isUp(data.day_pct);
    const dv = sub.addText((data.day_var || "") + "  " + data.day_pct);
    dv.font = Font.mediumSystemFont(10);
    dv.textColor = up ? C.green : C.red;
  }
  sub.addSpacer();
  if (data.signal_label && data.signal_label !== "—" && family !== "small") {
    const pill = sub.addText(data.signal_label);
    pill.font = Font.boldSystemFont(9);
    pill.textColor = new Color(data.signal_color || "#94a3b8");
  }

  // Petit widget : on s'arrête au total + variation
  if (family === "small") return w;

  w.addSpacer(9);

  poche(w, "🥈", "Silver", data.silver);
  w.addSpacer(4);
  poche(w, "📈", "PEA", data.etf);
  w.addSpacer(4);
  poche(w, "💵", "Cash", data.cash);
  w.addSpacer(4);
  poche(w, "₿", "Crypto", data.crypto);

  return w;
}

// ── Widget écran de verrouillage (accessory) ─────────────────────────────────
function buildAccessory(data, family) {
  const w = new ListWidget();

  if (family === "accessoryInline") {
    // Une seule ligne (au-dessus de l'heure)
    w.addText("💰 " + fmtEur(data.total, 0) +
              (data.day_pct && data.day_pct !== "—" ? "  " + data.day_pct : ""));
    return w;
  }

  // accessoryRectangular / accessoryCircular → 2-3 lignes compactes, monochrome
  const t1 = w.addText("Patrimoine");
  t1.font = Font.semiboldSystemFont(11);
  const t2 = w.addText(fmtEur(data.total, 0));
  t2.font = Font.boldSystemFont(16);
  if (data.day_pct && data.day_pct !== "—") {
    const t3 = w.addText("Jour " + data.day_pct +
                         (data.signal_label && data.signal_label !== "—"
                            ? " · " + data.signal_label : ""));
    t3.font = Font.systemFont(10);
  }
  return w;
}

// ── Assemblage ────────────────────────────────────────────────────────────────
let widget;
const family = config.widgetFamily; // small | medium | large | accessory*
try {
  const data = await fetchData();
  if (family && family.startsWith("accessory")) {
    widget = buildAccessory(data, family);
  } else {
    widget = buildHome(data, family || "medium");
  }
} catch (e) {
  widget = new ListWidget();
  widget.backgroundColor = C.bg1;
  const t = widget.addText("⚠️ Widget indisponible");
  t.textColor = C.muted; t.font = Font.systemFont(12);
  const d = widget.addText(String(e));
  d.textColor = C.faint; d.font = Font.systemFont(8);
}

// iOS limite la fréquence réelle, mais on suggère un refresh ~15 min
widget.refreshAfterDate = new Date(Date.now() + 15 * 60 * 1000);

if (config.runsInWidget) {
  Script.setWidget(widget);
} else {
  // Aperçu quand on lance le script manuellement dans Scriptable
  if (family && family.startsWith("accessory")) widget.presentSmall();
  else widget.presentMedium();
}
Script.complete();
