# UI/UX Brief
## Patient Readmission Risk Predictor (PRRP)
**Version:** 1.0

---

## 1. Design Direction

This is a clinical decision-support tool, not a consumer product. The visual language draws from clinical monitoring instruments and medical journal typography — calm, precise, authoritative. It should look like something a hospital's data team built and a clinician would trust at a glance, not like a hackathon dashboard.

**Explicitly avoided:** cream/terracotta warm-minimalist look, dark-mode-with-neon-accent look, gradient hero banners, emoji icons, rounded bubble buttons, decorative illustrations.

---

## 2. Design Tokens

### 2.1 Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `navy-900` | `#0F2238` | Primary brand, sidebar background, headers |
| `navy-700` | `#1C3A5E` | Secondary surfaces, hover states |
| `teal-600` | `#2A9D8F` | Low-risk indicator, positive actions, links |
| `amber-600` | `#E76F51` | High-risk indicator ONLY — never decorative |
| `gold-500` | `#D4A24E` | Medium-risk indicator |
| `warm-grey-50` | `#F7F6F4` | App background |
| `warm-grey-100` | `#EFEDE9` | Card backgrounds, dividers |
| `charcoal-900` | `#1C1C1E` | Primary text |
| `charcoal-500` | `#6B6B6E` | Secondary text, captions |
| `border-hairline` | `#DDDAD3` | All borders (1px, never shadows) |

**Rule:** Color carries clinical meaning. Teal/gold/amber are reserved exclusively for risk-tier indicators. No other UI element uses these hues — this keeps risk signals unambiguous.

### 2.2 Typography

| Role | Typeface | Weight | Usage |
|------|----------|--------|-------|
| Display / Headers | Source Serif 4 | 600 | Page titles, section headers, patient name |
| Body / UI | IBM Plex Sans | 400 / 500 | Form labels, body text, buttons, nav |
| Data / Numerals | IBM Plex Mono | 500 | Risk scores, percentages, metric tables |

**Type scale:**
```
H1   32px / 40px   Source Serif 4, 600
H2   24px / 32px   Source Serif 4, 600
H3   18px / 26px   IBM Plex Sans, 500
Body 14px / 22px   IBM Plex Sans, 400
Caption 12px / 18px IBM Plex Sans, 400, charcoal-500
Data-lg 36px / 40px IBM Plex Mono, 500   (risk score display)
Data-sm 14px / 20px IBM Plex Mono, 500   (table figures)
```

Why this pairing: Source Serif 4 gives headers the gravity of a clinical journal rather than a tech product. IBM Plex Sans was designed by IBM specifically for dense data interfaces and stays legible at small sizes. IBM Plex Mono prevents numeral jitter when scores update live — critical for a tool clinicians will glance at repeatedly.

### 2.3 Spacing & Layout Grid
```
Base unit: 8px
Spacing scale: 8 / 16 / 24 / 32 / 48 / 64

Sidebar width:    240px (fixed)
Content max-width: 1200px
Card padding:      24px
Border radius:     4px (sharp, clinical — not 12-16px bubble style)
Border:            1px solid border-hairline (no box-shadows anywhere)
```

---

## 3. Layout Concept

```
┌────────────┬──────────────────────────────────────────────┐
│            │  Header: Patient Readmission Risk Predictor    │
│  SIDEBAR   │  ─────────────────────────────────────────    │
│  (navy-900)│                                                │
│            │  ┌─────────────────┐  ┌─────────────────────┐ │
│ ● Score    │  │  Patient Form     │  │  Risk Gauge          │ │
│   Patient  │  │  (left, 60%)      │  │  (right, 40%)        │ │
│ ● Bulk     │  │                   │  │                      │ │
│   Upload   │  │  Age bracket      │  │   ╭───────────╮      │ │
│ ● Model    │  │  Time in hosp.    │  │   │  72%      │      │ │
│   Compare  │  │  Procedures       │  │   │  HIGH     │      │ │
│ ● About    │  │  Medications      │  │   ╰───────────╯      │ │
│            │  │  A1C result       │  │   [calibrated dial]  │ │
│            │  │  Insulin status   │  │                      │ │
│            │  │                   │  │  Top risk factors:   │ │
│            │  │  [Calculate Risk] │  │  1. Time in hospital │ │
│            │  └─────────────────┘  │  2. # medications     │ │
│            │                       │  3. A1C result        │ │
│            │                       └─────────────────────┘ │
└────────────┴──────────────────────────────────────────────┘
```

**Why asymmetric 60/40 split:** the form is the primary task (data entry), the gauge is the outcome — giving it less width but high visual weight (large mono numerals, the dial) balances the composition without making input the afterthought.

---

## 4. Signature Element — The Risk Dial

The single memorable element of this interface is a **calibrated horizontal risk dial**, modeled on real clinical instrument displays (e.g., a sphygmomanometer scale) rather than a generic circular progress ring or gradient bar.

```
  0%        35%           65%              100%
  │──────────│──────────────│────────────────│
  │  LOW     │   MEDIUM     │      HIGH       │
  │  teal    │    gold      │     amber       │
  │──────────│──────────────│────────────────│
                    ▲
                  72.3%
```

- Tick marks at every 10%, labeled at 0/35/65/100 (the actual clinical thresholds)
- The needle/marker is a thin vertical line with the exact score in IBM Plex Mono beneath it
- Background bands use the risk-tier colors at 15% opacity, marker uses full-opacity tier color
- This single element ties the whole interface's color logic together and is reused identically across single-patient view, bulk-results rows (as mini sparkline version), and the leaderboard

---

## 5. Component Specifications

### 5.1 Buttons
```
Primary:    navy-900 background, white text, 4px radius, 12px/24px padding
            Hover: navy-700
Secondary:  transparent, 1px navy-900 border, navy-900 text
            Hover: warm-grey-100 background
Disabled:   warm-grey-100 background, charcoal-500 text, no border
```

### 5.2 Risk Tier Badge
```
LOW:     teal-600 text, teal-600 10%-opacity background, 1px teal-600 border
MEDIUM:  gold-500 text, gold-500 10%-opacity background, 1px gold-500 border
HIGH:    amber-600 text, amber-600 10%-opacity background, 1px amber-600 border
Shape:   4px radius pill, IBM Plex Sans 500, uppercase, letter-spacing 0.04em
```

### 5.3 Cards
```
Background: white (#FFFFFF)
Border:     1px solid border-hairline
Radius:     4px
Padding:    24px
No box-shadow — separation achieved purely through hairline borders and
the warm-grey-50 app background showing through gaps
```

### 5.4 Form Inputs
```
Height:        40px
Border:        1px solid border-hairline
Radius:        4px
Focus state:   1px navy-900 border + 2px navy-900 outline at 20% opacity
               (no glow/shadow — flat outline ring for accessibility)
Label:         12px IBM Plex Sans 500, charcoal-500, above input, 4px gap
```

### 5.5 Data Tables (Leaderboard / Bulk Results)
```
Header row:    warm-grey-100 background, IBM Plex Sans 500, uppercase, 12px
Row height:    48px
Row border:    1px solid border-hairline (bottom only)
Row hover:     warm-grey-50 background
Numeric cols:  right-aligned, IBM Plex Mono
Champion row:  left border 3px solid teal-600, star icon (★) prefix
```

---

## 6. Motion

Minimal and functional only:
- Risk gauge marker animates from 0 to final value on load (400ms ease-out) — this is the one orchestrated moment, reinforcing that a calculation just happened
- Tab switches: no transition (instant) — clinical tools should feel responsive, not decorative
- Form validation errors: 150ms fade-in, no shake/bounce
- No hover-lift effects, no parallax, no decorative micro-interactions elsewhere

---

## 7. Accessibility

- All interactive elements keyboard-navigable with visible focus ring (navy-900 outline, 2px, 20% opacity background)
- Color is never the sole indicator — risk tier badges always include text label ("HIGH"/"MEDIUM"/"LOW"), not just color
- Minimum contrast ratio 4.5:1 for all text (verified: charcoal-900 on warm-grey-50 = 14.2:1, white on navy-900 = 13.8:1)
- Form errors announced via `aria-live` region
- Reduced-motion: gauge animation disabled if `prefers-reduced-motion` is set

---

## 8. Writing & Microcopy

| Context | Copy |
|---------|------|
| Empty bulk upload | "No file uploaded yet. Upload a CSV to score multiple patients at once." |
| Invalid CSV | "3 rows are missing required fields: time_in_hospital, A1Cresult. Fix and re-upload." |
| API unreachable | "Can't reach the prediction service. Check that the API is running and try again." |
| Successful prediction | No success toast — the gauge appearing IS the confirmation |
| Champion model label | "★ Champion — selected automatically by highest PR-AUC" |

Tone: direct, clinical, no exclamation marks, no "Oops!" — errors state what happened and what to do next, nothing more.
