"""
Injects the clinical design system CSS into the Streamlit app.
Implements the design tokens from UI_UX_Brief.md: navy/teal/amber palette,
Source Serif 4 + IBM Plex Sans/Mono typography, hairline borders, no shadows.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --navy-900: #0F2238;
    --navy-700: #1C3A5E;
    --teal-600: #2A9D8F;
    --amber-600: #E76F51;
    --gold-500: #D4A24E;
    --warm-grey-50: #F7F6F4;
    --warm-grey-100: #EFEDE9;
    --charcoal-900: #1C1C1E;
    --charcoal-500: #6B6B6E;
    --border-hairline: #DDDAD3;
}

html, body {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--charcoal-900);
}

.stApp {
    background-color: var(--warm-grey-50);
}

h1, h2, h3 {
    font-family: 'Source Serif 4', serif !important;
    font-weight: 600 !important;
    color: var(--navy-900) !important;
    letter-spacing: -0.01em;
}

h1 { font-size: 32px !important; }
h2 { font-size: 24px !important; }
h3 { font-size: 18px !important; font-family: 'IBM Plex Sans', sans-serif !important; }

/* Sidebar navy-900 background */
[data-testid="stSidebar"] {
    background-color: var(--navy-900) !important;
}
[data-testid="stSidebar"] * {
    color: #E8EBF0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
}
[data-testid="stSidebar"] input {
    color: var(--charcoal-900) !important;
}
[data-testid="stSidebar"] div[data-baseweb="input"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid var(--border-hairline) !important;
}

/* Global Select Box styling (ensures high-contrast light background inputs) */
div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] div,
div[class*="stSelectbox"] div {
    background-color: #FFFFFF !important;
    border-color: var(--border-hairline) !important;
}

div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] span,
div[class*="stSelectbox"] span {
    color: var(--charcoal-900) !important;
}

div[data-baseweb="select"] svg,
div[data-testid="stSelectbox"] svg,
div[class*="stSelectbox"] svg {
    fill: var(--charcoal-900) !important;
}

/* Style the dropdown menu options list */
ul[role="listbox"] {
    background-color: #FFFFFF !important;
}
ul[role="listbox"] li {
    color: var(--charcoal-900) !important;
    background-color: #FFFFFF !important;
}
ul[role="listbox"] li:hover {
    background-color: var(--warm-grey-100) !important;
}

/* Sidebar Specific Input Text Colors override */
[data-testid="stSidebar"] div[data-baseweb="select"] div,
[data-testid="stSidebar"] div[data-testid="stSelectbox"] div {
    background-color: #FFFFFF !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-testid="stSelectbox"] span {
    color: var(--charcoal-900) !important;
}

.clinical-card {
    background: #FFFFFF;
    border: 1px solid var(--border-hairline);
    border-radius: 4px;
    padding: 24px;
    margin-bottom: 16px;
}

.app-header {
    border-bottom: 1px solid var(--border-hairline);
    padding-bottom: 16px;
    margin-bottom: 24px;
}
.app-header .eyebrow {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: var(--charcoal-500);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 4px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.risk-badge.low {
    color: var(--teal-600);
    background: rgba(42, 157, 143, 0.10);
    border: 1px solid var(--teal-600);
}
.risk-badge.medium {
    color: var(--gold-500);
    background: rgba(212, 162, 78, 0.12);
    border: 1px solid var(--gold-500);
}
.risk-badge.high {
    color: var(--amber-600);
    background: rgba(231, 111, 81, 0.10);
    border: 1px solid var(--amber-600);
}

.risk-score-display {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 36px;
    line-height: 40px;
    color: var(--navy-900);
}

.risk-score-caption {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    color: var(--charcoal-500);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.factor-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-hairline);
    font-size: 14px;
}
.factor-row:last-child { border-bottom: none; }
.factor-rank {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--charcoal-500);
    margin-right: 8px;
}

.stButton > button {
    background-color: var(--navy-900);
    color: white;
    border-radius: 4px;
    border: none;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    padding: 10px 24px;
}
.stButton > button:hover {
    background-color: var(--navy-700);
    color: white;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--navy-900) !important;
}

.champion-row {
    border-left: 3px solid var(--teal-600);
    padding-left: 12px;
}

hr {
    border-color: var(--border-hairline) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border-hairline);
    border-radius: 4px;
}

/* Styled table support */
table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    margin-bottom: 24px;
}
th {
    background-color: var(--warm-grey-100) !important;
    color: var(--charcoal-900) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid var(--border-hairline) !important;
    text-align: left !important;
}
td {
    padding: 12px 16px !important;
    border-bottom: 1px solid var(--border-hairline) !important;
    color: var(--charcoal-900) !important;
    height: 48px !important;
}
tr:hover {
    background-color: var(--warm-grey-50) !important;
}
/* Right-align numeric columns and use IBM Plex Mono */
td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6),
th:nth-child(3), th:nth-child(4), th:nth-child(5), th:nth-child(6) {
    text-align: right !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
}
/* Champion row highlight (first row) */
tr:nth-child(1) td {
    border-left: 3px solid var(--teal-600) !important;
}
</style>
"""
