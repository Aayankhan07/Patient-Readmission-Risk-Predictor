import streamlit as st


def get_tier_color(tier: str) -> str:
    """Returns the design system color hex for a given risk tier."""
    if tier == "low":
        return "#2A9D8F"  # teal-600
    elif tier == "medium":
        return "#D4A24E"  # gold-500
    else:
        return "#E76F51"  # amber-600


def render_risk_card(score: float, tier: str, inference_ms: float):
    """Renders a flat, hairline-bordered clinical risk card with left highlight."""
    color = get_tier_color(tier)
    percent = score * 100

    card_html = f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #DDDAD3;
        border-left: 6px solid {color};
        border-radius: 4px;
        padding: 24px;
        margin-bottom: 25px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="color: #6B6B6E; margin: 0; font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 500;">Patient Risk Score</h4>
                <h2 style="color: #1C1C1E; margin: 8px 0; font-family: 'Source Serif 4', serif; font-size: 32px; font-weight: 600;">{percent:.1f}%</h2>
            </div>
            <div style="
                background: {color}1A; /* 10% opacity hex is 1A */
                border: 1px solid {color};
                border-radius: 4px;
                padding: 6px 16px;
                color: {color};
                font-family: 'IBM Plex Sans', sans-serif;
                font-weight: 500;
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 0.04em;
            ">
                {tier} Risk
            </div>
        </div>
        <div style="display: flex; gap: 20px; margin-top: 15px; border-top: 1px solid #DDDAD3; padding-top: 15px; font-family: 'IBM Plex Sans', sans-serif; font-size: 12px;">
            <div>
                <span style="color: #6B6B6E;">Inference Latency:</span>
                <strong style="color: #1C1C1E; font-family: 'IBM Plex Mono', monospace; font-weight: 500; margin-left: 5px;">{inference_ms:.1f} ms</strong>
            </div>
            <div>
                <span style="color: #6B6B6E;">Decision Threshold:</span>
                <strong style="color: #1C1C1E; font-family: 'IBM Plex Mono', monospace; font-weight: 500; margin-left: 5px;">35.0% / 65.0%</strong>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_gauge_chart(score: float, tier: str):
    """Renders a calibrated horizontal risk dial modeled on clinical instrument scales."""
    color = get_tier_color(tier)
    percent = score * 100

    dial_html = f"""
    <div style="
        font-family: 'IBM Plex Sans', sans-serif; 
        background: #FFFFFF; 
        border: 1px solid #DDDAD3; 
        border-radius: 4px; 
        padding: 24px; 
        margin-bottom: 25px;
    ">
        <h4 style="color: #6B6B6E; margin: 0 0 15px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 500;">Risk Classification Scale</h4>
        <div style="position: relative; height: 35px; margin-bottom: 20px; margin-top: 10px;">
            <!-- Colored background bands -->
            <div style="position: absolute; left: 0; right: 0; top: 0; bottom: 8px; display: flex; border-radius: 2px; overflow: hidden; border: 1px solid #DDDAD3;">
                <div style="flex: 35; background: rgba(42, 157, 143, 0.15); border-right: 1px dashed rgba(42, 157, 143, 0.5);"></div>
                <div style="flex: 30; background: rgba(212, 162, 78, 0.15); border-right: 1px dashed rgba(212, 162, 78, 0.5);"></div>
                <div style="flex: 35; background: rgba(231, 111, 81, 0.15);"></div>
            </div>
            <!-- 10% Tick Marks -->
            <div style="position: absolute; left: 0; right: 0; bottom: 8px; display: flex; justify-content: space-between; padding: 0 1px;">
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
                <div style="height: 6px; width: 1px; background: #DDDAD3;"></div>
            </div>
            <!-- Pointer Needle -->
            <div style="position: absolute; left: {percent}%; top: -6px; bottom: 3px; width: 2px; background: {color}; transform: translateX(-50%); z-index: 10;">
                <!-- Pointer arrowhead -->
                <div style="position: absolute; top: -14px; left: -5px; width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 8px solid {color};"></div>
            </div>
        </div>
        <!-- Labels below the scale -->
        <div style="position: relative; height: 20px; font-size: 11px; color: #6B6B6E; font-family: 'IBM Plex Mono', monospace; font-weight: 500;">
            <span style="position: absolute; left: 0;">0% (LOW)</span>
            <span style="position: absolute; left: 35%; transform: translateX(-50%); color: #D4A24E;">35%</span>
            <span style="position: absolute; left: 65%; transform: translateX(-50%); color: #E76F51;">65%</span>
            <span style="position: absolute; right: 0;">100% (HIGH)</span>
        </div>
        <!-- Large score numeric below dial -->
        <div style="text-align: center; margin-top: 15px; border-top: 1px solid #DDDAD3; padding-top: 15px;">
            <div style="font-size: 10px; text-transform: uppercase; color: #6B6B6E; font-family: 'IBM Plex Sans', sans-serif; letter-spacing: 0.04em; margin-bottom: 2px;">Calibrated Risk Index</div>
            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 500; color: #1C1C1E;">{percent:.1f}%</span>
        </div>
    </div>
    """
    st.markdown(dial_html, unsafe_allow_html=True)
