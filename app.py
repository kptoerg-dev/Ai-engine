import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Wir importieren die Konfiguration aus deiner Trading-Engine
from ai_trading_engine_v8 import Config, main_with_config 

# --- Seiten-Setup ---
st.set_page_config(page_title="KI Trading Assistent", layout="wide", page_icon="📈")

# --- Header ---
st.title("🤖 Dein KI Trading Assistent")
st.markdown("Willkommen! Diese App nutzt künstliche Intelligenz, um historische Daten zu analysieren und Handelsstrategien zu testen.")

st.info("💡 **Tipp für Einsteiger:** Lass die Einstellungen am Anfang einfach so, wie sie sind, und klicke unten auf 'Simulation starten'!")

# --- Layout in zwei Spalten aufteilen ---
col_links, col_rechts = st.columns([1, 2])

with col_links:
    st.header("⚙️ Grundeinstellungen")
    
    assets = st.multiselect(
        "Welche Anlagen soll die KI handeln?", 
        options=["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"],
        default=["BTC-USD", "ETH-USD"],
        help="Wähle aus, welche Kryptowährungen oder Aktien die KI in ihr Portfolio aufnehmen soll."
    )
    
    startkapital = st.number_input(
        "Dein Startkapital ($)", 
        value=10000, 
        step=1000,
        help="Mit wie viel fiktivem Geld soll die KI in der Simulation starten?"
    )
    
    risiko_level = st.select_slider(
        "Wie viel Kapital pro Trade riskieren?",
        options=["Sehr Vorsichtig", "Ausgewogen", "Aggressiv"],
        value="Ausgewogen",
        help="Vorsichtig = Wenig Einsatz pro Trade. Aggressiv = Hoher Einsatz, mehr Gewinnchance, aber auch höheres Verlustrisiko."
    )
    
    # KORRIGIERT: Der Schieberegler erlaubt jetzt realistische mathematische Werte
    ki_sicherheit = st.slider(
        "Ab welcher Sicherheit soll die KI kaufen?", 
        min_value=25.0, max_value=60.0, value=35.0, step=1.0,
        help="Dein Gewinnziel ist doppelt so groß wie der Stop-Loss. Mathematisch reicht hier schon eine Trefferchance von >33%, um Gewinn zu machen! 35% ist ein guter Startwert."
    )
    
    # Übersetzung für den Hintergrund der Trading Engine
    kelly_map = {"Sehr Vorsichtig": 0.1, "Ausgewogen": 0.3, "Aggressiv": 0.6}
    kelly_fraction = kelly_map[risiko_level]
    
    # Macht aus 35.0 für die Maschine 0.35
    min_probability = ki_sicherheit / 100.0  

    st.markdown("---")
    
    with st.expander("🛠️ Experten-Einstellungen (Nur für Profis)"):
        st.write("Hier kannst du tief in die Maschine eingreifen.")
        n_estimators = st.slider("Anzahl Entscheidungsbäume", 50, 500, 100)
        run_optuna = st.checkbox("Auto-Tuning (Optuna) aktivieren - dauert lange!")

    start_button = st.button("🚀 Simulation starten", type="primary", use_container_width=True)

with col_rechts:
    st.header("📊 Deine Ergebnisse")
    
    if start_button:
        if not assets:
            st.warning("Bitte wähle mindestens eine Anlage aus!")
        else:
            with st.spinner("Die KI durchsucht den Markt nach Mustern... Das dauert ein paar Sekunden! ☕"):
                
                # Konfiguration an die Maschine übergeben
                cfg = Config(
                    tickers=tuple(assets),
                    initial_cash=startkapital,
                    kelly_fraction=kelly_fraction,
                    min_probability=min_probability,
                    n_estimators=n_estimators,
                    run_optuna=run_optuna
                )
                
                try:
                    # Simulation starten
                    results = main_with_config(cfg) 
                    metrics = results["metrics"]
                    
                    if metrics["Trades"] == 0:
                        st.error("📉 Die KI hat 0 Trades gefunden! Stelle den Regler 'Ab welcher Sicherheit soll die KI kaufen?' weiter nach links (z.B. auf 30%), damit sie mutiger wird.")
                    else:
                        st.success(f"Tada! Die Simulation ist fertig. Die KI hat {metrics['Trades']} Trades gemacht.")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            label="Endkapital", 
                            value=f"${metrics['Endkapital']:,.2f}", 
                            delta=f"{metrics['Total Return']*100:.2f}% Rendite"
                        )
                        m2.metric(
                            label="Trefferquote", 
                            value=f"{metrics['Win Rate']*100:.1f}%"
                        )
                        m3.metric(
                            label="Größter Rückschlag", 
                            value=f"{metrics['Max Drawdown']*100:.1f}%",
                            delta_color="inverse",
                            help="Der tiefste finanzielle Einschnitt (Drawdown), den du hättest aushalten müssen."
                        )
                        
                        st.markdown("---")
                        
                        st.subheader("📈 So hätte sich dein Geld entwickelt")
                        equity = results["equity"]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=equity.index, y=equity.values, 
                            mode='lines', line=dict(color='#00ff00', width=3),
                            fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)'
                        ))
                        fig.update_layout(
                            template="plotly_dark", 
                            margin=dict(l=0, r=0, t=0, b=0),
                            yaxis_title="Kontostand in $"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Hoppla, da ist etwas schiefgelaufen: {e}")
    else:
        st.info("👈 Wähle links deine Einstellungen und starte die Simulation, um hier Ergebnisse zu sehen.")
