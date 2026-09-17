from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from quinte_candidate import prepare, predict_scores

st.set_page_config(page_title='HorseProno Quinté V1', page_icon='🏇', layout='wide')
st.title('🏇 HorseProno Quinté V1')
st.caption('Challenger expérimental spécialisé sur les supports Quinté — classement Top 7.')

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'quinte_v1_artifact.json'

REQUIRED_HISTORY = {
    'race_id', 'race_date', 'horse_number', 'horse_name', 'finish_position',
    'odds', 'distance', 'draw', 'weight', 'age', 'discipline', 'sex',
    'jockey', 'trainer'
}
REQUIRED_RACE = {
    'race_id', 'race_date', 'horse_number', 'horse_name', 'odds', 'distance',
    'draw', 'weight', 'age', 'discipline', 'sex', 'jockey', 'trainer'
}


def check_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} : colonnes manquantes : {', '.join(missing)}")


@st.cache_data(show_spinner=False)
def load_model(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


st.info(
    "Cette version utilise le modèle Quinté V1 déjà entraîné. "
    "Charge l'historique validé puis le CSV des partants de la course à classer."
)

c1, c2 = st.columns(2)
with c1:
    history_file = st.file_uploader('1. Historique validé (CSV)', type=['csv'], key='history')
with c2:
    race_file = st.file_uploader('2. Course Quinté à pronostiquer (CSV)', type=['csv'], key='race')

if history_file is None or race_file is None:
    st.stop()

try:
    history = pd.read_csv(history_file)
    race = pd.read_csv(race_file)

    check_columns(history, REQUIRED_HISTORY, 'Historique')
    check_columns(race, REQUIRED_RACE, 'Course')

    history['race_date'] = pd.to_datetime(history['race_date'], errors='raise').dt.normalize()
    race['race_date'] = pd.to_datetime(race['race_date'], errors='raise').dt.normalize()

    if 'is_non_runner' in race.columns:
        race = race[~race['is_non_runner'].fillna(False).astype(bool)].copy()

    if race.duplicated(['race_id', 'horse_number']).any():
        raise ValueError('La course contient un doublon race_id + horse_number.')

    artifact = load_model(str(MODEL_PATH))
    features = prepare(history, race)
    race['quinte_v1_score'] = predict_scores(artifact, features)
    race = race.sort_values(
        ['race_id', 'quinte_v1_score', 'horse_number'],
        ascending=[True, False, True],
    ).copy()
    race['quinte_rank'] = race.groupby('race_id').cumcount() + 1

    top7 = race[race['quinte_rank'] <= 7].copy()

    st.subheader('🎯 Sélection Quinté V1 — Top 7')
    st.markdown('### ' + ' - '.join(top7['horse_number'].astype(int).astype(str).tolist()))

    show_cols = [
        'quinte_rank', 'horse_number', 'horse_name', 'quinte_v1_score',
        'odds', 'jockey', 'trainer'
    ]
    available = [c for c in show_cols if c in race.columns]
    display = race[available].copy()
    if 'quinte_v1_score' in display.columns:
        display['quinte_v1_score'] = display['quinte_v1_score'].round(4)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button(
        'Télécharger le pronostic CSV',
        data=race.to_csv(index=False).encode('utf-8'),
        file_name='pronostic_quinte_v1.csv',
        mime='text/csv',
    )

    with st.expander('Informations modèle'):
        st.json({
            'model_name': artifact.get('model_name'),
            'status': artifact.get('status'),
            'target': artifact.get('target'),
            'training_supports': artifact.get('training_supports'),
            'training_end': artifact.get('training_end'),
            'C': artifact.get('C'),
        })

except Exception as exc:
    st.error(f'Erreur : {exc}')
    st.exception(exc)
