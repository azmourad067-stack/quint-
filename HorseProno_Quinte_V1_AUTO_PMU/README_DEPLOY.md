# HorseProno Quinté V1 — programme PMU automatique

## Fonctionnement

1. `streamlit_app.py` récupère le programme PMU pour la date choisie.
2. Il conserve uniquement la course contenant le code officiel `E_QUINTE_PLUS`.
3. Il récupère automatiquement `/R{n}/C{n}/participants`.
4. Il convertit les partants au format attendu par Quinté V1.
5. Il charge `validated_history.csv` puis, si des secrets Supabase sont présents, ajoute les nouvelles courses historiques validées antérieures à la course cible.
6. Il produit le classement et le Top 7.

Aucun CSV de course n'est demandé à l'utilisateur.

## Streamlit Community Cloud

Main file path :

`horse_prono_v2/streamlit_app.py`

Si le dossier du dépôt porte un autre nom, adaptez simplement le chemin.

## Secrets Supabase (facultatifs mais recommandés pour les jours postérieurs à l'historique embarqué)

Dans Streamlit Cloud > Manage app > Settings > Secrets :

```toml
SUPABASE_URL = "https://VOTRE-PROJET.supabase.co"
SUPABASE_KEY = "VOTRE_CLE_DE_LECTURE"
```

Utiliser de préférence une clé limitée à la lecture des tables nécessaires. Ne jamais committer une vraie clé secrète dans GitHub.
