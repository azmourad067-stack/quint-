# HorseProno Quinté V4.6 — Streamlit

Main file path : `streamlit_app.py`

V4.6 conserve exactement la shortlist Top7 de V4.2/V4.3. Sur le PLAT, elle part du reclassement Winner Top3 de V4.4 puis ajoute un signal expérimental de dynamique des cotes vers T-2 : momentum, volatilité et confiance.

## Déploiement

1. Déposer tous les fichiers du dossier à la racine du dépôt GitHub.
2. Dans Streamlit Community Cloud, choisir `streamlit_app.py` comme Main file path.
3. Les secrets Supabase sont optionnels pour lancer le moteur. Pour conserver les snapshots entre les sessions, ajouter dans les Secrets Streamlit :

```toml
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_KEY = "<publishable/anon read key>"
SUPABASE_SECRET_KEY = "<server-side secret/service-role key>"
```

Ne jamais committer `.streamlit/secrets.toml` ou la clé serveur dans GitHub.

Voir `README_V45.md` pour le fonctionnement et `supabase_v45_market_dynamics.sql` pour la structure attendue côté Supabase.
