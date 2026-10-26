# Asesor LOPDP para MiPymes

Aplicación Streamlit de orientación sobre la Ley Orgánica de Protección de Datos Personales del Ecuador.

## Archivo principal
`03_chatbot_lopdp.py`

## Variables requeridas
Configurar en Streamlit Community Cloud > Advanced settings > Secrets:

```toml
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
GROQ_API_KEY = "..."
GROQ_MODEL = "qwen/qwen3.8-27b"
```

No subir `.env` ni claves privadas al repositorio.
