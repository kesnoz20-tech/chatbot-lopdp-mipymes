import os

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Asesor LOPDP",
    page_icon="⚖️",
    layout="centered"
)

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "qwen/qwen3.8-27b"
)

if not SUPABASE_URL:
    st.error("Falta SUPABASE_URL en el archivo .env")
    st.stop()

if not SUPABASE_KEY:
    st.error("Falta SUPABASE_KEY en el archivo .env")
    st.stop()

if not GROQ_API_KEY:
    st.error("Falta GROQ_API_KEY en el archivo .env")
    st.stop()


# ============================================================
# CLIENTES
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# CONFIGURACIÓN RAG
# ============================================================

MODEL_EMBEDDING = "intfloat/multilingual-e5-base"
EXPECTED_DIM = 768

MODEL_LLM = GROQ_MODEL

# Top final que verá el LLM y que usaremos en el benchmark.
TOP_K = 3

# Número de candidatos semánticos recuperados antes del reranking.
CANDIDATE_K = 10


@st.cache_resource
def cargar_modelo():
    return SentenceTransformer(
        MODEL_EMBEDDING
    )


modelo = cargar_modelo()


# ============================================================
# RECUPERACIÓN + RERANKING JURÍDICO
# ============================================================

def buscar_en_corpus(
    pregunta_usuario,
    top_k=TOP_K
):
    """
    1. Convierte la pregunta en embedding multilingual-E5.
    2. Recupera 10 candidatos por similitud coseno desde Supabase.
    3. Aplica un pequeño bonus de jerarquía documental.
    4. Devuelve únicamente los Top-3 finales.

    E5:
      query:   para consultas
      passage: para documentos
    """

    pregunta_limpia = pregunta_usuario.strip()

    if not pregunta_limpia:
        return "", []

    texto_query = "query: " + pregunta_limpia

    vector = modelo.encode(
        texto_query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    if vector.ndim != 1:
        raise ValueError(
            f"Forma inesperada del embedding: {vector.shape}"
        )

    if len(vector) != EXPECTED_DIM:
        raise ValueError(
            f"Embedding de {len(vector)} dimensiones; "
            f"se esperaban {EXPECTED_DIM}."
        )

    # --------------------------------------------------------
    # RECUPERACIÓN SEMÁNTICA
    # --------------------------------------------------------

    respuesta = supabase.rpc(
        "match_documents_lopdp",
        {
            "query_embedding": vector.tolist(),
            "match_count": CANDIDATE_K,
            "collection_filter": None
        }
    ).execute()

    candidatos = respuesta.data or []

    # --------------------------------------------------------
    # RERANKING JURÍDICO
    # --------------------------------------------------------
    #
    # La similitud semántica sigue siendo el criterio principal.
    # La jerarquía solo añade un pequeño bonus cuando varias
    # evidencias tienen una similitud muy cercana.
    #
    # Ley > Reglamento > Resolución > Guía
    # --------------------------------------------------------

    bonus_por_tipo = {
        "ley": 0.020,
        "reglamento": 0.015,
        "resolucion": 0.008,
        "guia": 0.000
    }

    for doc in candidatos:
        meta = doc.get("metadata") or {}

        tipo = str(
            meta.get(
                "tipo_documento",
                ""
            )
        ).lower()

        similitud = float(
            doc.get(
                "similarity",
                0
            )
        )

        bonus = bonus_por_tipo.get(
            tipo,
            0.0
        )

        doc["_rerank_score"] = (
            similitud + bonus
        )

    candidatos.sort(
        key=lambda x: x["_rerank_score"],
        reverse=True
    )

    # Top final utilizado por el chatbot y el benchmark.
    docs = candidatos[:top_k]

    # --------------------------------------------------------
    # CONSTRUCCIÓN DEL CONTEXTO PARA QWEN
    # --------------------------------------------------------

    bloques = []

    for posicion, doc in enumerate(
        docs,
        start=1
    ):
        meta = doc.get("metadata") or {}

        fuente = meta.get(
            "fuente",
            "Fuente no identificada"
        )

        tipo = meta.get(
            "tipo_documento",
            ""
        )

        articulo = meta.get(
            "articulo"
        )

        seccion = meta.get(
            "seccion"
        )

        coleccion = meta.get(
            "coleccion",
            ""
        )

        referencias = [
            f"Fuente: {fuente}"
        ]

        if tipo:
            referencias.append(
                f"Tipo: {tipo}"
            )

        if articulo:
            referencias.append(
                f"Artículo: {articulo}"
            )
        elif seccion:
            referencias.append(
                f"Sección: {seccion}"
            )

        if coleccion:
            referencias.append(
                f"Colección: {coleccion}"
            )

        chunk_id = doc.get(
            "chunk_id",
            "N/D"
        )

        contenido = doc.get(
            "content",
            ""
        )

        bloques.append(
            f"[EVIDENCIA {posicion}]\n"
            + " | ".join(referencias)
            + f"\nChunk ID: {chunk_id}"
            + f"\nTexto:\n{contenido}"
        )

    contexto = "\n\n---\n\n".join(
        bloques
    )

    return contexto, docs


# ============================================================
# GENERACIÓN CON GROQ / QWEN
# ============================================================

def generar_respuesta_groq(
    pregunta,
    contexto
):
    """
    Envía a Qwen únicamente las evidencias Top-3
    recuperadas y rerankeadas.
    """

    if not contexto.strip():
        return (
            "No se recuperó evidencia suficiente del "
            "corpus normativo para responder con seguridad."
        )

    prompt_sistema = f"""
Eres un asistente de orientación sobre protección de datos personales
para micro, pequeñas y medianas empresas ecuatorianas.

Tu respuesta debe estar fundamentada EXCLUSIVAMENTE en la evidencia
normativa recuperada que aparece al final de este mensaje.

REGLAS ESTRICTAS:

1. Usa lenguaje sencillo, claro, directo y comprensible para una MiPyme.

2. No inventes artículos, obligaciones, plazos, sanciones, excepciones,
   autoridades, procedimientos ni definiciones.

3. Si la evidencia recuperada no permite responder de manera directa y
   suficientemente sustentada, NO intentes completar la respuesta usando
   fragmentos irrelevantes ni conocimiento general. En ese caso debes iniciar
   tu respuesta exactamente con la marca:
   [[EVIDENCIA_INSUFICIENTE]]

   Después de esa marca, explica brevemente que la evidencia recuperada no
   permite responder con seguridad. Si la pregunta admite dos interpretaciones
   razonables, formula UNA sola pregunta breve de aclaración. No cites ni
   presentes como respaldo fuentes que no respondan directamente a la consulta.

4. Distingue correctamente la naturaleza de las fuentes:
   - Ley Orgánica de Protección de Datos Personales.
   - Reglamento General.
   - Resoluciones de la SPDP.
   - Guías oficiales.

5. Una guía oficial no debe utilizarse para contradecir una disposición
   de mayor jerarquía normativa.

6. Cuando la evidencia contenga un artículo, resolución o sección
   identificable, menciona esa referencia en la respuesta.

7. No afirmes que una disposición está derogada, sustituida o reformada
   salvo que la propia evidencia proporcionada lo permita establecer.

8. No presentes la respuesta como asesoría jurídica individualizada
   ni como certificación de cumplimiento.

9. Si varias evidencias son complementarias, intégralas sin atribuir
   a una fuente contenido que pertenece a otra.

10. Responde únicamente a la pregunta realizada por el usuario.

11. ALCANCE TEMÁTICO:
    Este asistente está especializado exclusivamente en protección de
    datos personales y en la normativa ecuatoriana relacionada con la LOPDP.

    Si la pregunta no está relacionada con protección de datos personales,
    privacidad, tratamiento de datos, derechos de los titulares, obligaciones
    de responsables o encargados, seguridad de datos o normativa relacionada,
    no intentes responderla utilizando conocimiento general.

    En ese caso responde únicamente:
    "Esta consulta está fuera del ámbito de este asistente. Puedo ayudarte
    con preguntas relacionadas con la LOPDP y la protección de datos
    personales en Ecuador."

12. PERTINENCIA DE LA EVIDENCIA:
    Antes de responder, verifica si los fragmentos recuperados contienen
    información que responda realmente a la pregunta. La mera presencia de
    artículos de la LOPDP no significa que sean pertinentes. Si los Top-3
    tratan asuntos distintos al consultado, aplica la regla 3 y utiliza la
    marca [[EVIDENCIA_INSUFICIENTE]].

EVIDENCIA RECUPERADA:
-----------------------
{contexto}
-----------------------
"""

    respuesta = (
        groq_client
        .chat
        .completions
        .create(
            messages=[
                {
                    "role": "system",
                    "content": prompt_sistema
                },
                {
                    "role": "user",
                    "content": pregunta
                }
            ],
            model=MODEL_LLM,
            temperature=0.3,
            max_tokens=700
        )
    )

    return (
        respuesta
        .choices[0]
        .message
        .content
    )


# ============================================================
# CONTROL DE PERTINENCIA DE LA RESPUESTA
# ============================================================

MARCA_EVIDENCIA_INSUFICIENTE = "[[EVIDENCIA_INSUFICIENTE]]"


def procesar_respuesta_evidencia(respuesta):
    """
    Retira la marca interna usada por Qwen y decide si deben mostrarse
    las evidencias recuperadas. No modifica el retrieval ni el reranking.
    """
    texto = str(respuesta or "").strip()

    if texto.startswith(MARCA_EVIDENCIA_INSUFICIENTE):
        texto_limpio = texto[len(MARCA_EVIDENCIA_INSUFICIENTE):].lstrip()
        return texto_limpio, False

    # Respaldo por si el modelo omite la marca pero expresa claramente
    # insuficiencia al inicio de la respuesta.
    inicio = texto[:500].lower()
    expresiones_insuficiencia = (
        "no es posible responder con seguridad",
        "no se recupero evidencia suficiente",
        "la evidencia recuperada no permite responder",
        "la evidencia normativa proporcionada no permite responder",
    )

    if any(frase in inicio for frase in expresiones_insuficiencia):
        return texto, False

    return texto, True


# ============================================================
# INTERFAZ
# ============================================================

# La interfaz no modifica la recuperación, el reranking ni los parámetros de generación.

PREGUNTAS_SUGERIDAS = [
    (
        "Consentimiento",
        "¿Qué requisitos debe cumplir el consentimiento para tratar datos personales?"
    ),
    (
        "Vulneración de datos",
        "¿Qué debe hacer una empresa ante una vulneración de seguridad de datos personales?"
    ),
    (
        "Eliminar datos",
        "¿Un cliente puede solicitar que elimine sus datos personales?"
    ),
    (
        "Delegado de protección",
        "¿Cuándo es obligatorio designar un Delegado de Protección de Datos?"
    ),
]


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 900px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.20);
        }

        .lopdp-hero {
            text-align: center;
            padding: 0.9rem 0 1.2rem 0;
        }

        .lopdp-hero h1 {
            margin-bottom: 0.25rem;
            font-size: 2.15rem;
        }

        .lopdp-hero p {
            opacity: 0.78;
            margin: 0 auto;
            max-width: 680px;
            line-height: 1.5;
        }

        .lopdp-badge {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 999px;
            font-size: 0.82rem;
            margin-bottom: 0.65rem;
            opacity: 0.86;
        }

        .suggestion-label {
            text-align: center;
            font-weight: 600;
            margin: 1rem 0 0.7rem 0;
        }

        .privacy-note {
            font-size: 0.82rem;
            opacity: 0.78;
            line-height: 1.45;
        }

        .legal-note {
            margin-top: 1.4rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(128, 128, 128, 0.20);
            font-size: 0.80rem;
            opacity: 0.72;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# ESTADO DE SESIÓN
# ------------------------------------------------------------

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


# ------------------------------------------------------------
# FUNCIONES DE PRESENTACIÓN
# ------------------------------------------------------------

def preparar_evidencias_para_sesion(docs):
    """Convierte los Top-3 en datos serializables para mostrarlos en el chat."""
    evidencias = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.get("metadata") or {}

        similarity = doc.get("similarity")
        rerank_score = doc.get("_rerank_score")

        evidencias.append({
            "top": i,
            "chunk_id": doc.get("chunk_id", "N/D"),
            "fuente": meta.get("fuente", "Fuente no identificada"),
            "tipo": meta.get("tipo_documento", "N/D"),
            "articulo": meta.get("articulo") or "",
            "seccion": meta.get("seccion") or "",
            "coleccion": meta.get("coleccion", "N/D"),
            "similarity": (
                float(similarity)
                if similarity is not None
                else None
            ),
            "rerank_score": (
                float(rerank_score)
                if rerank_score is not None
                else None
            ),
            "content": doc.get("content", "")
        })

    return evidencias


def mostrar_evidencias(evidencias):
    """Muestra fuentes legibles y deja los datos técnicos en un panel separado."""
    if not evidencias:
        return

    with st.expander("📚 Fuentes consultadas"):
        for ev in evidencias:
            referencia = []

            if ev.get("articulo"):
                referencia.append(
                    f"Artículo {ev['articulo']}"
                )
            elif ev.get("seccion"):
                referencia.append(
                    f"Sección {ev['seccion']}"
                )

            if ev.get("tipo") and ev.get("tipo") != "N/D":
                referencia.append(str(ev["tipo"]).capitalize())

            ref_txt = " · ".join(referencia)

            st.markdown(
                f"**{ev['top']}. {ev['fuente']}**"
                + (f"  \n{ref_txt}" if ref_txt else "")
            )

    with st.expander("🔬 Ver detalles técnicos"):
        st.caption(
            "Información para auditoría del sistema. "
            "No es necesaria para interpretar la respuesta."
        )

        for ev in evidencias:
            similarity_txt = (
                f"{ev['similarity']:.4f}"
                if ev.get("similarity") is not None
                else "N/D"
            )

            rerank_txt = (
                f"{ev['rerank_score']:.4f}"
                if ev.get("rerank_score") is not None
                else "N/D"
            )

            st.markdown(
                f"**Top {ev['top']} — {ev['chunk_id']}**  \n"
                f"Similitud coseno: `{similarity_txt}`  \n"
                f"Score reranking: `{rerank_txt}`  \n"
                f"Fuente: {ev['fuente']}  \n"
                f"Tipo: {ev['tipo']}  \n"
                f"Artículo: {ev.get('articulo') or 'N/D'}  \n"
                f"Sección: {ev.get('seccion') or 'N/D'}  \n"
                f"Colección: {ev.get('coleccion') or 'N/D'}"
            )

            if ev.get("content"):
                st.code(ev["content"])


def crear_transcripcion():
    """Genera un TXT local para que el usuario pueda conservar su conversación."""
    lineas = [
        "ASISTENTE LOPDP - TRANSCRIPCIÓN DE LA CONVERSACIÓN",
        "",
        (
            "Esta conversación fue descargada por el usuario. "
            "La aplicación no la almacena de forma permanente."
        ),
        "",
        "=" * 60,
        ""
    ]

    for mensaje in st.session_state.mensajes:
        rol = (
            "USUARIO"
            if mensaje.get("role") == "user"
            else "ASISTENTE LOPDP"
        )

        lineas.append(f"{rol}:")
        lineas.append(mensaje.get("content", ""))

        evidencias = mensaje.get("evidencias") or []
        if evidencias:
            lineas.append("")
            lineas.append("Fuentes consultadas:")

            for ev in evidencias:
                ref = ev.get("articulo") or ev.get("seccion") or "N/D"
                lineas.append(
                    f"- {ev.get('fuente', 'N/D')} | "
                    f"Art./Sec.: {ref}"
                )

        lineas.extend(["", "-" * 60, ""])

    return "\n".join(lineas)


def mensaje_error_amigable(error):
    """Evita mostrar errores técnicos de Groq a participantes de la evaluación."""
    texto = str(error)
    texto_lower = texto.lower()

    if (
        "429" in texto_lower
        or "rate_limit" in texto_lower
        or "rate limit" in texto_lower
    ):
        return (
            "El servicio está recibiendo muchas consultas en este momento. "
            "Por favor, intenta nuevamente en unos segundos."
        )

    return (
        "No fue posible completar la consulta en este momento. "
        "Por favor, intenta nuevamente."
    )


# ------------------------------------------------------------
# BARRA LATERAL
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚖️ Asistente LOPDP")
    st.caption("Orientación para MiPymes ecuatorianas")

    if st.button(
        "＋ Nueva conversación",
        use_container_width=True
    ):
        st.session_state.mensajes = []
        st.rerun()

    if st.session_state.mensajes:
        st.download_button(
            "⬇️ Descargar conversación",
            data=crear_transcripcion(),
            file_name="conversacion_lopdp.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()

    st.markdown(
        """
        <div class="privacy-note">
        <b>Privacidad de la sesión</b><br>
        El historial se mantiene únicamente mientras esta sesión permanezca
        activa. Esta versión no guarda las conversaciones de forma permanente
        en una base de datos.
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# CABECERA
# ------------------------------------------------------------

st.markdown(
    """
    <div class="lopdp-hero">
        <div class="lopdp-badge">Normativa ecuatoriana · LOPDP</div>
        <h1>⚖️ Asesor Inteligente LOPDP</h1>
        <p>
            Consulta de forma sencilla obligaciones, derechos y medidas
            relacionadas con la protección de datos personales en Ecuador.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# PREGUNTAS SUGERIDAS
# ------------------------------------------------------------

pregunta_sugerida = None

if not st.session_state.mensajes:
    st.markdown(
        '<div class="suggestion-label">Puedes comenzar con una de estas consultas</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            f"💬 {PREGUNTAS_SUGERIDAS[0][0]}",
            key="sugerida_1",
            use_container_width=True
        ):
            pregunta_sugerida = PREGUNTAS_SUGERIDAS[0][1]

        if st.button(
            f"💬 {PREGUNTAS_SUGERIDAS[2][0]}",
            key="sugerida_3",
            use_container_width=True
        ):
            pregunta_sugerida = PREGUNTAS_SUGERIDAS[2][1]

    with col2:
        if st.button(
            f"💬 {PREGUNTAS_SUGERIDAS[1][0]}",
            key="sugerida_2",
            use_container_width=True
        ):
            pregunta_sugerida = PREGUNTAS_SUGERIDAS[1][1]

        if st.button(
            f"💬 {PREGUNTAS_SUGERIDAS[3][0]}",
            key="sugerida_4",
            use_container_width=True
        ):
            pregunta_sugerida = PREGUNTAS_SUGERIDAS[3][1]


# ------------------------------------------------------------
# HISTORIAL DE LA SESIÓN
# ------------------------------------------------------------

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

        if mensaje["role"] == "assistant":
            mostrar_evidencias(
                mensaje.get("evidencias") or []
            )


# ------------------------------------------------------------
# ENTRADA DEL USUARIO
# ------------------------------------------------------------

pregunta_escrita = st.chat_input(
    "Escribe tu consulta sobre la LOPDP..."
)

pregunta = pregunta_sugerida or pregunta_escrita

if pregunta:
    pregunta = pregunta.strip()

    if pregunta:
        with st.chat_message("user"):
            st.markdown(pregunta)

        st.session_state.mensajes.append({
            "role": "user",
            "content": pregunta
        })

        with st.chat_message("assistant"):
            with st.spinner(
                "Consultando la normativa oficial..."
            ):
                try:
                    # MISMA recuperación y reranking evaluados.
                    # Toda consulta entra al RAG; no existe filtro previo
                    # basado en palabras clave.
                    contexto, docs = buscar_en_corpus(
                        pregunta
                    )

                    # Qwen conserva la regla de alcance dentro del prompt.
                    # Por tanto, una consulta ajena al dominio puede ser
                    # rechazada por el propio modelo sin bloquear preguntas
                    # jurídicas válidas antes de la recuperación.
                    respuesta_bruta = generar_respuesta_groq(
                        pregunta,
                        contexto
                    )

                    respuesta_ia, mostrar_fuentes = procesar_respuesta_evidencia(
                        respuesta_bruta
                    )

                    if mostrar_fuentes:
                        evidencias_sesion = preparar_evidencias_para_sesion(
                            docs
                        )
                    else:
                        evidencias_sesion = []

                    st.markdown(respuesta_ia)
                    if evidencias_sesion:
                        mostrar_evidencias(evidencias_sesion)

                except Exception as e:
                    respuesta_ia = mensaje_error_amigable(e)
                    evidencias_sesion = []
                    st.error(respuesta_ia)

        st.session_state.mensajes.append({
            "role": "assistant",
            "content": respuesta_ia,
            "evidencias": evidencias_sesion
        })


# ------------------------------------------------------------
# AVISO LEGAL
# ------------------------------------------------------------

st.markdown(
    """
    <div class="legal-note">
        Este asistente brinda orientación general basada en normativa oficial
        y no sustituye asesoría jurídica profesional ni certifica cumplimiento.
    </div>
    """,
    unsafe_allow_html=True
)
