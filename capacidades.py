"""Las capabilities de tu asistente: lo que vimos en clase. Cámbialas o agrega las tuyas.

Una Capability junta tools + instrucciones en una pieza que se enchufa al agente (asistente.py).
El docstring y los tipos de cada tool son lo único que el modelo lee para decidir cuándo usarla.
"""

import csv
import sqlite3
from datetime import date
from functools import cache
from typing import Literal

import chromadb
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic_ai.capabilities import Capability
from pypdf import PdfReader

from config import DATOS, settings

# ═══ S1 · Tools de lectura con APIs públicas ══════════════════════════════════════════
apis = Capability(
    id="apis_publicas",
    description=(
        "Consulta indicadores económicos, clima, clasificación actual de Fórmula 1 "
        "y cotizaciones de criptomonedas con APIs públicas gratuitas"
    ),
    instructions=(
        "Usa las APIs para obtener datos reales de economía, clima, Fórmula 1 y criptomonedas. "
        "Indica la fuente y no inventes datos."
    ),
)


@apis.tool_plain
def inflacion(pais: str, anio: int) -> dict:
    """Inflación anual de un país en un año, según el Banco Mundial.

    Args:
        pais: código ISO del país, por ejemplo 'BO' para Bolivia o 'PE' para Perú
        anio: año, por ejemplo 2024
    """
    url = f"https://api.worldbank.org/v2/country/{pais}/indicator/FP.CPI.TOTL.ZG"
    respuesta = httpx.get(url, params={"date": anio, "format": "json"}, timeout=10)
    respuesta.raise_for_status()
    datos = respuesta.json()

    # La API responde [metadatos, registros]. Si no hay dato, devolvemos un aviso que el modelo puede leer
    if len(datos) < 2 or not datos[1] or datos[1][0]["value"] is None:
        return {"error": f"No hay datos de inflación disponibles para {pais} en {anio}"}

    registro = datos[1][0]
    return {
        "pais": registro["country"]["value"],
        "anio": registro["date"],
        "inflacion_pct": round(registro["value"], 2),
        "fuente": "Banco Mundial",  # devolver la fuente evita que el modelo invente una
    }


@apis.tool_plain
def clima_cochabamba() -> dict:
    """Temperatura actual de Cochabamba, según Open Meteo."""
    params = {"latitude": -17.38, "longitude": -66.15, "current": "temperature_2m"}
    respuesta = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    respuesta.raise_for_status()
    return {"temperatura": respuesta.json()["current"]["temperature_2m"], "fuente": "Open Meteo"}


@apis.tool_plain
def clasificacion_f1(tipo: Literal["pilotos", "constructores"] = "pilotos") -> dict:
    """Clasificación actual de Fórmula 1, con los primeros diez puestos.

    Args:
        tipo: "pilotos" para la clasificación de pilotos o "constructores" para equipos.
    """
    endpoint = "driverStandings" if tipo == "pilotos" else "constructorStandings"
    url = f"https://api.jolpi.ca/ergast/f1/current/{endpoint}.json"
    try:
        respuesta = httpx.get(url, timeout=10)
        respuesta.raise_for_status()
    except httpx.HTTPError as error:
        return {"error": f"No se pudo consultar la clasificación de F1: {error}"}

    datos = respuesta.json().get("MRData", {}).get("StandingsTable", {})
    listas = datos.get("StandingsLists", [])
    if not listas:
        return {"error": "La API de F1 no tiene una clasificación disponible en este momento"}

    registros = listas[0].get("DriverStandings" if tipo == "pilotos" else "ConstructorStandings", [])
    if not registros:
        return {"error": "La API de F1 no devolvió puestos para la clasificación solicitada"}

    puestos = []
    for registro in registros[:10]:
        competidor = registro["Driver"] if tipo == "pilotos" else registro["Constructor"]
        nombre = (
            f"{competidor['givenName']} {competidor['familyName']}"
            if tipo == "pilotos"
            else competidor["name"]
        )
        puesto = {
            "posicion": int(registro["position"]),
            "nombre": nombre,
            "puntos": float(registro["points"]),
            "victorias": int(registro["wins"]),
        }
        if tipo == "pilotos" and registro.get("Constructors"):
            puesto["equipo"] = registro["Constructors"][0]["name"]
        puestos.append(puesto)

    return {
        "temporada": datos.get("season"),
        "ronda": datos.get("round"),
        "tipo": tipo,
        "clasificacion": puestos,
        "fuente": "Jolpica F1 API",
    }


_MONEDAS = {
    "btc": "btc-bitcoin",
    "bitcoin": "btc-bitcoin",
    "eth": "eth-ethereum",
    "ethereum": "eth-ethereum",
    "sol": "sol-solana",
    "solana": "sol-solana",
    "xrp": "xrp-xrp",
    "doge": "doge-dogecoin",
    "dogecoin": "doge-dogecoin",
    "ada": "ada-cardano",
    "cardano": "ada-cardano",
    "bnb": "bnb-binance-coin",
    "binance coin": "bnb-binance-coin",
}


@apis.tool_plain
def cotizacion_cripto(moneda: str) -> dict:
    """Precio actual en dólares de una criptomoneda conocida.

    Args:
        moneda: símbolo o nombre: BTC/Bitcoin, ETH/Ethereum, SOL/Solana, XRP, DOGE/Dogecoin, ADA/Cardano o BNB.
    """
    simbolo = moneda.strip().lower()
    identificador = _MONEDAS.get(simbolo)
    if identificador is None:
        return {
            "error": (
                f"No reconozco la criptomoneda {moneda!r}. "
                "Prueba con BTC, ETH, SOL, XRP, DOGE, ADA o BNB."
            )
        }

    try:
        respuesta = httpx.get(f"https://api.coinpaprika.com/v1/tickers/{identificador}", timeout=10)
        respuesta.raise_for_status()
    except httpx.HTTPError as error:
        return {"error": f"No se pudo consultar la cotización de {moneda}: {error}"}

    datos = respuesta.json()
    cotizacion = datos.get("quotes", {}).get("USD")
    if not cotizacion:
        return {"error": f"La API no tiene una cotización en USD disponible para {moneda}"}
    return {
        "nombre": datos.get("name"),
        "simbolo": datos.get("symbol"),
        "precio_usd": cotizacion.get("price"),
        "cambio_24h_pct": cotizacion.get("percent_change_24h"),
        "actualizado": datos.get("last_updated"),
        "fuente": "CoinPaprika",
    }


# ═══ S1 · Mi contador: gastos en SQLite, una tool que lee y otra que escribe ══════════
Categoria = Literal["transporte", "alimentacion", "educacion", "ocio", "salud"]
CSV = DATOS / "gastos.csv"  # 10 gastos ficticios de mayo de 2030
DB = DATOS / "gastos.db"  # la base: se crea desde el CSV la primera vez


def crear_base() -> None:
    """Crea gastos.db desde cero con los gastos del CSV."""
    DB.unlink(missing_ok=True)
    with CSV.open(encoding="utf-8") as archivo:
        filas = [(g["fecha"], g["categoria"], g["descripcion"], float(g["monto"])) for g in csv.DictReader(archivo)]
    conexion = sqlite3.connect(DB)
    with conexion:
        conexion.execute(
            "CREATE TABLE gastos (fecha TEXT, categoria TEXT, descripcion TEXT, monto REAL CHECK (monto > 0))"
        )
        conexion.executemany("INSERT INTO gastos VALUES (?, ?, ?, ?)", filas)
    conexion.close()


def hoy() -> str:
    """Instrucción dinámica: el modelo no sabe qué día es; se lo decimos en cada mensaje."""
    return f"Hoy es {date.today().isoformat()}."


# Lectura y escritura van en capabilities separadas: a otro agente le podrías dar solo la primera
consulta = Capability(
    id="consulta_gastos",
    description="Consulta gastos personales almacenados en una base de datos SQLite",
    instructions="Usa la base de datos para responder preguntas sobre los gastos. No inventes datos.",
)
registro = Capability(
    id="registro_gastos",
    description="Registra nuevos gastos en la base de datos SQLite",
    instructions=["Agrega nuevos gastos a la base de datos. Asegúrate de que los datos sean válidos.", hoy],
)


@consulta.tool_plain
def consultar_gastos(categoria: Categoria) -> dict:
    """Gastos de una categoría y su total en bolivianos."""
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)  # solo lectura
    conexion.row_factory = sqlite3.Row  # cada fila como diccionario
    filas = conexion.execute(
        "SELECT fecha, descripcion, monto FROM gastos WHERE categoria = ?",  # SQL fijo; el valor va en ?
        (categoria,),
    ).fetchall()
    conexion.close()
    return {"categoria": categoria, "total_bs": sum(f["monto"] for f in filas), "gastos": [dict(f) for f in filas]}


@registro.tool_plain
def registrar_gasto(fecha: str, categoria: Categoria, descripcion: str, monto: float) -> str:
    """Anota un gasto nuevo. La fecha va como AAAA-MM-DD y el monto en bolivianos."""
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB)  # sin mode=ro: esta tool sí escribe
    try:
        with conexion:  # transacción: guarda todo o nada
            conexion.execute("INSERT INTO gastos VALUES (?, ?, ?, ?)", (fecha, categoria, descripcion, monto))
    except sqlite3.IntegrityError:  # el CHECK de la tabla rechaza montos de 0 o negativos
        return f"No se guardó: el monto debe ser mayor que 0 (recibí {monto})."
    finally:
        conexion.close()
    return f"Gasto registrado: {descripcion}, {monto} Bs en {categoria} el {fecha}."


# ═══ S2 · RAG: buscar en mis documentos (PDFs) y citar la fuente ══════════════════════
DOCUMENTOS = DATOS / "documentos"  # PDFs ficticios de finanzas personales


def embeddings_de(textos: list[str]) -> list[list[float]]:
    """Convierte cada texto en un vector: textos de significado parecido dan vectores cercanos."""
    # Gemini también ofrece una API compatible con la de OpenAI; la usamos para los embeddings
    cliente = OpenAI(
        api_key=settings.gemini_api_key_local.get_secret_value(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    respuesta = cliente.embeddings.create(model="gemini-embedding-001", input=textos)
    return [item.embedding for item in respuesta.data]


@cache  # se indexa la primera vez que alguien busca, y una sola vez mientras el programa corre
def indice():
    # Base vectorial en memoria (se pierde al terminar); "cosine" mide la cercanía por el ángulo
    coleccion = chromadb.EphemeralClient().get_or_create_collection(
        "documentos", configuration={"hnsw": {"space": "cosine"}}
    )
    for pdf in sorted(DOCUMENTOS.glob("*.pdf")):
        texto = "\n".join(pagina.extract_text() for pagina in PdfReader(pdf).pages)
        # Trozos de ~300 caracteres que se solapan 50, para no cortar una idea por la mitad
        trozos = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_text(texto)
        coleccion.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(trozos))],  # el id dice de qué PDF sale: sirve para citar
            documents=trozos,
            embeddings=embeddings_de(trozos),
        )
    return coleccion


def buscar(pregunta: str, k: int = 3) -> list[dict]:
    """Los k trozos más parecidos a la pregunta, con su fuente y su similitud (1 = idéntico)."""
    r = indice().query(query_embeddings=embeddings_de([pregunta]), n_results=k)
    return [
        {"fuente": fuente, "similitud": round(1 - distancia, 2), "texto": texto}
        for fuente, distancia, texto in zip(r["ids"][0], r["distances"][0], r["documents"][0])
    ]


documentos = Capability(
    id="mis_documentos",
    description=(
        "Documentos adjuntos: contratos, facturas, garantías y recetas de cocina "
        "(incluidas las recetas de chuletas glaseadas)"
    ),
    # Citar la fuente y admitir cuando no hay datos: así se ve qué sale de los documentos y qué no
    instructions=(
        "Para preguntas sobre pagos, contratos, garantías o recetas adjuntas, usa buscar_documentos. "
        "En las preguntas de cocina, busca ingredientes y pasos en recetas.pdf; no completes la receta "
        "con ingredientes o instrucciones inventados. Cita la fuente entre corchetes, por ejemplo "
        "[contrato_alquiler-1] o [recetas-1]. "
        "Si no hay información suficiente para responder, di que no lo encuentras, no lo inventes."
    ),
)


@documentos.tool_plain
def buscar_documentos(pregunta: str) -> list[dict]:
    """Busca en los documentos los fragmentos que responden a la pregunta."""
    resultados = buscar(pregunta, k=3)
    for r in resultados:  # en la terminal (o en los Logs de Render) se ve qué recuperó
        print(f"  📄 {r['fuente']} (similitud {r['similitud']})", flush=True)
    return resultados
