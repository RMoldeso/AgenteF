"""Las capabilities de tu asistente: lo que vimos en clase. Cámbialas o agrega las tuyas.

Una Capability junta tools + instrucciones en una pieza que se enchufa al agente (asistente.py).
El docstring y los tipos de cada tool son lo único que el modelo lee para decidir cuándo usarla.
"""

import csv
import sqlite3
import unicodedata
from datetime import date
from functools import cache
from typing import Literal

import chromadb
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openpyxl import load_workbook
from openai import OpenAI
from pydantic_ai.capabilities import Capability
from pypdf import PdfReader

from config import DATOS, settings

# ═══ S1 · Tools de lectura con APIs públicas ══════════════════════════════════════════
apis = Capability(
    id="apis_publicas",
    description=(
        "Consulta la clasificación actual de Fórmula 1 y cotizaciones de criptomonedas "
        "con APIs públicas gratuitas"
    ),
    instructions=(
        "Usa las APIs para obtener datos reales de Fórmula 1 y criptomonedas. "
        "Indica la fuente y no inventes datos."
    ),
)


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


# ═══ S1 · Buscar clientes en la lista adjunta ═════════════════════════════════════════
LISTA_CLIENTES = DATOS / "Lista-de-clientes-con-nombre-y-direccion.csv"
clientes = Capability(
    id="lista_clientes",
    description="Busca clientes por nombre o ID y devuelve su nombre y dirección desde la lista adjunta",
    instructions=(
        "Para consultar clientes, usa buscar_clientes. Si piden toda la lista, muestra el resumen limitado "
        "que devuelve la herramienta; para encontrar a alguien, busca por su nombre o ID. "
        "Devuelve únicamente nombre y dirección/localidad: no reveles fecha de nacimiento, teléfono ni correo."
    ),
)


def _normalizar_busqueda(texto: object) -> str:
    """Compara texto ignorando mayúsculas y tildes."""
    descompuesto = unicodedata.normalize("NFKD", str(texto or ""))
    return "".join(caracter for caracter in descompuesto if not unicodedata.combining(caracter)).casefold().strip()


@clientes.tool_plain
def buscar_clientes(busqueda: str = "") -> dict:
    """Busca hasta 20 clientes por nombre o ID; sin búsqueda muestra los primeros 20.

    Args:
        busqueda: nombre o ID del cliente. Puede quedar vacío para ver una muestra de la lista.
    """
    if not LISTA_CLIENTES.is_file():
        return {"error": f"No se encontró el archivo de clientes: {LISTA_CLIENTES.name}"}

    try:
        archivo = LISTA_CLIENTES.open("rb")
    except OSError as error:
        return {"error": f"No se pudo abrir la lista de clientes: {error}"}

    with archivo:
        libro = load_workbook(archivo, read_only=True, data_only=True)
        try:
            hoja = libro.active
            filas = hoja.iter_rows(values_only=True)
            encabezados = next(filas, None)
            if not encabezados:
                return {"error": "La lista de clientes está vacía"}

            columnas = {_normalizar_busqueda(nombre): indice for indice, nombre in enumerate(encabezados)}
            indice_nombre = columnas.get("nombre completo")
            indice_direccion = columnas.get("direccion")
            indice_localidad = columnas.get("localidad y codigo postal")
            indice_id = columnas.get("id")
            if indice_nombre is None or indice_direccion is None:
                return {"error": "La lista no contiene las columnas necesarias de nombre y dirección"}

            termino = _normalizar_busqueda(busqueda)
            resultados = []
            coincidencias = 0
            for fila in filas:
                nombre = fila[indice_nombre] if indice_nombre < len(fila) else None
                identificador = fila[indice_id] if indice_id is not None and indice_id < len(fila) else None
                if termino and termino not in _normalizar_busqueda(nombre) and termino not in _normalizar_busqueda(
                    identificador
                ):
                    continue

                coincidencias += 1
                if len(resultados) < 20:
                    direccion = fila[indice_direccion] if indice_direccion < len(fila) else None
                    localidad = (
                        fila[indice_localidad]
                        if indice_localidad is not None and indice_localidad < len(fila)
                        else None
                    )
                    resultado = {"nombre": nombre, "direccion": direccion}
                    if localidad:
                        resultado["localidad_y_codigo_postal"] = localidad
                    resultados.append(resultado)
        finally:
            libro.close()

    return {
        "busqueda": busqueda.strip() or "lista completa",
        "coincidencias": coincidencias,
        "resultados": resultados,
        "limite": 20,
        "truncado": coincidencias > 20,
        "fuente": LISTA_CLIENTES.name,
    }


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
DOCUMENTOS = DATOS / "documentos"  # documentos adjuntos ficticios


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
        "Recetas de cocina adjuntas, incluidas las recetas de chuletas glaseadas"
    ),
    # Citar la fuente y admitir cuando no hay datos: así se ve qué sale de los documentos y qué no
    instructions=(
        "Para preguntas sobre las recetas adjuntas, usa buscar_documentos. "
        "Busca ingredientes y pasos en recetas.pdf; no completes la receta "
        "con ingredientes o instrucciones inventados. Cita la fuente entre corchetes, por ejemplo "
        "[recetas-1]. "
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
