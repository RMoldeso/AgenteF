"""Pruebas sin modelo real ni internet: uv run pytest

Un modelo falso (FunctionModel) pide la tool que queremos, y comprobamos lo que hace el código.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY_LOCAL", "clave-de-prueba")  # para importar sin .env (no se llama a Gemini)

import httpx  # noqa: E402
import pytest  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

import api  # noqa: E402
import asistente  # noqa: E402
import bot  # noqa: E402
import capacidades  # noqa: E402
from config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def base_temporal(tmp_path, monkeypatch):
    """Cada prueba usa su propia gastos.db, no la de datos/."""
    monkeypatch.setattr(capacidades, "DB", tmp_path / "gastos.db")


def pide(tool: str, args: dict) -> FunctionModel:
    """Un modelo falso: la 1.ª vez pide la tool; cuando recibe su resultado, responde texto."""

    def responder(mensajes, info):
        ultimo = mensajes[-1].parts[-1]
        if isinstance(ultimo, ToolReturnPart):
            return ModelResponse(parts=[TextPart(f"Resultado: {ultimo.content}")])
        return ModelResponse(parts=[ToolCallPart(tool, args)])

    return FunctionModel(responder)


def crear_lista_clientes(ruta):
    libro = Workbook()
    hoja = libro.active
    hoja.append(["ID", "Nombre completo", "Fecha de nacimiento", "Dirección", "Localidad y Código postal",
                 "Teléfono", "Correo electrónico"])
    hoja.append([1, "José Álvarez", "1990-01-01", "Calle de prueba 123", "Ciudad de prueba", "555-0100",
                 "prueba@example.invalid"])
    hoja.append([2, "Ana Ejemplo", "1991-02-02", "Avenida de muestra 456", "Otra ciudad", "555-0101",
                 "ana@example.invalid"])
    libro.save(ruta)


def test_consultar_gastos_sin_modelo():
    gastos = capacidades.consultar_gastos("transporte")  # una tool es una función normal
    assert gastos["total_bs"] == 67


def test_clasificacion_f1_usa_datos_de_la_api(monkeypatch):
    def get(url, timeout):
        assert url.endswith("/current/driverStandings.json")
        return httpx.Response(200, request=httpx.Request("GET", url), json={
            "MRData": {
                "StandingsTable": {
                    "season": "2030",
                    "round": "4",
                    "StandingsLists": [{
                        "DriverStandings": [{
                            "position": "1",
                            "points": "100",
                            "wins": "3",
                            "Driver": {"givenName": "Ejemplo", "familyName": "Piloto"},
                            "Constructors": [{"name": "Equipo"}],
                        }],
                    }],
                },
            },
        })

    monkeypatch.setattr(capacidades.httpx, "get", get)
    resultado = capacidades.clasificacion_f1()
    assert resultado["clasificacion"][0] == {
        "posicion": 1,
        "nombre": "Ejemplo Piloto",
        "puntos": 100.0,
        "victorias": 3,
        "equipo": "Equipo",
    }
    assert resultado["fuente"] == "Jolpica F1 API"


def test_cotizacion_cripto_reconoce_simbolo_y_nombre(monkeypatch):
    def get(url, timeout):
        assert url.endswith("/btc-bitcoin")
        return httpx.Response(200, request=httpx.Request("GET", url), json={
            "name": "Bitcoin",
            "symbol": "BTC",
            "last_updated": "2030-01-01T00:00:00Z",
            "quotes": {"USD": {"price": 50000, "percent_change_24h": 2.5}},
        })

    monkeypatch.setattr(capacidades.httpx, "get", get)
    resultado = capacidades.cotizacion_cripto(" Bitcoin ")
    assert resultado["simbolo"] == "BTC"
    assert resultado["precio_usd"] == 50000
    assert resultado["fuente"] == "CoinPaprika"


def test_cotizacion_cripto_rechaza_moneda_desconocida(monkeypatch):
    def no_debe_consultar(*args, **kwargs):
        raise AssertionError("no debe llamar a la API para una moneda desconocida")

    monkeypatch.setattr(capacidades.httpx, "get", no_debe_consultar)
    assert "No reconozco" in capacidades.cotizacion_cripto("moneda inventada")["error"]


def test_documentos_incluye_recetas_adjuntas():
    texto = "\n".join(
        pagina.extract_text() or ""
        for pagina in capacidades.PdfReader(capacidades.DOCUMENTOS / "recetas.pdf").pages
    )
    assert "CHULETAS" in texto.upper()
    assert "GLASEADAS" in texto.upper()


async def test_agente_puede_buscar_clientes_por_nombre(monkeypatch, tmp_path):
    archivo = tmp_path / "Lista-de-clientes-con-nombre-y-direccion.csv"
    crear_lista_clientes(archivo)
    monkeypatch.setattr(capacidades, "LISTA_CLIENTES", archivo)

    with asistente.asistente.override(model=pide("buscar_clientes", {"busqueda": "Jose Alvarez"})):
        respuesta = await asistente.responder("Busca a José Álvarez y dime su dirección")

    assert "José Álvarez" in respuesta
    assert "Calle de prueba 123" in respuesta
    assert "555-0100" not in respuesta
    assert "prueba@example.invalid" not in respuesta


def test_buscar_clientes_limita_listado_y_acepta_archivo_xlsx_con_extension_csv(monkeypatch, tmp_path):
    archivo = tmp_path / "Lista-de-clientes-con-nombre-y-direccion.csv"
    libro = Workbook()
    hoja = libro.active
    hoja.append(["ID", "Nombre completo", "Dirección"])
    for numero in range(25):
        hoja.append([numero, f"Cliente {numero}", f"Dirección {numero}"])
    libro.save(archivo)
    monkeypatch.setattr(capacidades, "LISTA_CLIENTES", archivo)

    resultado = capacidades.buscar_clientes()

    assert resultado["coincidencias"] == 25
    assert len(resultado["resultados"]) == 20
    assert resultado["truncado"] is True


async def test_registrar_gasto_escribe_en_la_base():
    with asistente.asistente.override(model=pide("registrar_gasto", {
        "fecha": "2030-05-14", "categoria": "ocio", "descripcion": "Teatro", "monto": 30,
    })):
        respuesta = await asistente.responder("Anota 30 Bs de teatro.")
    assert "Gasto registrado" in respuesta
    assert capacidades.consultar_gastos("ocio")["total_bs"] == 70  # 40 del CSV + 30


async def test_monto_negativo_no_se_guarda():
    with asistente.asistente.override(model=pide("registrar_gasto", {
        "fecha": "2030-05-14", "categoria": "ocio", "descripcion": "Error", "monto": -5,
    })):
        respuesta = await asistente.responder("Anota -5 Bs.")
    assert "No se guardó" in respuesta
    assert capacidades.consultar_gastos("ocio")["total_bs"] == 40


async def test_chat_ajeno_se_rechaza_sin_llamar_al_modelo(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", "123")
    respuestas = []

    async def reply_text(texto):
        respuestas.append(texto)

    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=999), message=SimpleNamespace(text="hola", reply_text=reply_text)
    )
    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        await bot.mensaje(update, None)
    assert "No autorizado" in respuestas[0] and "999" in respuestas[0]


def test_permitidos(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", " 123, 456 ,")
    assert settings.permitidos == {"123", "456"}
    monkeypatch.setattr(settings, "telegram_permitidos", "")
    assert settings.permitidos == set()  # vacío = nadie


# ── Extra opcional: la API (API.md) ──


async def post_chat(cuerpo: dict, token: str | None = None) -> httpx.Response:
    """Llama a la API en el mismo proceso, sin servidor ni red."""
    cabeceras = {"Authorization": f"Bearer {token}"} if token else {}
    transporte = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
        return await cliente.post("/chat", json=cuerpo, headers=cabeceras)


async def test_api_sin_token_valido_no_llama_al_modelo(monkeypatch):
    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        monkeypatch.setattr(settings, "api_tokens", SecretStr(""))
        assert (await post_chat({"mensaje": "hola"}, "cualquiera")).status_code == 401  # vacío = nadie
        monkeypatch.setattr(settings, "api_tokens", SecretStr("abc, def"))
        assert (await post_chat({"mensaje": "hola"})).status_code == 401
        assert (await post_chat({"mensaje": "hola"}, "xyz")).status_code == 401
        assert (await post_chat({"mensaje": ""}, "abc")).status_code == 422  # Pydantic valida el pedido


async def test_api_con_token_responde(monkeypatch):
    monkeypatch.setattr(settings, "api_tokens", SecretStr("abc,def"))
    with asistente.asistente.override(model=pide("consultar_gastos", {"categoria": "transporte"})):
        r = await post_chat({"mensaje": "¿Cuánto gasté en transporte?"}, "def")
    assert r.status_code == 200
    assert "67" in r.json()["texto"]
