
"""
Prueba los modelos de Gemini disponibles hasta que alguno responda la pregunta.
 
Uso:
    python probar_modelos.py
    python probar_modelos.py "Tu pregunta acá"
"""
 
import os
import sys
 
from dotenv import load_dotenv
from google import genai
from google.genai import errors
 
PREGUNTA_POR_DEFECTO = "Explicame qué es RAG en dos oraciones."
 
# Palabras que indican modelos que no sirven para responder texto
# (embeddings, imágenes, video, audio, etc.)
EXCLUIR = ("embedding", "imagen", "image", "veo", "tts", "audio", "live", "aqa")
 
 
def prioridad(nombre: str) -> int:
    """Ordena los modelos: primero los que suelen ser gratuitos."""
    if "flash-lite" in nombre:
        return 0
    if "flash" in nombre:
        return 1
    if "pro" in nombre:
        return 3  # suelen ser pagos: se prueban al final
    return 2
 
 
def obtener_modelos(client: genai.Client) -> list[str]:
    """Devuelve los modelos que pueden generar texto, ordenados por prioridad."""
    modelos = []
    for modelo in client.models.list():
        acciones = modelo.supported_actions or []
        if "generateContent" not in acciones:
            continue
        nombre = modelo.name.removeprefix("models/")
        if any(palabra in nombre for palabra in EXCLUIR):
            continue
        modelos.append(nombre)
    return sorted(modelos, key=lambda n: (prioridad(n), n))
 
 
def main() -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No encontré GEMINI_API_KEY. Revisá tu archivo .env.")
        return 1
 
    pregunta = sys.argv[1] if len(sys.argv) > 1 else PREGUNTA_POR_DEFECTO
    client = genai.Client(api_key=api_key)
 
    try:
        modelos = obtener_modelos(client)
    except errors.APIError as e:
        print(f"❌ No pude obtener la lista de modelos ({e.code}): {e.message}")
        return 1
 
    if not modelos:
        print("❌ No hay ningún modelo disponible para responder.")
        return 1
 
    print(f"Pregunta: {pregunta}")
    print(f"Voy a probar {len(modelos)} modelos...\n")
 
    for nombre in modelos:
        print(f"→ Probando {nombre}... ", end="", flush=True)
        try:
            respuesta = client.models.generate_content(model=nombre, contents=pregunta)
        except errors.APIError as e:
            # 429 = sin cuota, 404 = modelo no disponible, 403 = sin permiso, etc.
            print(f"falló ({e.code} {e.status})")
            continue
        except Exception as e:  # errores de red u otros imprevistos
            print(f"falló ({type(e).__name__})")
            continue
 
        if not respuesta.text:
            print("falló (respuesta vacía)")
            continue
 
        print("✅ funcionó\n")
        print(f"Modelo: {nombre}")
        print(f"Respuesta:\n{respuesta.text}")
        print(f"\nTip: poné GEMINI_MODEL={nombre} en tu .env")
        return 0
 
    print("\n❌ No hay ningún modelo disponible para responder.")
    return 1
 
 
if __name__ == "__main__":
    sys.exit(main())