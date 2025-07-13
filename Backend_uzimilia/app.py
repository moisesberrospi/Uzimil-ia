"""
UZIMIL-IA - Agente de Inteligencia Artificial para Análisis Comercial

Autor        : Moises Berrospi Farias
Archivo      : app.py
Versión      : 1.0
Tipo         : Script principal (PY)
Scheduler    : Cloud Run

Descripción:
    Este script implementa un agente de IA capaz de analizar datos comerciales y responder preguntas
    relacionadas con el inventario y las ventas de una tienda de informática. Utiliza herramientas
    de Langchain y OpenAI para procesamiento de lenguaje natural, análisis de datos, búsqueda semántica
    y generación de texto, permitiendo la interacción inteligente con los datos comerciales.

Observaciones:
    - Forma parte del proyecto UZIMIL-IA.
    - Diseñado para ejecutarse en entornos cloud (Cloud Run).
    - Facilita la toma de decisiones comerciales mediante análisis automatizado y respuestas conversacionales.
"""


# ==================== IMPORTS ====================
import os
import re
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from difflib import get_close_matches
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_elasticsearch import ElasticsearchStore
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from elasticsearch import Elasticsearch
from langchain.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ==================== CONFIGURACIÓN DE VARIABLES DE ENTORNO ====================
# Variables para trazabilidad y acceso a APIs de Langchain y OpenAI
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "_your_key_here_"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "gcpaiagent"
os.environ["OPENAI_API_KEY"] = "_your_key_here_"
os.environ["ELASTIC_SEARCH_API_KEY"] = "_your_key_here_"


# ==================== INICIALIZACIÓN DE LA APP FLASK ====================


app = Flask(__name__)

# Endpoint principal para Cloud Run
@app.route('/agent', methods=['GET'])
def agent():
    try:
        result = main()
        # Si la respuesta es un string, la devolvemos como JSON
        return result
    except Exception as e:
        # Devuelve el error para facilitar el debug
        return jsonify({"error": str(e)}), 500

### === FUNCIONES DE INVENTARIO ===
def obtener_inventario_completo():
    """
    Obtiene todo el inventario desde Elasticsearch y lo retorna como un DataFrame de pandas.
    Devuelve: DataFrame con los productos, stock y precios.
    """
    es = Elasticsearch("http://34.9.158.34:9200", basic_auth=("elastic", os.getenv("ELASTIC_SEARCH_API_KEY")))
    res = es.search(
        index="lg-stockdata",
        body={
            "size": 1000,
            "query": {"match_all": {}}
        }
    )
    docs = [hit["_source"] for hit in res["hits"]["hits"]]
    # Extraer metadatos si están anidados
    data = []
    for doc in docs:
        if "metadata" in doc:
            row = doc["metadata"]
            row["Producto"] = doc.get("text", "")
            data.append(row)
        else:
            data.append(doc)
    df = pd.DataFrame(data)
    return df

def stock_total() -> str:
    """
    Calcula y devuelve el stock total disponible en el inventario.
    """
    df = obtener_inventario_completo()
    total = df["stock_disponible"].sum()
    return f"El stock total disponible es: {total} unidades"

def precio_promedio() -> str:
    """
    Calcula y devuelve el precio unitario promedio del inventario.
    """
    df = obtener_inventario_completo()
    avg_price = df["preciounitario"].mean()
    return f"El precio unitario promedio del inventario es: ${avg_price:.2f}"

def producto_mas_caro() -> str:
    """
    Devuelve el producto más caro del inventario y su precio.
    """
    df = obtener_inventario_completo()
    prod = df.loc[df["preciounitario"].idxmax()]
    return f"El producto más caro es: {prod['Producto']} (${prod['preciounitario']})"

def producto_mas_barato() -> str:
    """
    Devuelve el producto más barato del inventario y su precio.
    """
    df = obtener_inventario_completo()
    prod = df.loc[df["preciounitario"].idxmin()]
    return f"El producto más barato es: {prod['Producto']} (${prod['preciounitario']})"

def inventario_con_precios() -> str:
    """
    Devuelve un listado de productos con su stock y precio unitario en formato tabla.
    """
    df = obtener_inventario_completo()
    return df[["Producto", "stock_disponible", "preciounitario"]].to_string(index=False)

def stock_por_categoria() -> str:
    """
    Devuelve el stock total disponible agrupado por categoría.
    """
    df = obtener_inventario_completo()
    resumen = df.groupby("categoria")["stock_disponible"].sum().sort_values(ascending=False)
    return f"Stock total disponible por categoría:\n{resumen.to_string()}"

def stock_por_marca() -> str:
    """
    Devuelve el stock total disponible agrupado por marca.
    """
    df = obtener_inventario_completo()
    resumen = df.groupby("marca")["stock_disponible"].sum().sort_values(ascending=False)
    return f"Stock total disponible por marca:\n{resumen.to_string()}"

def stock_total() -> str:
    """
    Calcula y devuelve el stock total disponible en el inventario.
    """
    df = obtener_inventario_completo()
    total = df["stock_disponible"].sum()
    return f"El stock total disponible es: {total} unidades"

def precio_promedio() -> str:
    """
    Calcula y devuelve el precio unitario promedio del inventario.
    """
    df = obtener_inventario_completo()
    avg_price = df["preciounitario"].mean()
    return f"El precio unitario promedio del inventario es: ${avg_price:.2f}"

def producto_mas_caro() -> str:
    """
    Devuelve el producto más caro del inventario y su precio.
    """
    df = obtener_inventario_completo()
    prod = df.loc[df["preciounitario"].idxmax()]
    return f"El producto más caro es: {prod['Producto']} (${prod['preciounitario']})"

def producto_mas_barato() -> str:
    """
    Devuelve el producto más barato del inventario y su precio.
    """
    df = obtener_inventario_completo()
    prod = df.loc[df["preciounitario"].idxmin()]
    return f"El producto más barato es: {prod['Producto']} (${prod['preciounitario']})"

def inventario_con_precios() -> str:
    """
    Devuelve un listado de productos con su stock y precio unitario en formato tabla.
    """
    df = obtener_inventario_completo()
    return df[["Producto", "stock_disponible", "preciounitario"]].to_string(index=False)


### === FUNCIONES DE VENTAS ===
def obtener_ventas_completas():
    """
    Obtiene todas las ventas desde Elasticsearch y las retorna como un DataFrame de pandas.
    Devuelve: DataFrame con los registros de ventas.
    """
    es = Elasticsearch("http://34.9.158.34:9200", basic_auth=("elastic", os.getenv("ELASTIC_SEARCH_API_KEY")))
    res = es.search(
        index="lg-ventadata",
        body={
            "size": 1000,
            "query": {"match_all": {}}
        }
    )
    docs = [hit["_source"] for hit in res["hits"]["hits"]]
    data = []
    for doc in docs:
        if "metadata" in doc:
            row = doc["metadata"]
            row["producto"] = doc.get("text", "")
            data.append(row)
        else:
            data.append(doc)
    df = pd.DataFrame(data)
    return df

### === FUNCIONES DE ANÁLISIS TEMPORAL ===
def filtrar_ventas_por_periodo(df, periodo="semana", ultimos_n=4):
    """
    Filtra el DataFrame de ventas por el periodo indicado y retorna solo los últimos N periodos.
    periodo: 'año', 'mes', 'semana', 'dia'
    ultimos_n: cuántos periodos atrás considerar
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    if periodo == "año":
        df["periodo"] = df["fecha"].dt.year
    elif periodo == "mes":
        df["periodo"] = df["fecha"].dt.to_period("M")
    elif periodo == "semana":
        df["periodo"] = df["fecha"].dt.isocalendar().week
        df["año"] = df["fecha"].dt.year
        df["periodo"] = df["año"].astype(str) + "-S" + df["periodo"].astype(str)
    elif periodo == "dia":
        df["periodo"] = df["fecha"].dt.date
    else:
        raise ValueError("Periodo no soportado")
    # Ordenar y filtrar los últimos N periodos
    periodos_ordenados = df["periodo"].drop_duplicates().sort_values()
    ultimos_periodos = periodos_ordenados.iloc[-ultimos_n:]
    df_filtrado = df[df["periodo"].isin(ultimos_periodos)]
    return df_filtrado, ultimos_periodos.tolist()

def cantidad_vendida_y_ganancia_por(df, por="categoria", periodo="semana", ultimos_n=4):
    """
    Devuelve la cantidad vendida y la ganancia real agrupada por categoria, marca o producto en los últimos N periodos.
    por: 'categoria', 'marca', 'producto'
    periodo: 'año', 'mes', 'semana', 'dia'
    ultimos_n: cuántos periodos atrás considerar
    """
    # Validación de columnas necesarias
    columnas_necesarias = ["fecha", por, "precioventafinalunitario", "preciounitario", "cantidadvendida"]
    for col in columnas_necesarias:
        if col not in df.columns:
            return f"No se encontró la columna '{col}' en los datos de ventas. Por favor revisa el archivo o la fuente de datos."

    # Validación de datos vacíos
    if df.empty:
        return "No hay datos de ventas disponibles para analizar."

    try:
        df_filtrado, periodos = filtrar_ventas_por_periodo(df, periodo, ultimos_n)
    except Exception as e:
        return f"Error al filtrar ventas por periodo: {str(e)}"

    if df_filtrado.empty:
        return f"No hay ventas registradas para los últimos {ultimos_n} {periodo}s."

    # Limpieza de datos: eliminar $ y , y convertir a float
    for col in ["precioventafinalunitario", "preciounitario"]:
        df_filtrado[col] = df_filtrado[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors="coerce")
    df_filtrado["cantidadvendida"] = pd.to_numeric(df_filtrado["cantidadvendida"], errors="coerce")

    df_filtrado["ganancia_real"] = (df_filtrado["precioventafinalunitario"] - df_filtrado["preciounitario"]) * df_filtrado["cantidadvendida"]
    resumen = df_filtrado.groupby(["periodo", por]).agg({
        "cantidadvendida": "sum",
        "ganancia_real": "sum"
    }).reset_index()

    salida = f"Cantidad vendida y ganancia real por {por} en los últimos {ultimos_n} {periodo}s:\n"
    hay_datos = False
    for p in periodos:
        sub = resumen[resumen["periodo"] == p]
        if sub.empty:
            continue
        hay_datos = True
        salida += f"\nPeriodo: {p}\n"
        for _, row in sub.iterrows():
            salida += f"  {row[por]}: {int(row['cantidadvendida'])} vendidos, ganancia real: ${row['ganancia_real']:.2f}\n"
    if not hay_datos:
        return f"No hay ventas registradas para los últimos {ultimos_n} {periodo}s."
    return salida.strip()

# Wrappers para tools
def cantidad_vendida_y_ganancia_por_categoria(periodo="semana", ultimos_n=4):
    df = obtener_ventas_completas()
    return cantidad_vendida_y_ganancia_por(df, por="categoria", periodo=periodo, ultimos_n=ultimos_n)

def cantidad_vendida_y_ganancia_por_marca(periodo="semana", ultimos_n=4):
    df = obtener_ventas_completas()
    return cantidad_vendida_y_ganancia_por(df, por="marca", periodo=periodo, ultimos_n=ultimos_n)

def cantidad_vendida_y_ganancia_por_producto(periodo="semana", ultimos_n=4):
    df = obtener_ventas_completas()
    return cantidad_vendida_y_ganancia_por(df, por="producto", periodo=periodo, ultimos_n=ultimos_n)

### === FUNCIONES DE ANÁLISIS POR CATEGORÍA Y MARCA ===
def ganancia_total_por_categoria() -> str:
    """
    Devuelve la ganancia total (precio venta - precio compra) agrupada por categoría.
    """
    df = obtener_ventas_completas()
    df["ganancia"] = (df["precioventafinalunitario"] - df["preciounitario"]) * df["cantidadvendida"]
    resumen = df.groupby("categoria")["ganancia"].sum().sort_values(ascending=False)
    return f"Ganancia total por categoría:\n{resumen.round(2).to_string()}"

def ganancia_total_por_marca() -> str:
    """
    Devuelve la ganancia total (precio venta - precio compra) agrupada por marca.
    """
    df = obtener_ventas_completas()
    df["ganancia"] = (df["precioventafinalunitario"] - df["preciounitario"]) * df["cantidadvendida"]
    resumen = df.groupby("marca")["ganancia"].sum().sort_values(ascending=False)
    return f"Ganancia total por marca:\n{resumen.round(2).to_string()}"

def cantidad_vendida_por_categoria() -> str:
    """
    Devuelve la cantidad total de productos vendidos agrupada por categoría.
    """
    df = obtener_ventas_completas()
    resumen = df.groupby("categoria")["cantidadvendida"].sum().sort_values(ascending=False)
    return f"Cantidad total de productos vendidos por categoría:\n{resumen.to_string()}"

def cantidad_vendida_por_marca() -> str:
    """
    Devuelve la cantidad total de productos vendidos agrupada por marca.
    """
    df = obtener_ventas_completas()
    resumen = df.groupby("marca")["cantidadvendida"].sum().sort_values(ascending=False)
    return f"Cantidad total de productos vendidos por marca:\n{resumen.to_string()}"

def productos_mas_vendidos(top_n: int = 5) -> str:
    """
    Devuelve el top N de productos más vendidos por cantidad total.
    Parámetros:
        top_n (int): Número de productos a mostrar.
    """
    df = obtener_ventas_completas()
    ventas = df.groupby("producto")["cantidadvendida"].sum().sort_values(ascending=False).head(top_n)
    return f"Top {top_n} productos más vendidos:\n{ventas.to_string()}"

def variacion_precios() -> str:
    """
    Muestra la variación promedio entre el precio de compra y el precio de venta final por producto.
    """
    df = obtener_ventas_completas()
    df["variacion"] = df["precioventafinalunitario"] - df["preciounitario"]
    resumen = df.groupby("producto").agg({
        "preciounitario": "mean",
        "precioventafinalunitario": "mean",
        "variacion": ["mean", "min", "max"]
    })
    resumen.columns = ["precio_compra_prom", "precio_venta_prom", "variacion_prom", "variacion_min", "variacion_max"]
    resumen = resumen.sort_values("variacion_prom", ascending=False)
    return f"Variación promedio de precios de venta respecto a compra por producto:\n{resumen[['precio_compra_prom','precio_venta_prom','variacion_prom']].round(2).to_string()}"

def ventas_por_tiempo(periodo: str = "mes") -> str:
    """
    Devuelve el total de ventas agrupado por periodo (año, mes, semana o día).
    Parámetros:
        periodo (str): Puede ser 'año', 'mes', 'semana' o 'dia'.
    """
    df = obtener_ventas_completas()
    df["fecha"] = pd.to_datetime(df["fecha"])
    if periodo == "año":
        df["periodo"] = df["fecha"].dt.year
    elif periodo == "mes":
        df["periodo"] = df["fecha"].dt.to_period("M")
    elif periodo == "semana":
        df["periodo"] = df["fecha"].dt.isocalendar().week
        df["año"] = df["fecha"].dt.year
        df["periodo"] = df["año"].astype(str) + "-S" + df["periodo"].astype(str)
    elif periodo == "dia":
        df["periodo"] = df["fecha"].dt.date
    else:
        return "Periodo no soportado. Usa: año, mes, semana o dia."
    resumen = df.groupby("periodo")["totalventa"].sum().sort_index()
    return f"Ventas totales por {periodo}:\n{resumen.round(2).to_string()}"

### === FUNCIONES DE UTILIDAD Y CONTEXTO ===
def alertas_automaticas(tipo="todas"):
    """
    Analiza los datos y retorna un string con las alertas detectadas:
    - Caída brusca en ventas
    - Productos con stock bajo
    - Incremento inesperado de precios
    No imprime ni loguea nada, solo retorna el resultado para uso interno del modelo.
    """
    mensajes = []
    # Alerta de caída brusca en ventas (comparación mes actual vs anterior)
    try:
        df_ventas = obtener_ventas_completas()
        df_ventas["fecha"] = pd.to_datetime(df_ventas["fecha"])
        ventas_mes = df_ventas.groupby(df_ventas["fecha"].dt.to_period("M"))["totalventa"].sum().sort_index()
        if len(ventas_mes) >= 2:
            variacion = (ventas_mes.iloc[-1] - ventas_mes.iloc[-2]) / (ventas_mes.iloc[-2] + 1e-8)
            if variacion < -0.3:
                mensajes.append(f"⚠️ Alerta: Las ventas del mes actual han caído un {abs(variacion)*100:.1f}% respecto al mes anterior.")
    except Exception as e:
        pass  
    # Alerta de productos con stock bajo
    try:
        df_inv = obtener_inventario_completo()
        bajo_stock = df_inv[df_inv["stock_disponible"] <= 3]
        if not bajo_stock.empty:
            productos = ", ".join(bajo_stock["Producto"].astype(str))
            mensajes.append(f"⚠️ Alerta: Los siguientes productos tienen stock bajo (≤3 unidades): {productos}")
    except Exception as e:
        pass  
    # Alerta de incremento inesperado de precios (más del 20% respecto al promedio histórico)
    try:
        if "preciounitario" in df_inv.columns:
            for _, row in df_inv.iterrows():
                nombre = row["Producto"]
                precio_actual = row["preciounitario"]
                # Buscar histórico de ventas de ese producto
                ventas_prod = df_ventas[df_ventas["producto"] == nombre]
                if not ventas_prod.empty:
                    precio_hist = ventas_prod["precioventafinalunitario"].mean()
                    if precio_hist > 0 and precio_actual > 1.2 * precio_hist:
                        mensajes.append(f"⚠️ Alerta: El precio actual de '{nombre}' (${precio_actual}) es más de 20% superior al promedio histórico de ventas (${precio_hist:.2f}).")
    except Exception as e:
        pass  
    if not mensajes:
        return "No se detectaron alertas ni anomalías relevantes."
    return "\n".join(mensajes)

def obtener_alertas_contexto():
    """
    Obtiene las alertas automáticas solo para usarlas como contexto del modelo, nunca para mostrar al usuario.
    """
    return alertas_automaticas() 

def buscar_productos_similares(nombre: str, top_n: int = 5) -> str:
    """
    Busca productos similares por nombre usando coincidencia aproximada.
    Parámetros:
        nombre (str): Nombre a buscar.
        top_n (int): Número de sugerencias a mostrar.
    """
    df = obtener_inventario_completo()
    nombres = df["Producto"].astype(str).tolist()
    similares = get_close_matches(nombre, nombres, n=top_n, cutoff=0.5)
    if similares:
        return f"¿Quizás quisiste decir?:\n" + "\n".join(similares)
    else:
        return "No se encontraron productos similares."

def buscar_producto_semantico(nombre_usuario, top_n=3):
    """
    Busca productos similares usando embeddings y similitud semántica.
    Parámetros:
        nombre_usuario (str): Nombre a buscar semánticamente.
        top_n (int): Número de sugerencias a mostrar.
    """
    df = obtener_inventario_completo()
    nombres = df["Producto"].astype(str).tolist()
    # Obtener embeddings de los nombres de productos
    embeddings = OpenAIEmbeddings()
    vectores_productos = embeddings.embed_documents(nombres)
    # Obtener embedding del texto del usuario
    vector_usuario = embeddings.embed_query(nombre_usuario)
    # Calcular similitud de coseno
    similitudes = np.dot(vectores_productos, vector_usuario) / (
        np.linalg.norm(vectores_productos, axis=1) * np.linalg.norm(vector_usuario) + 1e-8
    )
    indices_top = np.argsort(similitudes)[-top_n:][::-1]
    productos_similares = [nombres[i] for i in indices_top if similitudes[i] > 0.5]
    if productos_similares:
        return f"¿Quizás quisiste decir (búsqueda semántica)?:\n" + "\n".join(productos_similares)
    else:
        return "No se encontraron productos similares (ni semánticamente)."

def resumir_contexto(texto: str, max_tokens: int = 300) -> str:
    """
    Resume un texto largo para reducir el número de tokens y facilitar el análisis.
    Parámetros:
        texto (str): Texto a resumir.
        max_tokens (int): Máximo de tokens por resumen parcial.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens*4, chunk_overlap=0)
    partes = splitter.split_text(texto)
    if len(partes) == 1:
        return partes[0]
    # Usar el modelo para resumir cada parte y concatenar
    model = ChatOpenAI(model="gpt-3.5-turbo-16k")
    resumenes = []
    for parte in partes:
        prompt = f"Resume el siguiente texto en máximo {max_tokens} tokens:\n{parte}"
        respuesta = model.invoke(prompt)
        resumenes.append(respuesta.content)
    return " ".join(resumenes)

def parsear_periodo_desde_texto(texto):
    """
    Extrae periodo ('semana', 'mes', 'año', 'dia') y ultimos_n de frases como:
    'últimas 2 semanas', 'últimos 3 meses', 'últimos 5 días', etc.
    Devuelve (periodo:str, ultimos_n:int) o (None, None) si no encuentra.
    """
    texto = texto.lower()
    patrones = [
        (r'(últimas|ultimas|últimos|ultimos)\s+(\d+)\s*(semanas|semanas?)', 'semana'),
        (r'(últimas|ultimas|últimos|ultimos)\s+(\d+)\s*(meses|mes)', 'mes'),
        (r'(últimas|ultimas|últimos|ultimos)\s+(\d+)\s*(años|año|anos|ano)', 'año'),
        (r'(últimas|ultimas|últimos|ultimos)\s+(\d+)\s*(días|dias|día|dia)', 'dia'),
    ]
    for patron, periodo in patrones:
        m = re.search(patron, texto)
        if m:
            try:
                ultimos_n = int(m.group(2))
                return periodo, ultimos_n
            except:
                continue
    # Si no encuentra, busca solo periodo
    for palabra, periodo in [('semana', 'semana'), ('mes', 'mes'), ('año', 'año'), ('dia', 'dia'), ('día', 'dia')]:
        if palabra in texto:
            return periodo, 1
    return None, None

### === FUNCIONES DE ANÁLISIS AVANZADO ===
def rentabilidad_por_producto() -> str:
    """
    Devuelve la rentabilidad (ganancia/ingreso) de cada producto, ordenando del más rentable al menos rentable.
    """
    df = obtener_ventas_completas()
    df["ganancia"] = (df["precioventafinalunitario"] - df["preciounitario"]) * df["cantidadvendida"]
    df["ingreso"] = df["precioventafinalunitario"] * df["cantidadvendida"]
    resumen = df.groupby("producto").agg({"ganancia": "sum", "ingreso": "sum"})
    resumen["rentabilidad"] = resumen["ganancia"] / (resumen["ingreso"] + 1e-8)
    resumen = resumen.sort_values("rentabilidad", ascending=False)
    return f"Rentabilidad por producto (ganancia/ingreso):\n{resumen[['rentabilidad']].round(2).to_string()}"

def productos_menos_vendidos(top_n: int = 5) -> str:
    """
    Devuelve el top N de productos menos vendidos por cantidad total.
    """
    df = obtener_ventas_completas()
    ventas = df.groupby("producto")["cantidadvendida"].sum().sort_values(ascending=True).head(top_n)
    return f"Top {top_n} productos menos vendidos:\n{ventas.to_string()}"

def evolucion_precios_producto(nombre_producto: str) -> str:
    """
    Muestra la evolución de precios de costo y venta de un producto específico a lo largo del tiempo.
    """
    df = obtener_ventas_completas()
    df = df[df["producto"] == nombre_producto]
    if df.empty:
        return f"No hay datos de ventas para el producto '{nombre_producto}'."
    df = df.sort_values("fecha")
    salida = "Evolución de precios para '{}':\n".format(nombre_producto)
    salida += df[["fecha", "preciounitario", "precioventafinalunitario"]].to_string(index=False)
    return salida

def margen_promedio_por_marca() -> str:
    """
    Muestra el margen promedio (precio venta - costo) por marca.
    """
    df = obtener_ventas_completas()
    df["margen"] = df["precioventafinalunitario"] - df["preciounitario"]
    resumen = df.groupby("marca")["margen"].mean().sort_values(ascending=False)
    return f"Margen promedio por marca:\n{resumen.round(2).to_string()}"

def productos_con_margen_negativo() -> str:
    """
    Lista productos que se han vendido por debajo de su precio de costo.
    """
    df = obtener_ventas_completas()
    df["margen"] = df["precioventafinalunitario"] - df["preciounitario"]
    negativos = df[df["margen"] < 0]
    if negativos.empty:
        return "No se han detectado productos vendidos por debajo del costo."
    resumen = negativos.groupby("producto")["margen"].count()
    return f"Productos vendidos por debajo del costo (veces):\n{resumen.to_string()}"

def proporcion_ventas_por_categoria(periodo: str = "mes") -> str:
    """
    Porcentaje de ventas de cada categoría respecto al total, por periodo.
    """
    df = obtener_ventas_completas()
    df["fecha"] = pd.to_datetime(df["fecha"])
    if periodo == "año":
        df["periodo"] = df["fecha"].dt.year
    elif periodo == "mes":
        df["periodo"] = df["fecha"].dt.to_period("M")
    elif periodo == "semana":
        df["periodo"] = df["fecha"].dt.isocalendar().week
        df["año"] = df["fecha"].dt.year
        df["periodo"] = df["año"].astype(str) + "-S" + df["periodo"].astype(str)
    elif periodo == "dia":
        df["periodo"] = df["fecha"].dt.date
    else:
        return "Periodo no soportado. Usa: año, mes, semana o dia."
    resumen = df.groupby(["periodo", "categoria"])["cantidadvendida"].sum()
    total = resumen.groupby("periodo").sum()
    proporcion = resumen / total
    salida = "Proporción de ventas por categoría y periodo (%):\n"
    for p in proporcion.index.levels[0]:
        salida += f"\nPeriodo: {p}\n"
        sub = proporcion[p].sort_values(ascending=False) * 100
        salida += sub.round(2).to_string()
        salida += "\n"
    return salida.strip()

def stock_valorizado() -> str:
    """
    Valor total del stock actual por producto, marca y categoría.
    """
    df = obtener_inventario_completo()
    df["valor_stock"] = df["stock_disponible"] * df["preciounitario"]
    por_producto = df.groupby("Producto")["valor_stock"].sum().sort_values(ascending=False)
    por_marca = df.groupby("marca")["valor_stock"].sum().sort_values(ascending=False)
    por_categoria = df.groupby("categoria")["valor_stock"].sum().sort_values(ascending=False)
    salida = "Valor total del stock actual:\n"
    salida += "\nPor producto:\n" + por_producto.round(2).to_string()
    salida += "\n\nPor marca:\n" + por_marca.round(2).to_string()
    salida += "\n\nPor categoría:\n" + por_categoria.round(2).to_string()
    return salida

def productos_sin_rotacion(ultimos_n=3, periodo="mes") -> str:
    """
    Lista productos que no han tenido ventas en los últimos N periodos.
    """
    df_ventas = obtener_ventas_completas()
    df_ventas["fecha"] = pd.to_datetime(df_ventas["fecha"])
    if periodo == "año":
        df_ventas["periodo"] = df_ventas["fecha"].dt.year
    elif periodo == "mes":
        df_ventas["periodo"] = df_ventas["fecha"].dt.to_period("M")
    elif periodo == "semana":
        df_ventas["periodo"] = df_ventas["fecha"].dt.isocalendar().week
        df_ventas["año"] = df_ventas["fecha"].dt.year
        df_ventas["periodo"] = df_ventas["año"].astype(str) + "-S" + df_ventas["periodo"].astype(str)
    elif periodo == "dia":
        df_ventas["periodo"] = df_ventas["fecha"].dt.date
    else:
        return "Periodo no soportado. Usa: año, mes, semana o dia."
    periodos_ordenados = df_ventas["periodo"].drop_duplicates().sort_values()
    ultimos_periodos = periodos_ordenados.iloc[-ultimos_n:]
    vendidos = df_ventas[df_ventas["periodo"].isin(ultimos_periodos)]["producto"].unique()
    df_inv = obtener_inventario_completo()
    sin_rotacion = set(df_inv["Producto"]) - set(vendidos)
    if not sin_rotacion:
        return "Todos los productos han tenido ventas recientes."
    return "Productos sin rotación en los últimos {} {}s:\n{}".format(ultimos_n, periodo, "\n".join(sin_rotacion))

# ==================== FUNCION PRINCIPAL ====================
def main():
    """
    Función principal que orquesta la ejecución del agente:
    - Captura los parámetros de entrada
    - Inicializa herramientas y modelo
    - Ejecuta el agente y retorna la respuesta
    """
    # ==================== TOOLS DE INVENTARIO ====================
    tool_stock_total = Tool(
        name="stock_total",
        func=lambda _: stock_total(),
        description="Devuelve el stock total disponible en el inventario."
    )
    tool_precio_promedio = Tool(
        name="precio_promedio",
        func=lambda _: precio_promedio(),
        description="Devuelve el precio unitario promedio del inventario."
    )
    tool_producto_mas_caro = Tool(
        name="producto_mas_caro",
        func=lambda _: producto_mas_caro(),
        description="Devuelve el producto más caro del inventario."
    )
    tool_producto_mas_barato = Tool(
        name="producto_mas_barato",
        func=lambda _: producto_mas_barato(),
        description="Devuelve el producto más barato del inventario."
    )
    tool_inventario_precios = Tool(
        name="inventario_con_precios",
        func=lambda _: inventario_con_precios(),
        description="Devuelve el listado de productos con stock y precio unitario."
    )
    tool_stock_por_categoria = Tool(
        name="stock_por_categoria",
        func=lambda _: stock_por_categoria(),
        description="Devuelve el stock total disponible agrupado por categoría."
    )
    tool_stock_por_marca = Tool(
        name="stock_por_marca",
        func=lambda _: stock_por_marca(),
        description="Devuelve el stock total disponible agrupado por marca."
    )

    # ==================== TOOLS DE VENTAS ====================
    tool_productos_mas_vendidos = Tool(
        name="productos_mas_vendidos",
        func=lambda _: productos_mas_vendidos(),
        description="Devuelve el top 5 de productos más vendidos por cantidad total."
    )
    tool_variacion_precios = Tool(
        name="variacion_precios",
        func=lambda _: variacion_precios(),
        description="Muestra la variación promedio entre el precio de compra y el precio de venta final por producto."
    )
    tool_ventas_por_mes = Tool(
        name="ventas_por_mes",
        func=lambda _: ventas_por_tiempo("mes"),
        description="Devuelve el total de ventas agrupado por mes."
    )
    tool_ventas_por_anio = Tool(
        name="ventas_por_anio",
        func=lambda _: ventas_por_tiempo("año"),
        description="Devuelve el total de ventas agrupado por año."
    )
    tool_ventas_por_semana = Tool(
        name="ventas_por_semana",
        func=lambda _: ventas_por_tiempo("semana"),
        description="Devuelve el total de ventas agrupado por semana."
    )
    tool_ventas_por_dia = Tool(
        name="ventas_por_dia",
        func=lambda _: ventas_por_tiempo("dia"),
        description="Devuelve el total de ventas agrupado por día."
    )
    tool_ganancia_total_por_categoria = Tool(
        name="ganancia_total_por_categoria",
        func=lambda _: ganancia_total_por_categoria(),
        description="Devuelve la ganancia total agrupada por categoría."
    )
    tool_ganancia_total_por_marca = Tool(
        name="ganancia_total_por_marca",
        func=lambda _: ganancia_total_por_marca(),
        description="Devuelve la ganancia total agrupada por marca."
    )
    tool_cantidad_vendida_por_categoria = Tool(
        name="cantidad_vendida_por_categoria",
        func=lambda _: cantidad_vendida_por_categoria(),
        description="Devuelve la cantidad total de productos vendidos agrupada por categoría."
    )
    tool_cantidad_vendida_por_marca = Tool(
        name="cantidad_vendida_por_marca",
        func=lambda _: cantidad_vendida_por_marca(),
        description="Devuelve la cantidad total de productos vendidos agrupada por marca."
    )

    # ==================== TOOLS DE UTILIDAD Y CONTEXTO ====================
    tool_buscar_similares = Tool(
        name="buscar_productos_similares",
        func=lambda nombre: buscar_productos_similares(nombre),
        description="Busca productos similares por nombre si el usuario no lo escribe correctamente."
    )
    tool_buscar_semantico = Tool(
        name="buscar_producto_semantico",
        func=lambda nombre: buscar_producto_semantico(nombre),
        description="Busca productos similares usando embeddings si el nombre ingresado es muy diferente al real."
    )
    tool_resumir_contexto = Tool(
        name="resumir_contexto",
        func=lambda texto: resumir_contexto(texto),
        description="Resume un texto largo para reducir el número de tokens y facilitar el análisis."
    )
    tool_alertas_automaticas = Tool(
        name="alertas_automaticas",
        func=lambda _: alertas_automaticas(),
        description="Analiza los datos y notifica automáticamente sobre caídas bruscas en ventas, productos con stock bajo o incrementos inesperados de precios."
    )
    tool_obtener_alertas_contexto = Tool(
        name="obtener_alertas_contexto",
        func=lambda _: obtener_alertas_contexto(),
        description="Obtiene las alertas automáticas SOLO para usarlas como contexto interno del modelo. NUNCA mostrar la salida de esta tool directamente al usuario, solo usarla para razonar internamente."
    )

    # ==================== TOOLS DE ANÁLISIS TEMPORAL ====================
    tool_cant_ganancia_categoria = Tool(
        name="cantidad_y_ganancia_por_categoria",
        func=lambda args: cantidad_vendida_y_ganancia_por_categoria(
            periodo=args.get("periodo", "semana"),
            ultimos_n=int(args.get("ultimos_n", 4))
        ),
        description="Devuelve la cantidad vendida y la ganancia real agrupada por categoría en los últimos N periodos (año, mes, semana, día). Parámetros: periodo (str), ultimos_n (int)."
    )
    tool_cant_ganancia_marca = Tool(
        name="cantidad_y_ganancia_por_marca",
        func=lambda args: cantidad_vendida_y_ganancia_por_marca(
            periodo=args.get("periodo", "semana"),
            ultimos_n=int(args.get("ultimos_n", 4))
        ),
        description="Devuelve la cantidad vendida y la ganancia real agrupada por marca en los últimos N periodos (año, mes, semana, día). Parámetros: periodo (str), ultimos_n (int)."
    )
    tool_cant_ganancia_producto = Tool(
        name="cantidad_y_ganancia_por_producto",
        func=lambda args: cantidad_vendida_y_ganancia_por_producto(
            periodo=args.get("periodo", "semana"),
            ultimos_n=int(args.get("ultimos_n", 4))
        ),
        description="Devuelve la cantidad vendida y la ganancia real agrupada por producto en los últimos N periodos (año, mes, semana, día). Parámetros: periodo (str), ultimos_n (int)."
    )

    # ==================== TOOLS DE ANÁLISIS AVANZADO ====================
    tool_rentabilidad_producto = Tool(
        name="rentabilidad_por_producto",
        func=lambda _: rentabilidad_por_producto(),
        description="Devuelve la rentabilidad (ganancia/ingreso) de cada producto, ordenando del más rentable al menos rentable."
    )
    tool_productos_menos_vendidos = Tool(
        name="productos_menos_vendidos",
        func=lambda _: productos_menos_vendidos(),
        description="Devuelve el top 5 de productos menos vendidos por cantidad total."
    )
    tool_evolucion_precios_producto = Tool(
        name="evolucion_precios_producto",
        func=lambda args: evolucion_precios_producto(args.get("producto", "")),
        description="Muestra la evolución de precios de costo y venta de un producto específico a lo largo del tiempo. Parámetro: producto (str)."
    )
    tool_margen_promedio_marca = Tool(
        name="margen_promedio_por_marca",
        func=lambda _: margen_promedio_por_marca(),
        description="Muestra el margen promedio (precio venta - costo) por marca."
    )
    tool_productos_margen_negativo = Tool(
        name="productos_con_margen_negativo",
        func=lambda _: productos_con_margen_negativo(),
        description="Lista productos que se han vendido por debajo de su precio de costo."
    )
    tool_proporcion_ventas_categoria = Tool(
        name="proporcion_ventas_por_categoria",
        func=lambda args: proporcion_ventas_por_categoria(args.get("periodo", "mes")),
        description="Porcentaje de ventas de cada categoría respecto al total, por periodo (año, mes, semana, día). Parámetro: periodo (str)."
    )
    tool_stock_valorizado = Tool(
        name="stock_valorizado",
        func=lambda _: stock_valorizado(),
        description="Valor total del stock actual por producto, marca y categoría."
    )
    tool_productos_sin_rotacion = Tool(
        name="productos_sin_rotacion",
        func=lambda args: productos_sin_rotacion(args.get("ultimos_n", 3), args.get("periodo", "mes")),
        description="Lista productos que no han tenido ventas en los últimos N periodos (año, mes, semana, día). Parámetros: ultimos_n (int), periodo (str)."
    )

    # ==================== AGRUPACIÓN DE TOOLS POR CATEGORÍA ====================
    toolkit = [
        # === INVENTARIO ===
        tool_stock_total,
        tool_precio_promedio,
        tool_producto_mas_caro,
        tool_producto_mas_barato,
        tool_inventario_precios,
        tool_stock_por_categoria,
        tool_stock_por_marca,

        # === VENTAS ===
        tool_productos_mas_vendidos,
        tool_productos_menos_vendidos,
        tool_ganancia_total_por_categoria,
        tool_ganancia_total_por_marca,
        tool_cantidad_vendida_por_categoria,
        tool_cantidad_vendida_por_marca,
        tool_variacion_precios,

        # === ANÁLISIS TEMPORAL ===
        tool_ventas_por_mes,
        tool_ventas_por_anio,
        tool_ventas_por_semana,
        tool_ventas_por_dia,
        tool_cant_ganancia_categoria,
        tool_cant_ganancia_marca,
        tool_cant_ganancia_producto,

        # === ANÁLISIS AVANZADO ===
        tool_rentabilidad_producto,
        tool_evolucion_precios_producto,
        tool_margen_promedio_marca,
        tool_productos_margen_negativo,
        tool_proporcion_ventas_categoria,
        tool_stock_valorizado,
        tool_productos_sin_rotacion,

        # === UTILIDAD Y CONTEXTO ===
        tool_buscar_similares,
        tool_buscar_semantico,
        tool_resumir_contexto,
        tool_alertas_automaticas,
        tool_obtener_alertas_contexto
    ]

    # Capturamos variables enviadas
    id_agente = request.args.get('idagente')
    msg = request.args.get('msg')

    # Intentar parsear periodo y ultimos_n del mensaje
    periodo, ultimos_n = parsear_periodo_desde_texto(msg or "")

    # datos de configuracion
    DB_URI = os.environ.get(
        "DB_URI",
        "postgresql://@usuario:@contraseña@34.46.222.230:5432/mibasedatos?sslmode=disable"
    )
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }

    # Inicializamos la memoria
    with ConnectionPool(
            conninfo=DB_URI,
            max_size=20,
            kwargs=connection_kwargs,
    ) as pool:
        checkpointer = PostgresSaver(pool)

        # Inicializamos el modelo
        model = ChatOpenAI(model="gpt-4.1-2025-04-14")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """
                Te llamas Uzimil-IA y eres un analista de datos comercial con experiencia en desarrollo de aplicaciones, 
                análisis de datos y análisis comercial. Sabes usar Python para interpretar y analizar la información de 
                la empresa, y tienes mucho conocimiento sobre los productos de informática que se tiene en el stock como 
                por ejemplo computadoras, laptopts, tablets, tarjetas de videos, celulares, impresoras, router de red y 
                entre otros componentes de infórmatica.

                Utiliza únicamente las herramientas disponibles para responder y brindar información. 
                Si no cuentas con una herramienta específica para resolver una pregunta, infórmalo claramente e indica cómo puedes ayudar.

                Tu objetivo es guiar al usuario de forma amigable, breve y conversacional. Sigue estos pasos:

                1.Inicio de la conversación: Da un saludo cálido al iniciar la conversación. Luego, preséntate y menciona todas tus capacidades 
                como agente, basándote en tu experiencia y herramientas disponibilizadas. Finalmente, pregunta al usuario qué necesita hoy.

                2.Análisis de datos: Utiliza las herramientas para realizar un análisis óptimo del mensaje del usuario y de los datos.

                3.Buscar productos similares: Si no entiendes sobre qué productos está hablando el usuario, utiliza tool_buscar_similares y 
                tool_buscar_semantico para obtener más información sobre productos mencionados.

                4.Resumir contexto: Usa tool_resumir_contexto para resumir el contexto de la conversación con el objetivo de reducir la cantidad 
                de tokens enviados al modelo LLM.

                5.Temas prohibidos: Evita salir del foco tocando temas de política, filosofía, temas tabú en la sociedad, sexo, lenguaje vulgar 
                o cualquier contenido explícito que pueda conllevar una conversación inapropiada.

                6.Restricciones de alcance: No generes ayuda, tutoriales ni asesoramiento sobre herramientas informáticas externas como Power BI, 
                Excel, u otras. Limítate exclusivamente a las herramientas que tienes y al contexto relacionado con la información de ventas y stock.                 
                """),
                ("human", "{messages}"),
            ]
        )
        # inicializamos el agente
        agent_executor = create_react_agent(model, toolkit, checkpointer=checkpointer, prompt=prompt)

        # Si el mensaje requiere análisis temporal y se detectó periodo y ultimos_n, modificar el mensaje para que el agente lo interprete
        if periodo and ultimos_n:
            # Reescribir el mensaje para que el agente lo entienda como parámetros
            msg_mod = f"{msg}\n[PARÁMETROS_ANALISIS_TEMPORAL] periodo={periodo}, ultimos_n={ultimos_n}"
            response = agent_executor.invoke({"messages": [HumanMessage(content=msg_mod)]}, config={"configurable": {"thread_id": id_agente}})
        else:
            response = agent_executor.invoke({"messages": [HumanMessage(content=msg)]}, config={"configurable": {"thread_id": id_agente}})
        return response['messages'][-1].content


if __name__ == '__main__':
    # La aplicación escucha en el puerto 8080, requerido por Cloud Run
    app.run(host='0.0.0.0', port=8080)