"use client";
import React from "react";
import { createPortal } from "react-dom";

export default function CapacidadesBot() {
  const [showModal, setShowModal] = React.useState(false);
  return (
    <div className="w-full mt-6">
      <div
        className="text-lg font-bold mb-2 text-center"
        style={{
          letterSpacing: "-0.04em",
          color: "transparent",
          background: "linear-gradient(90deg, #fff 40%, #00e0ff 60%, #00ffb3 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          textShadow: "0 4px 24px #00e0ff88, 0 1px 0 #fff, 0 8px 32px #00ffb388",
          fontFamily: "Poppins, Segoe UI, Arial, sans-serif",
        }}
      >
        ¿Qué puede hacer Uzimil-IA?
      </div>
      <div className="text-white text-sm leading-relaxed max-h-72 overflow-y-auto pr-2">
        <ul className="list-disc list-inside px-4 pr-6 space-y-4">
          <li>
            <b>Análisis de ventas</b>: consulta ventas y ganancias por producto, categoría o marca en distintos periodos. Evolución y tendencias de ventas, productos más/menos vendidos y sin rotación.
          </li>
          <li>
            <b>Inventario</b>: consulta stock total, por categoría o marca, valor total del stock, productos con bajo stock y lista de precios actuales.
          </li>
          <li>
            <b>Rentabilidad y precios</b>: analiza margen, rentabilidad, evolución y variaciones de precios, ventas bajo costo y precios promedio.
          </li>
          <li>
            <b>Comparaciones</b>: compara ventas, ganancias y rentabilidad entre periodos, detecta tendencias y caídas bruscas.
          </li>
          <li>
            <b>Búsqueda inteligente</b>: busca productos similares aunque el nombre esté incompleto o mal escrito, y compara productos, marcas y categorías.
          </li>
          <li>
            <b>Alertas y resúmenes</b>: resume hallazgos importantes y alerta sobre caídas de ventas, bajo stock o cambios inusuales de precios.
          </li>
        </ul>
        <div className="flex justify-center w-full mt-6">
          <button
            className="max-w-[180px] w-full py-2 px-4 bg-gradient-to-r from-[#00e0ff] to-[#00ffb3] hover:from-[#00b8d9] hover:to-[#00cfa3] text-[#181a20] text-base font-semibold rounded-lg shadow transition-all duration-200 border-0 focus:outline-none focus:ring-2 focus:ring-[#00e0ff] focus:ring-offset-2"
            style={{ fontWeight: 'bold', fontSize: '1.05rem', boxShadow: '0 4px 16px #00e0ff33' }}
            onClick={() => setShowModal(true)}
          >
            <span className="flex items-center justify-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9 9 4.03 9 9z" /></svg>
              Ver todas las capacidades
            </span>
          </button>
        </div>
        {/* Modal de capacidades completas */}
        {showModal && typeof window !== 'undefined' && createPortal(
          <div className="fixed inset-0 z-[1000] flex items-center justify-center" style={{minHeight:'100vh', minWidth:'100vw', overflow:'visible'}}>
            {/* Fondo opaco */}
            <div className="fixed inset-0 bg-black bg-opacity-70 transition-opacity" onClick={() => setShowModal(false)} />
            {/* Caja del modal */}
            <div className="relative z-10 rounded-2xl shadow-2xl max-w-lg w-[95vw] sm:w-[80vw] md:w-[60vw] lg:w-[40vw] pt-6 p-6 flex flex-col items-center border border-gray-700" style={{backdropFilter:'blur(6px)', background: 'rgba(20, 22, 34, 0.98)'}}>
              {/* Botón de cerrar en la esquina superior derecha del modal */}
              <button
                className="absolute top-4 right-4 w-10 h-10 flex items-center justify-center bg-gradient-to-br from-red-600 via-red-500 to-yellow-400 hover:from-red-700 hover:to-yellow-500 text-white text-2xl font-bold rounded-full shadow-2xl border-2 border-white transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-yellow-400"
                onClick={() => setShowModal(false)}
                aria-label="Cerrar"
                style={{boxShadow: '0 0 0 4px #181a20, 0 4px 24px #ff000088', zIndex: 20}}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                  <path fillRule="evenodd" d="M10 8.586l4.95-4.95a1 1 0 111.414 1.414L11.414 10l4.95 4.95a1 1 0 01-1.414 1.414L10 11.414l-4.95 4.95a1 1 0 01-1.414-1.414L8.586 10l-4.95-4.95A1 1 0 115.05 3.636L10 8.586z" clipRule="evenodd" />
                </svg>
              </button>
              <div className="w-full mb-2 flex items-center justify-center">
                <h2 className="text-2xl font-bold text-white w-full text-center">Capacidades completas de UZIMIL-IA</h2>
              </div>
              <div className="text-xs text-gray-300 mb-4 text-center w-full">Versión 1.0</div>
              <ol className="list-decimal list-inside text-gray-200 space-y-3 text-base w-full max-h-[60vh] overflow-y-auto px-2">
                <li>
                  <b>Análisis de Ventas</b>
                  <ul className="list-disc ml-6 mt-1 text-sm">
                    <li>Consultar cantidad de productos vendidos y ganancias por producto, categoría o marca en distintos periodos (día, semana, mes, año).</li>
                    <li>Mostrar evolución de ventas en el tiempo.</li>
                    <li>Identificar productos más y menos vendidos.</li>
                    <li>Detectar productos sin ventas recientes.</li>
                    <li>Calcular proporción de ventas por categoría.</li>
                  </ul>
                </li>
                <li>
                  <b>Gestión y análisis de inventario</b>
                  <ul className="list-disc ml-6 mt-1 text-sm">
                    <li>Consultar stock disponible total, por categoría y marca.</li>
                    <li>Ver valor total del stock, detallado por producto, marca o categoría.</li>
                    <li>Revisar productos con bajo stock.</li>
                    <li>Listar productos con precios actuales y stock disponible.</li>
                  </ul>
                </li>
                <li>
                  <b>Rentabilidad y precios</b>
                  <ul className="list-disc ml-6 mt-1 text-sm">
                    <li>Analizar ganancia total, margen y rentabilidad por producto, marca o categoría.</li>
                    <li>Identificar productos vendidos por debajo del costo.</li>
                    <li>Ver evolución del precio de costo y venta en el tiempo.</li>
                    <li>Calcular variaciones promedio entre precios de compra y venta.</li>
                    <li>Mostrar precio promedio por categoría, producto o marca.</li>
                  </ul>
                </li>
                <li>
                  <b>Comparaciones y tendencias</b>
                  <ul className="list-disc ml-6 mt-1 text-sm">
                    <li>Comparar ventas y ganancias entre diferentes periodos.</li>
                    <li>Mostrar tendencias, caídas bruscas en ventas o incrementos inesperados de precios.</li>
                    <li>Ver rentabilidad ordenada de mayor a menor.</li>
                  </ul>
                </li>
                <li>
                  <b>Búsqueda y ayuda con productos</b>
                  <ul className="list-disc ml-6 mt-1 text-sm">
                    <li>Buscar productos similares aunque el nombre esté incompleto o mal escrito.</li>
                    <li>Proporcionar detalles y comparaciones de productos, marcas y categorías.</li>
                  </ul>
                </li>
                <li>
                  <b>Alertas y resumen de contexto</b>
                  <ul className="list-disc ml-6 mt-1 text-sm">
                    <li>Resumir automáticamente el contexto de análisis y hallazgos importantes.</li>
                    <li>Detectar alertas como caídas de ventas, productos con bajo stock o cambios de precios inusuales.</li>
                  </ul>
                </li>
              </ol>
            </div>
          </div>,
          document.body
        )}
      </div>
    </div>
  );
}
