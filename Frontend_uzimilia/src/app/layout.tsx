// src/app/layout.tsx

import './globals.css';
import type { ReactNode } from 'react';
import AuthProvider from './AuthProvider';
import CapacidadesBot from './CapacidadesBot';

export const metadata = {
  title: 'UZIMIL-IA',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="es">
      <body className="flex h-screen bg-gray-50">
        <AuthProvider>
          <div className="flex w-full h-screen">
            {/* Panel lateral izquierdo */}
            <aside className="w-64 min-w-[16rem] max-w-[18rem] bg-[linear-gradient(180deg,_#103955_0%,_#253161_100%)] flex flex-col items-center justify-start py-10 px-4 shadow-lg">
              <div className="w-full flex flex-col items-center justify-start mb-8">
                <span
                  className="font-extrabold text-[2.8rem] md:text-[3.5rem] mb-4 tracking-tight text-center w-full"
                  style={{
                    letterSpacing: '-0.04em',
                    color: 'transparent',
                    background: 'linear-gradient(90deg, #fff 40%, #00e0ff 60%, #00ffb3 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    textShadow: '0 4px 24px #00e0ff88, 0 1px 0 #fff, 0 8px 32px #00ffb388',
                    fontFamily: 'Poppins, Segoe UI, Arial, sans-serif',
                  }}
                >
                  UZIMIL-IA
                </span>
                <img src="/robot-white.svg" alt="Robot" className="w-20 h-20 mt-2 mb-4 opacity-90 mx-auto" style={{maxWidth:'5rem', maxHeight:'5rem'}} />
            </div>
            {/* Espacio extra entre el logo y la lista */}
            <div className="w-full" style={{ minHeight: '2.5rem' }} />
            {/* Resumen de capacidades del bot (componente cliente) */}
            <CapacidadesBot />
            </aside>
            {/* Contenido principal */}
            <main className="flex-1 bg-[var(--background)] w-full h-full p-0 overflow-hidden">
              {children}
            </main>
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}
