// src/app/page.tsx
'use client';

import { useSession, signIn, signOut } from 'next-auth/react';
import React, { useState, FormEvent } from 'react';

// type Mensaje = { de: 'usuario' | 'bot'; texto: string };
type Mensaje = { de: 'usuario' | 'bot'; texto: string; hora?: string };

export default function Page() {
  const { data: session } = useSession();
  const [chat, setChat] = useState<Mensaje[]>([]);
  const [msg, setMsg] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  // Si no hay sesión, mostramos botón de login
  if (!session) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="bg-white rounded-xl p-8 flex flex-col items-center gap-4 max-w-md border-2 border-dotted border-white shadow-[0_0_24px_4px_rgba(255,255,255,0.7)]" style={{boxShadow:'0 0 32px 4px #fff8, 0 0 0 2px #fff8'}}>
          <span className="text-[4.5rem] md:text-[6rem] font-extrabold mb-2 tracking-tight text-center w-full"
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
          <img src="/robot-white.svg" alt="Robot" className="w-20 h-20 mb-2 max-w-[64px] max-h-[64px] mx-auto" style={{filter:'drop-shadow(0 2px 8px #aaa8)'}} />
          <br />
          <span className="text-base text-gray-700 mb-6 text-center">Bienvenido, inicia sesión para comenzar a chatear con tu analista IA comercial.</span>
          <br />
          <br />
          <div className="w-full flex justify-center">
            <button
              onClick={() => signIn('google')}
              className="flex items-center w-full max-w-xs bg-white border border-[#e0e0e0] rounded transition hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-400 px-0 py-0 overflow-hidden"
              style={{height:'48px'}}>
              <span className="flex items-center justify-center h-full w-12 border-r border-[#e0e0e0] bg-white">
                <img src="/google-logo.svg" alt="Google" className="w-6 h-6" />
              </span>
              <span className="flex-1 text-[#222] text-base font-medium text-center">Continuar con Google</span>
            </button>
          </div>
        </div>
      </div>
    );
  } // Fin del bloque de login
  // Generar o recuperar un id de sesión único por usuario
  const getOrCreateSessionId = () => {
    if (typeof window === 'undefined') return '';
    let sessionId = localStorage.getItem('uzimil_session_id');
    if (!sessionId) {
      sessionId = `${Date.now()}-${Math.random().toString(36).substring(2, 10)}`;
      localStorage.setItem('uzimil_session_id', sessionId);
    }
    return sessionId;
  };

  // Función para enviar mensaje
  const enviar = async (e: FormEvent) => {
    e.preventDefault();
    if (!msg) return;
    setLoading(true);
    const sessionId = getOrCreateSessionId();
    const res = await fetch(
      `/api/agent?idagente=${encodeURIComponent(sessionId)}&msg=${encodeURIComponent(msg)}`
    );
    const texto = await res.text();

    // Actualizar historial
    const hora = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setChat((c) => [
      ...c,
      { de: 'usuario', texto: msg, hora },
      { de: 'bot',     texto, hora }
    ]);

    setMsg('');
    setLoading(false);
  };

  return (
    <div className="flex flex-col w-full h-[100vh] max-h-[100vh] min-h-[100vh] mx-auto bg-[#181a20] p-0 max-w-full">
      {/* Header fijo solo logo */}

      <header className="w-full border-b border-[#23242a] bg-[#23242a] rounded-t-xl px-6 py-3 shadow-md h-16 flex items-center justify-between">
        <div className="flex-1 flex items-center justify-center">
          <span
            className="font-extrabold text-[2rem] md:text-[2.5rem] tracking-tight text-center"
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
            ¡ Bienvenido{session?.user?.name ? ` ${session.user.name}` : ''} !
          </span>
        </div>
        <div className="flex items-center gap-8 ml-4">
          <button
            onClick={() => {
              if (typeof window !== 'undefined') {
                localStorage.removeItem('uzimil_session_id');
              }
              signOut();
            }}
            className="text-lg font-bold bg-[#ff4b4b] hover:bg-[#c0392b] text-white px-8 py-3 rounded-full transition-colors shadow-md border-2 border-[#fff2]"
            style={{ minHeight: '3rem', color: '#fff', textShadow: '0 1px 8px #0008, 0 0 2px #fff', marginLeft: '1.5rem' }}
          >
            Cerrar sesión
          </button>
        </div>
      </header>


      {/* Salto de línea visual entre cerrar sesión y chat */}
      <div style={{height: '1.5rem'}} />

      {/* Salto de línea visual entre cerrar sesión y chat */}
      <div className="w-full h-4" />
      {/* Chat box */}
      <div className="flex-1 w-full overflow-y-auto px-8 py-4 bg-[#181a20] flex flex-col gap-4" style={{ paddingBottom: '7.5rem', marginTop: '0' }}>
        {chat.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 select-none pt-16">
            <svg xmlns='http://www.w3.org/2000/svg' className="w-16 h-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="#bdbdbd"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M21 12c0 4.418-4.03 8-9 8a9.77 9.77 0 01-4.39-1.01L3 21l1.11-3.34C3.4 16.07 3 14.57 3 13c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>
            <span>¡Comienza la conversación!</span>
          </div>
        )}
        {chat.map((m, i) => {
          let contenido: React.ReactNode = m.texto;
          if (m.de === 'bot') {
            try {
              const obj = JSON.parse(m.texto);
              if (obj.result) {
                contenido = (
                  <div className="whitespace-pre-line text-gray-100 text-base">
                    {obj.result}
                  </div>
                );
              }
            } catch {
              contenido = <div className="whitespace-pre-line text-gray-100 text-base">{m.texto}</div>;
            }
          }
          const isUser = m.de === 'usuario';
          return (
            <div
              key={i}
              className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`popup-bubble ${isUser ? 'user' : 'bot'} relative`}
                style={{marginBottom: 12, wordBreak: 'break-word'}}
              >
                {contenido}
                <span className="message-time">{m.hora}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Caja de texto fija abajo */}
      <form onSubmit={enviar} className="w-full flex gap-4 items-center bg-[#23242a] rounded-b-xl shadow-lg px-8 py-6 mt-0 sticky bottom-0 left-0 z-20" style={{height:'6.5rem'}}>
        <input
          className="flex-1 rounded-2xl border-2 border-[#e0e0e0] px-8 py-5 bg-white text-black focus:outline-none focus:ring-2 focus:ring-[#00e0ff] text-lg placeholder:text-gray-500 font-medium shadow-md"
          style={{minHeight:'56px', fontSize:'1.15rem'}} 
          placeholder="Escribe tu mensaje…"
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          disabled={loading}
          required
        />
        <button
          type="submit"
          disabled={loading || !msg.trim()}
          className="flex items-center gap-2 bg-gradient-to-r from-[#00e0ff] to-[#00ffb3] hover:from-[#00b8d9] hover:to-[#00cfa3] text-[#181a20] px-10 py-5 rounded-2xl disabled:opacity-50 transition-all font-semibold shadow-lg text-lg border-0 focus:outline-none focus:ring-2 focus:ring-[#00e0ff]"
          style={{minHeight:'56px', fontWeight:'bold', fontSize:'1.15rem', boxShadow:'0 4px 16px #00e0ff33'}}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="#181a20" strokeWidth={2} style={{marginRight: 6}}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
          {loading ? 'Enviando...' : 'Enviar'}
        </button>
      </form>

    </div>
  );
}

