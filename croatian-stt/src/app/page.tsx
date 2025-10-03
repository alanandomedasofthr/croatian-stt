"use client";

import { useCallback, useMemo, useRef, useState } from "react";

type ChatRole = "system" | "assistant" | "user";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
};

type SegmentMsg = {
  type: "segment";
  srtIndex: number;
  start: number;
  end: number;
  raw: string;
  corrected: string;
  correctedWrapped: string;
};

function secondsToClock(seconds: number): string {
  return new Date(seconds * 1000).toISOString().slice(11, 19);
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<"idle" | "connecting" | "live">("idle");
  const [segments, setSegments] = useState<SegmentMsg[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const WS_URL = process.env.NEXT_PUBLIC_WS_URL;
  if (!WS_URL) {
    throw new Error("NEXT_PUBLIC_WS_URL is not configured");
  }

  const append = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const transcript = useMemo(() => {
    if (!segments.length) return "";
    return segments
      .map((seg) => seg.corrected.trim())
      .join(" ")
      .trim();
  }, [segments]);

  async function start() {
    if (recording) return;
    setStatus("connecting");
    setMessages([]);
    setSegments([]);

    // fetch short-lived JWT
    const res = await fetch("/api/token");
    const { token } = await res.json();

    // mic + worklet
    const media = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = media;
    const ctx = new AudioContext({ sampleRate: 48000 });
    ctxRef.current = ctx;
    await ctx.audioWorklet.addModule("/pcm-worklet.js");
    const src = ctx.createMediaStreamSource(media);
    const node = new AudioWorkletNode(ctx, "pcm-worklet");
    nodeRef.current = node;
    src.connect(node);
    node.connect(ctx.destination); // optional (disconnect to avoid local playback)

    // websocket
    const ws = new WebSocket(`${WS_URL}?token=${encodeURIComponent(token)}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      setRecording(true);
      setStatus("live");
      append({
        id: crypto.randomUUID(),
        role: "system",
        content: "🎙️ Live transcription started.",
      });
    };
    ws.onclose = () => {
      setRecording(false);
      setStatus("idle");
      append({
        id: crypto.randomUUID(),
        role: "system",
        content: "⏹️ Stopped.",
      });
    };
    ws.onerror = () => {
      setRecording(false);
      setStatus("idle");
      append({
        id: crypto.randomUUID(),
        role: "system",
        content: "⚠️ Connection error.",
      });
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg?.type === "segment") {
          const seg = msg as SegmentMsg;
          setSegments((prev) => [...prev, seg]);
        }
      } catch {}
    };
    wsRef.current = ws;

    node.port.onmessage = (e) => {
      const ab: ArrayBuffer = e.data; // Int16 mono 16kHz
      if (ws.readyState === WebSocket.OPEN) ws.send(ab);
    };
  }

  function stop() {
    wsRef.current?.close();
    wsRef.current = null;

    nodeRef.current?.disconnect();
    nodeRef.current = null;

    ctxRef.current?.close();
    ctxRef.current = null;

    streamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });
    streamRef.current = null;

    setRecording(false);
    setStatus("idle");
  }

  return (
    <main className="mx-auto max-w-2xl p-6 space-y-6">
      <h1 className="text-2xl font-semibold">
        Croatian Live Transcription (WhisperX)
      </h1>
      <div className="flex gap-3">
        <button
          type="button"
          onClick={start}
          disabled={recording || status === "connecting"}
          className="rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
        >
          {status === "connecting" ? "Connecting…" : "Start"}
        </button>
        <button
          type="button"
          onClick={stop}
          disabled={!recording}
          className="rounded-xl px-4 py-2 border"
        >
          Stop
        </button>
      </div>

      <section className="space-y-2">
        <h2 className="text-sm uppercase tracking-wide text-zinc-500">
          Live transcript
        </h2>
        <div className="min-h-[120px] rounded-xl border bg-white p-4 shadow-sm">
          <p className="whitespace-pre-wrap text-lg leading-relaxed">
            {transcript
              ? transcript
              : status === "live"
                ? "Listening…"
                : "Press Start to begin."}
          </p>
        </div>
      </section>

      <section className="space-y-2">
        <h2 className="text-sm uppercase tracking-wide text-zinc-500">
          Recent segments
        </h2>
        <div className="space-y-3">
          {segments.map((seg) => (
            <div key={seg.srtIndex} className="rounded-xl border p-3">
              <div className="mb-1 text-xs font-mono uppercase tracking-wide text-zinc-500">
                {secondsToClock(seg.start)} → {secondsToClock(seg.end)}
              </div>
              <div className="whitespace-pre-wrap text-base leading-relaxed">
                {seg.correctedWrapped}
              </div>
              <div className="mt-2 whitespace-pre-wrap text-xs text-zinc-500">
                Raw: {seg.raw}
              </div>
            </div>
          ))}
          {!segments.length && (
            <div className="text-sm text-zinc-500">
              Waiting for the first segment…
            </div>
          )}
        </div>
      </section>

      <section className="space-y-2">
        <h2 className="text-sm uppercase tracking-wide text-zinc-500">
          Session log
        </h2>
        <div className="space-y-3">
          {messages.map((m) => (
            <div
              key={m.id}
              className="rounded-xl border p-3 whitespace-pre-wrap"
            >
              <div className="mb-1 text-xs opacity-60">{m.role}</div>
              <div>{m.content}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="pt-2 text-sm">
        <a
          className="underline"
          href="http://localhost:7860/download/srt"
          target="_blank"
          rel="noopener noreferrer"
        >
          Download SRT
        </a>
        {" · "}
        <a
          className="underline"
          href="http://localhost:7860/download/vtt"
          target="_blank"
          rel="noopener noreferrer"
        >
          Download VTT
        </a>
        {" · "}
        <a
          className="underline"
          href="http://localhost:7860/download/corrected"
          target="_blank"
          rel="noopener noreferrer"
        >
          TXT
        </a>
      </div>
    </main>
  );
}
