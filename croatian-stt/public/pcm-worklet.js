class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.downFactor = sampleRate / 16000; // e.g. 48000 -> 3
    this.acc = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];

    const out = [];
    for (let i = 0; i < ch.length; i++) {
      this.acc += 1;
      if (this.acc >= this.downFactor) {
        this.acc -= this.downFactor;
        const s = Math.max(-1, Math.min(1, ch[i]));
        out.push(s < 0 ? s * 0x8000 : s * 0x7fff); // float32 -> int16
      }
    }
    const buf = new Int16Array(out.length);
    for (let i = 0; i < out.length; i++) buf[i] = out[i] | 0;
    this.port.postMessage(buf.buffer, [buf.buffer]);
    return true;
  }
}
registerProcessor("pcm-worklet", PCMWorklet);
