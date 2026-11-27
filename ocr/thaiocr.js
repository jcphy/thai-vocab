// 泰文子音手寫辨識（小型 CNN，純 JavaScript，不需要其他套件）。模型以 Burapha-TH 資料集訓練。
var ThaiOCR = (function(){
  let M = null;
  function load(m){
    const bin = atob(m.w), buf = new Uint8Array(bin.length); for(let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    let off = 0; const f32 = n => { const a = new Float32Array(buf.buffer.slice(off, off + n * 4)); off += n * 4; return a; };
    const layers = m.layers.map(L => {
      const out = L.s[0], size = L.s.reduce((a, b) => a * b, 1), per = size / out;
      const q = new Int8Array(buf.buffer, off, size); off += size;
      const sc = f32(out), bias = f32(out), W = new Float32Array(size);
      for(let o = 0; o < out; o++) for(let k = 0; k < per; k++) W[o * per + k] = q[o * per + k] * sc[o];
      return {t:L.t, s:L.s, p:L.p, W, bias};
    });
    M = {chars:m.chars, layers};
  }
  function conv(x, c, n, L){   // 3x3, padding 1, 之後接 ReLU
    const oc = L.s[0], out = new Float32Array(oc * n * n);
    for(let o = 0; o < oc; o++){
      const base = o * n * n; out.fill(L.bias[o], base, base + n * n);
      for(let i = 0; i < c; i++) for(let ky = 0; ky < 3; ky++) for(let kx = 0; kx < 3; kx++){
        const w = L.W[((o * c + i) * 3 + ky) * 3 + kx]; if(w === 0) continue;
        const dy = ky - 1, dx = kx - 1, y0 = Math.max(0, -dy), y1 = Math.min(n, n - dy), x0 = Math.max(0, -dx), x1 = Math.min(n, n - dx);
        for(let y = y0; y < y1; y++){ const r = base + y * n, s = i * n * n + (y + dy) * n + dx; for(let X = x0; X < x1; X++) out[r + X] += w * x[s + X]; }
      }
      for(let k = base; k < base + n * n; k++) if(out[k] < 0) out[k] = 0;
    }
    return out;
  }
  function pool(x, c, n){
    const h = n >> 1, out = new Float32Array(c * h * h);
    for(let i = 0; i < c; i++) for(let y = 0; y < h; y++) for(let X = 0; X < h; X++){ const p = i * n * n + 2 * y * n + 2 * X; out[i * h * h + y * h + X] = Math.max(x[p], x[p + 1], x[p + n], x[p + n + 1]); }
    return out;
  }
  function forward(img){   // img: Float32Array 64x64, 0..1（筆跡為 1）
    let x = img, c = 1, n = 64;
    for(const L of M.layers){
      if(L.t === "conv"){ x = conv(x, c, n, L); c = L.s[0]; if(L.p){ x = pool(x, c, n); n >>= 1; } }
      else { const [o, k] = L.s, y = new Float32Array(o); for(let i = 0; i < o; i++){ let v = L.bias[i]; for(let j = 0; j < k; j++) v += L.W[i * k + j] * x[j]; y[i] = v; } x = y; }
    }
    const mx = Math.max(...x), e = Array.from(x, v => Math.exp(v - mx)), sum = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / sum);
  }
  // strokes: [[[x,y],...],...]（任何一致的單位）。格式比照訓練資料：置中、長邊縮成 52、固定粗細
  function rasterize(strokes){
    const pts = strokes.flat(); if(!pts.length) return null;
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x0 = Math.min(...xs), y0 = Math.min(...ys), w = Math.max(...xs) - x0, h = Math.max(...ys) - y0;
    if(Math.max(w, h) <= 0) return null;
    const s = 52 / Math.max(w, h), c = document.createElement("canvas"); c.width = c.height = 256;
    const g = c.getContext("2d", {willReadFrequently:true}); g.strokeStyle = "#000"; g.lineWidth = 12; g.lineCap = g.lineJoin = "round";
    const tr = p => [((p[0] - x0 - w / 2) * s + 32) * 4, ((p[1] - y0 - h / 2) * s + 32) * 4];
    strokes.forEach(st => { if(!st.length) return; g.beginPath(); const a = tr(st[0]); g.moveTo(a[0], a[1]); g.lineTo(a[0] + .1, a[1]); st.forEach(p => { const q = tr(p); g.lineTo(q[0], q[1]); }); g.stroke(); });
    const d = g.getImageData(0, 0, 256, 256).data, out = new Float32Array(4096);
    for(let y = 0; y < 256; y++) for(let X = 0; X < 256; X++) out[(y >> 2) * 64 + (X >> 2)] += d[(y * 256 + X) * 4 + 3] / 255 / 16;
    return out;
  }
  function recognize(strokes, k){
    const img = rasterize(strokes); if(!img) return [];
    return Array.from(forward(img), (p, i) => ({ch:M.chars[i], p})).sort((a, b) => b.p - a.p).slice(0, k || 3);
  }
  return {load, forward, rasterize, recognize};
})();
