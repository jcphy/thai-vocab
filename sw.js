// 離線快取：網頁本體「網路優先、失敗用快取」，音檔「快取優先」（iOS 需要支援 Range 請求，所以自己切片回應）
const SHELL = "shell-v1", AUDIO = "audio-v1";
const CORE = ["./", "index.html", "manifest.webmanifest", "icon-180.png", "icon-192.png", "icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(SHELL).then(c => Promise.all(CORE.map(u => c.add(u).catch(() => {})))).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== SHELL && k !== AUDIO).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

async function rangeResponse(full, rangeHdr){
  const m = /bytes=(\d*)-(\d*)/.exec(rangeHdr || "");
  if(!m) return full;
  const buf = await full.arrayBuffer();
  const size = buf.byteLength;
  let start = m[1] === "" ? Math.max(0, size - parseInt(m[2], 10)) : parseInt(m[1], 10);
  let end = m[1] === "" || m[2] === "" ? size - 1 : Math.min(parseInt(m[2], 10), size - 1);
  if(start >= size) return new Response(null, {status: 416, headers: {"Content-Range": "bytes */" + size}});
  return new Response(buf.slice(start, end + 1), {status: 206, headers: {
    "Content-Type": full.headers.get("Content-Type") || "audio/mpeg",
    "Content-Length": String(end - start + 1),
    "Content-Range": `bytes ${start}-${end}/${size}`,
    "Accept-Ranges": "bytes"
  }});
}

async function audioFetch(req){
  const cache = await caches.open(AUDIO);
  let hit = await cache.match(req.url);
  if(!hit){
    const net = await fetch(req.url);          // 不帶 Range，抓完整檔
    if(net.ok) cache.put(req.url, net.clone()).catch(() => {});
    hit = net;
  }
  return rangeResponse(hit, req.headers.get("range"));
}

async function networkFirst(req){
  const cache = await caches.open(SHELL);
  try{
    const net = await fetch(req);
    if(net.ok || net.type === "opaque") cache.put(req, net.clone()).catch(() => {});
    return net;
  }catch(err){
    const hit = await cache.match(req, {ignoreSearch: req.mode !== "navigate" ? false : true}) || (req.mode === "navigate" ? await cache.match("index.html") || await cache.match("./") : null);
    if(hit) return hit;
    throw err;
  }
}

self.addEventListener("fetch", e => {
  const req = e.request;
  if(req.method !== "GET") return;
  const u = new URL(req.url);
  if(u.origin === location.origin && /\.mp3$/i.test(u.pathname)){ e.respondWith(audioFetch(req)); return; }
  if(u.origin === location.origin || /(^|\.)(googleapis|gstatic)\.com$/.test(u.hostname)) e.respondWith(networkFirst(req));
});
