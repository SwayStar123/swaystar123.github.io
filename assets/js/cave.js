/*
 * The cave: pixel rock strata carved around the page's content, overgrown with
 * moss, ivy, vines, bushes, grass, ferns and mushrooms, with veins of ore in
 * the stone.
 *
 * Two pages use it:
 *   - the landing page (`_layouts/canvas.html`): a viewport-sized cavity around
 *     the content plate;
 *   - post pages (`_layouts/post.html`): a shaft the whole height of the
 *     article, rock down both sides, a ceiling above and a floor below.
 *
 * Started life as the Claude Design artboard `Redesign.dc.html` (project
 * "Interactive bouncing balls redesign"), which had balls bouncing around the
 * cavity and lasering the rock away. That simulation is gone — the cave is now
 * a pure function of the layout, painted once and repainted only when the
 * layout changes, so nothing here runs per frame.
 *
 * The file also runs as its own web worker: post pages hand the shaft to a
 * second copy of it, so generating a long article's worth of cave never holds
 * up the page.
 */
(function () {
  'use strict';

  var IN_WORKER = typeof document === 'undefined';
  var SELF_URL = !IN_WORKER && document.currentScript ? document.currentScript.src : null;

  var PLAIN_HOME = 720;   // matches canvas.html's media query
  var PLAIN_POST = 900;   // matches post.css's media query

  // px per cave pixel. Small enough that the cave reads as fine pixel art
  // rather than as big blocks, and 4 lands on whole device pixels at every
  // common display scaling (100, 125, 150, 175, 200%).
  var CELL = 4;
  var BACKDROP = '#12141a';   // open cave. post.css paints the same colour itself.

  // rock strata, cavity mouth -> screen edge: pale slate down to near-black
  var LAYERS = ['#9aa3b2', '#7b8492', '#5c6470', '#434a55', '#2c313a', '#1b1f26', '#0e1014'];
  var SHADOW = 0.66;      // rock in a plant's shadow keeps this much of its brightness
  var MOSS_TINT = ['#2f3f2b', '#465c33'];   // blended into the stratum underneath
  var MOSS_COVER = 0.7;
  var ORE_RARITY = 0.9993;   // share of deep-set rock cells that do NOT start a vein

  // what a cell can have growing on it (or set into it)
  var STEM = 1, LEAF = 2, TIP = 3, LIT = 4, MOSS_D = 5, MOSS_L = 6,
      PINK = 7, AMBER = 8, BLUE = 9, WHITE = 10, ORE = 11;
  var FLAT = [null, '#38492d', '#4e6738', '#6d8d4e', '#8db55f', null, null,
              '#f7768e', '#e0af68', '#a9c1ff', '#d8dce4'];
  var BLOOMS = [PINK, AMBER, BLUE, WHITE];
  // ores take two codes each from ORE up: the body of the vein, then its glint
  var DIAMOND = 0, GOLD = 1, AMETHYST = 2, COPPER = 3, QUARTZ = 4;
  [['#5873b8', '#8fb0f7'], ['#b98a3e', '#f0c66e'], ['#7d62b8', '#bb9af7'],
   ['#a8623f', '#e09a6c'], ['#8d94a3', '#e3e7ee']].forEach(function (o) { FLAT.push(o[0], o[1]); });

  /* ---- palette: every cell ends up as one index into PAL ---- */

  function rgb(hex) { return [1, 3, 5].map(function (i) { return parseInt(hex.slice(i, i + 2), 16); }); }
  function mix(a, b, t) { return a.map(function (v, k) { return Math.round(v * (1 - t) + b[k] * t); }); }

  var T = LAYERS.length;
  var C_ROCK = 1, C_SHADE = C_ROCK + T, C_MOSS_D = C_SHADE + T, C_MOSS_L = C_MOSS_D + T, C_FLAT = C_MOSS_L + T;
  var PAL = [[0, 0, 0]];
  (function () {
    var rock = LAYERS.map(rgb), black = [0, 0, 0], k;
    for (k = 0; k < T; k++) PAL[C_ROCK + k] = rock[k];
    for (k = 0; k < T; k++) PAL[C_SHADE + k] = mix(black, rock[k], SHADOW);
    for (k = 0; k < T; k++) PAL[C_MOSS_D + k] = mix(rock[k], rgb(MOSS_TINT[0]), MOSS_COVER);
    for (k = 0; k < T; k++) PAL[C_MOSS_L + k] = mix(rock[k], rgb(MOSS_TINT[1]), MOSS_COVER);
    for (k = 1; k < FLAT.length; k++) PAL[C_FLAT + k] = FLAT[k] ? rgb(FLAT[k]) : black;
  })();

  /* ---- noise ---- */

  function hash2(x, y) {
    var h = x * 374761393 + y * 668265263;
    h = (h ^ (h >> 13)) * 1274126177;
    return ((h ^ (h >> 16)) & 0x7fffffff) / 0x7fffffff;
  }
  function noise2(x, y) {
    var xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    var u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    var a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
  }
  function fbm(x, y) { return noise2(x, y) * 0.65 + noise2(x * 2.3, y * 2.3) * 0.35; }

  /* ---- the generator ----
     w, h: the area to fill, in px. safe: the box the content lives in — no rock,
     nothing growing. opts.maxMargin caps how far from the content the cavity is
     allowed to open up (the rest is deep rock). Returns one palette index per
     cell, 0 for open cave. */

  function generate(w, h, safe, opts) {
    opts = opts || {};
    var cols = Math.ceil(w / CELL), rows = Math.ceil(h / CELL), n = cols * rows;
    var rock = new Uint8Array(n), tier = new Uint8Array(n), kind = new Uint8Array(n);
    var depth = new Float32Array(n);   // 0 at the cavity mouth -> 1 at the screen edge
    var top = T - 1, gx, gy, i;

    var cap = opts.maxMargin || Infinity;
    var mL = Math.max(1, Math.min(safe.x0, cap)), mR = Math.max(1, Math.min(w - safe.x1, cap));
    var mT = Math.max(1, safe.y0), mB = Math.max(1, h - safe.y1);
    function inSafe(cx, cy) { return cx > safe.x0 && cx < safe.x1 && cy > safe.y0 && cy < safe.y1; }
    function isRock(x, y) { return x >= 0 && y >= 0 && x < cols && y < rows && rock[y * cols + x] === 1; }

    // rock: solid everywhere, with a jagged cavity carved around the content
    for (gy = 0; gy < rows; gy++) for (gx = 0; gx < cols; gx++) {
      var cx = gx * CELL + CELL / 2, cy = gy * CELL + CELL / 2;
      if (inSafe(cx, cy)) continue;
      // how far out this cell sits as a FRACTION of the margin available on its own side,
      // so top/bottom (thin margins) get the same rock as left/right (wide margins)
      var nd = Math.min(1, Math.max(
        cx < safe.x0 ? (safe.x0 - cx) / mL : 0,
        cx > safe.x1 ? (cx - safe.x1) / mR : 0,
        cy < safe.y0 ? (safe.y0 - cy) / mT : 0,
        cy > safe.y1 ? (cy - safe.y1) / mB : 0
      ));
      var carve = 0.1 + 0.62 * Math.pow(fbm(gx * 0.032, gy * 0.032), 1.7);
      if (nd <= carve) continue;
      i = gy * cols + gx;
      rock[i] = 1; depth[i] = nd;
      // strata follow that same fraction, with a little noise so the bands
      // wander. The curve keeps the pale and mid tones for most of the wall and
      // leaves the darkest rock for the last stretch before the screen edge.
      var t = Math.max(0, Math.min(1, nd + (fbm(gx * 0.07, gy * 0.07) - 0.5) * 0.22));
      var tr = Math.round(Math.pow(t, 1.4) * top);
      if (nd > 0.86) {
        // deep rock: break what would be one flat dark band into slabs
        var slab = fbm(gx * 0.05 + 90.5, gy * 0.05 + 12.25);
        if (slab > 0.52) tr = Math.min(tr, top - 1);
        if (slab > 0.66) tr = Math.min(tr, top - 2);
      }
      tier[i] = tr;
    }

    // how many cells in from an exposed face each rock cell is (1..4, 0 = deeper)
    var edge = new Uint8Array(n);
    for (i = 0; i < n; i++) {
      if (!rock[i]) continue;
      gx = i % cols;
      if ((gx > 0 && !rock[i - 1]) || (gx < cols - 1 && !rock[i + 1]) ||
          (i >= cols && !rock[i - cols]) || (i < n - cols && !rock[i + cols])) edge[i] = 1;
    }
    for (var d = 2; d <= 4; d++) for (i = 0; i < n; i++) {
      if (!rock[i] || edge[i]) continue;
      gx = i % cols;
      if ((gx > 0 && edge[i - 1] === d - 1) || (gx < cols - 1 && edge[i + 1] === d - 1) ||
          (i >= cols && edge[i - cols] === d - 1) || (i < n - cols && edge[i + cols] === d - 1)) edge[i] = d;
    }

    // Everything below grows ON the scene rather than only into the gaps: a
    // plant cell may sit over rock as readily as over open cave. First come,
    // first served — except that anything may grow over moss.
    function put(x, y, k) {
      if (x < 0 || y < 0 || x >= cols || y >= rows) return;
      if (inSafe(x * CELL + CELL / 2, y * CELL + CELL / 2)) return;
      var j = y * cols + x, cur = kind[j];
      if (cur && cur !== MOSS_D && cur !== MOSS_L) return;
      kind[j] = k;
    }

    // moss: carpets the rock near its exposed faces and in soft patches further
    // in, thinning out as the rock gets deeper and darker
    for (gy = 0; gy < rows; gy++) for (gx = 0; gx < cols; gx++) {
      i = gy * cols + gx;
      if (!rock[i]) continue;
      var patch = fbm(gx * 0.085 + 31.7, gy * 0.085 + 11.3);
      var near = edge[i] ? (5 - edge[i]) / 4 : 0;
      var want = ((patch - 0.53) * 2.6 + near * 0.26) * (1 - depth[i] * 0.62);
      // a narrow dithered band at the edge of each patch, so it reads as a
      // carpet with a ragged hem rather than as salt and pepper
      if (want <= 0.2 + hash2(gx * 7 + 3, gy * 13 + 1) * 0.3) continue;
      kind[i] = patch + near * 0.1 + (hash2(gx * 19 + 2, gy * 5 + 8) - 0.5) * 0.14 > 0.6 ? MOSS_L : MOSS_D;
    }

    // ledge flora: vines off overhangs, shrubs and grass on ledges, brackets off
    // vertical faces, specks on whatever is left
    var empty = function (x, y) {
      if (x < 0 || y < 0 || x >= cols || y >= rows) return true;
      return !rock[y * cols + x];
    };
    // the tall rock columns either side of the content. Vines have room to fall
    // the full height here, so they run several times longer than the ones
    // hanging off the ceiling.
    var beside = function (x, y) {
      var px = x * CELL + CELL / 2, py = y * CELL + CELL / 2;
      return (px < safe.x0 || px > safe.x1) && py > safe.y0 - 80 && py < safe.y1 + 80;
    };
    var plants = [];
    for (gy = 0; gy < rows; gy++) for (gx = 0; gx < cols; gx++) {
      if (!rock[gy * cols + gx]) continue;
      var seed = hash2(gx * 13 + 7, gy * 29 + 3);
      var side = beside(gx, gy);
      var px, s, len, hh, dx, up, b, bx, bh, reach, blades;

      if (empty(gx, gy + 1) && (side || gy < rows * 0.55) && seed > 0.86) {
        // vine: a stem of cells stepping down, leaf cells budding off it
        px = [];
        var ox = 0;
        len = 6 + ((seed * 100) % 10 | 0);
        if (side) len = Math.round(len * (2 + (seed * 37) % 1.4));
        for (s = 1; s <= len; s++) {
          if (hash2(gx + s, gy * 3) > 0.72) ox += hash2(gx, gy + s) > 0.5 ? 1 : -1;
          px.push([ox, s, STEM]);
          if (s > 1 && hash2(gx * 5 + s, gy) > 0.66) px.push([ox + (s % 2 ? 1 : -1), s, LEAF]);
          if (s === len) px.push([ox, s + 1, hash2(gx * 3, gy + s) > 0.8 ? BLOOMS[(seed * 977 | 0) % 4] : TIP]);
        }
        plants.push([gx, gy, px]);
      } else if (empty(gx, gy - 1) && seed > 0.76 && seed <= 0.86) {
        // shrub: a squat tuft of cells rising off the ledge
        px = [[0, -1, STEM]];
        hh = 2 + ((seed * 60) % 3 | 0);
        for (s = 2; s <= hh; s++) px.push([0, -s, STEM]);
        var dirs = [-1, 1];
        for (var di = 0; di < dirs.length; di++) {
          dx = dirs[di];
          if (hash2(gx + dx * 3, gy) < 0.3) continue;
          up = 1 + ((hash2(gx, gy + dx) * 3) | 0);
          px.push([dx, -up, LEAF]);
          if (hash2(gx * 7, gy + dx) > 0.5) px.push([dx * 2, -up - 1, TIP]);
        }
        px.push([0, -hh - 1, hash2(gx * 9 + 1, gy * 3) > 0.7 ? BLOOMS[(seed * 613 | 0) % 4] : TIP]);
        plants.push([gx, gy, px]);
      } else if (empty(gx, gy - 1) && seed > 0.66 && seed <= 0.76) {
        // grass: a few loose blades of uneven height, no crown
        px = [];
        blades = 2 + ((seed * 90) % 3 | 0);
        for (b = 0; b < blades; b++) {
          bx = b - (blades >> 1);
          bh = 1 + ((hash2(gx + b * 5, gy * 3) * 4) | 0);
          for (s = 1; s <= bh; s++) px.push([bx, -s, s === bh ? TIP : LEAF]);
        }
        plants.push([gx, gy, px]);
      } else if ((empty(gx - 1, gy) || empty(gx + 1, gy)) && seed > 0.56 && seed <= 0.66) {
        // bracket fern off a vertical wall, reaching into the open side
        dx = empty(gx - 1, gy) ? -1 : 1;
        px = [];
        reach = 2 + ((seed * 70) % 3 | 0);
        for (s = 1; s <= reach; s++) {
          px.push([dx * s, 0, STEM]);
          if (hash2(gx * 3 + s, gy * 7) > 0.45) px.push([dx * s, -1, LEAF]);
          if (s === reach) px.push([dx * s, 1, TIP]);
        }
        plants.push([gx, gy, px]);
      } else if (hash2(gx * 11, gy * 17 + 5) > 0.72) {
        // one speck on whichever face is exposed, so plain rock surfaces pick up
        // some texture instead of reading as bare blocks
        px = [];
        if (empty(gx, gy - 1)) px.push([0, -1, STEM]);
        else if (empty(gx - 1, gy)) px.push([-1, 0, STEM]);
        else if (empty(gx + 1, gy)) px.push([1, 0, STEM]);
        else if (empty(gx, gy + 1)) px.push([0, 1, STEM]);
        if (px.length) plants.push([gx, gy, px]);
      }
    }
    var f, p;
    for (f = 0; f < plants.length; f++) {
      for (p = 0; p < plants[f][2].length; p++) {
        put(plants[f][0] + plants[f][2][p][0], plants[f][1] + plants[f][2][p][1], plants[f][2][p][2]);
      }
    }

    for (gy = 0; gy < rows; gy++) for (gx = 0; gx < cols; gx++) {
      i = gy * cols + gx;
      if (!rock[i]) continue;
      var r, s2, x, y;

      // ivy: a creeper that starts at an exposed face and wanders across the
      // rock itself, mostly downhill, shedding leaves either side
      if (edge[i] === 1 && (r = hash2(gx * 17 + 5, gy * 23 + 9)) > 0.925) {
        x = gx; y = gy;
        var vy = hash2(gx * 3 + 1, gy * 7 + 2) > 0.3 ? 1 : -1;
        var drift = hash2(gx * 5 + 4, gy * 11 + 6) > 0.5 ? 1 : -1;
        var run = 7 + ((r * 4000) % 22 | 0);
        for (s2 = 0; s2 < run; s2++) {
          put(x, y, STEM);
          var leafy = hash2(x * 13 + s2, y * 7 + 3);
          if (leafy > 0.38) put(x + (s2 % 2 ? 1 : -1), y, leafy > 0.86 ? LIT : leafy > 0.62 ? TIP : LEAF);
          var turn = hash2(x * 7 + s2 * 3, y * 13 + 1);
          var nx = x + (turn > 0.62 ? drift : turn < 0.12 ? -drift : 0);
          var ny = y + (turn > 0.88 && nx !== x ? 0 : vy);
          if (!isRock(nx, ny)) {
            if (isRock(x, y + vy)) { nx = x; ny = y + vy; }
            else if (isRock(x + drift, y)) { nx = x + drift; ny = y; }
            else break;
          }
          x = nx; y = ny;
        }
      }

      // bush: a clump of foliage rooted at the face, spilling over rock and open
      // cave alike, lit from the upper left
      if (edge[i] === 1 && (r = hash2(gx * 29 + 1, gy * 31 + 7)) > 0.962) {
        var rx = 1.5 + hash2(gx * 3 + 8, gy * 5 + 1) * 2.1, ry = 1.1 + hash2(gx * 7 + 2, gy * 3 + 9) * 1.5;
        for (var oy = -3; oy <= 3; oy++) for (var ox2 = -4; ox2 <= 4; ox2++) {
          var q = (ox2 * ox2) / (rx * rx) + (oy * oy) / (ry * ry) + (hash2(gx + ox2 * 3, gy + oy * 5) - 0.5) * 0.55;
          if (q >= 1) continue;
          var k = LEAF;
          if (oy > 0 && q > 0.45) k = STEM;
          else if (oy <= 0 && ox2 <= 0 && q < 0.7) k = hash2(gx * 2 + ox2, gy * 2 + oy) > 0.6 ? LIT : TIP;
          else if (hash2(gx + ox2 * 7, gy + oy * 11) > 0.72) k = TIP;
          put(gx + ox2, gy + oy, k);
        }
        if (r > 0.978) {
          // in flower
          var bloom = BLOOMS[(r * 9973 | 0) % 4];
          for (s2 = 0; s2 < 3; s2++) {
            var fx = gx + Math.round((hash2(gx + s2 * 7, gy * 3) - 0.5) * rx * 1.6);
            var fy = gy - Math.round(hash2(gx * 3, gy + s2 * 5) * ry);
            var fi = fy * cols + fx;
            if (fx >= 0 && fy >= 0 && fx < cols && fy < rows && kind[fi] >= LEAF && kind[fi] <= LIT) kind[fi] = bloom;
          }
        }
      }

      // mushroom: a pale stalk and a three-cell cap on a ledge with headroom
      if (gy > 2 && edge[i] === 1 && !isRock(gx, gy - 1) && !isRock(gx - 1, gy - 2) && !isRock(gx, gy - 2) &&
          !isRock(gx + 1, gy - 2) && (r = hash2(gx * 41 + 13, gy * 43 + 5)) > 0.972) {
        var capc = r > 0.986 ? AMBER : PINK;
        if (!kind[i - cols] && !kind[i - 2 * cols]) {
          put(gx, gy - 1, WHITE); put(gx - 1, gy - 2, capc); put(gx, gy - 2, capc); put(gx + 1, gy - 2, capc);
        }
      }

      // ore: a small vein set into solid rock, well back from the faces. The
      // deep dark rock runs to diamond, amethyst and gold; the paler rock nearer
      // the cave to copper and quartz.
      if (!edge[i] && tier[i] >= 2 && (r = hash2(gx * 37 + 11, gy * 47 + 3)) > ORE_RARITY) {
        var pick = hash2(gx * 53 + 5, gy * 59 + 17);
        var ore = tier[i] >= top - 1
          ? (pick < 0.32 ? DIAMOND : pick < 0.56 ? AMETHYST : pick < 0.8 ? GOLD : pick < 0.9 ? COPPER : QUARTZ)
          : (pick < 0.42 ? COPPER : pick < 0.72 ? QUARTZ : pick < 0.9 ? GOLD : AMETHYST);
        var lean = hash2(gx * 5 + 7, gy * 9 + 2) > 0.5 ? 1 : -1;
        var size = (r - ORE_RARITY) / (1 - ORE_RARITY);   // 0..1
        var vein = [[0, 0, 0], [0, -1, 1], [lean, 0, 0]];   // [dx, dy, glint?]
        if (size > 0.3) vein.push([lean, -1, 0]);
        if (size > 0.55) vein.push([-lean, 1, 0], [lean * 2, 0, 0]);
        if (size > 0.8) vein.push([lean * 2, -1, 1], [-lean, 0, 0], [0, 1, 0]);
        for (s2 = 0; s2 < vein.length; s2++) {
          x = gx + vein[s2][0]; y = gy + vein[s2][1];
          if (!isRock(x, y)) continue;
          var vk = kind[y * cols + x];
          if (!vk || vk === MOSS_D || vk === MOSS_L) kind[y * cols + x] = ORE + ore * 2 + vein[s2][2];
        }
      }
    }

    // compose. Anything standing proud of the rock drops a one-cell shadow on
    // the stone below it, which is most of what makes it read as in front.
    var color = new Uint8Array(n);
    for (i = 0; i < n; i++) {
      var kd = kind[i];
      if (kd === MOSS_D) color[i] = C_MOSS_D + tier[i];
      else if (kd === MOSS_L) color[i] = C_MOSS_L + tier[i];
      else if (kd) color[i] = C_FLAT + kd;
      else if (rock[i]) {
        var above = i >= cols ? kind[i - cols] : 0;
        color[i] = (above && above !== MOSS_D && above !== MOSS_L && above < ORE ? C_SHADE : C_ROCK) + tier[i];
      }
    }
    return { cols: cols, rows: rows, color: color };
  }

  if (IN_WORKER) {
    self.onmessage = function (e) {
      var m = e.data, g = generate(m.w, m.h, m.safe, m.opts);
      self.postMessage({ id: m.id, cols: g.cols, rows: g.rows, color: g.color }, [g.color.buffer]);
    };
    return;
  }

  // generate() off the main thread where possible. Only the latest request
  // ever calls back: a newer layout supersedes whatever is still in flight.
  var worker, wanted = 0, waiting = null;
  function generateAsync(w, h, safe, opts, done) {
    if (worker === undefined) {
      worker = null;
      try { if (SELF_URL && window.Worker) worker = new Worker(SELF_URL); } catch (e) { worker = null; }
      if (worker) {
        worker.onmessage = function (e) {
          if (!waiting || e.data.id !== wanted) return;
          var cb = waiting.done;
          waiting = null;
          cb(e.data);
        };
        worker.onerror = function () {
          // it could not start, or fell over: do the work here instead
          worker = null;
          if (!waiting) return;
          var job = waiting;
          waiting = null;
          job.done(generate(job.w, job.h, job.safe, job.opts));
        };
      }
    }
    if (!worker) { done(generate(w, h, safe, opts)); return; }
    waiting = { w: w, h: h, safe: safe, opts: opts, done: done };
    worker.postMessage({ id: ++wanted, w: w, h: h, safe: safe, opts: opts });
  }

  // One canvas pixel per cell, scaled up by CSS (image-rendering: pixelated):
  // crisp at any zoom, and even a very long article is a small bitmap that
  // costs nothing to scroll. `fill` paints the open cave too, for a canvas with
  // no backdrop of that colour behind it.
  function paint(canvas, g, fill) {
    canvas.width = g.cols; canvas.height = g.rows;
    canvas.style.width = g.cols * CELL + 'px'; canvas.style.height = g.rows * CELL + 'px';
    var ctx = canvas.getContext('2d');
    var img = ctx.createImageData(g.cols, g.rows), px = img.data;
    var air = fill ? rgb(fill) : null;
    for (var i = 0; i < g.color.length; i++) {
      var c = g.color[i] ? PAL[g.color[i]] : air;
      if (!c) continue;
      var o = i * 4;
      px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /* ---- landing page: a cavity the size of the viewport ---- */

  function Cave(canvas, plate) {
    this.canvasEl = canvas;
    this.plateEl = plate;
  }

  Cave.prototype.start = function () {
    var self = this;
    this.render();
    addEventListener('resize', function () { self.render(); });
  };

  Cave.prototype.render = function () {
    if (innerWidth <= PLAIN_HOME) return;   // the canvas is display:none down there
    var r = this.plateEl.getBoundingClientRect(), pad = 20;
    var safe = { x0: r.left - pad, y0: r.top - pad, x1: r.right + pad, y1: r.bottom + pad };
    paint(this.canvasEl, generate(innerWidth, innerHeight, safe), BACKDROP);
  };

  /* ---- post pages: a shaft the height of the document ---- */

  var CEILING = 90;      // px of rock-bearing margin above the article
  var FLOOR_AIR = 110;   // px of open cave between the footer's top edge and the floor

  function Shaft(canvas, page, column, floor) {
    this.canvasEl = canvas;
    this.page = page;
    this.column = column;
    this.floor = floor;
  }

  Shaft.prototype.start = function () {
    var self = this, timer = null;
    this.render();
    // the page grows as images, fonts and maths arrive: repaint once it settles
    var settle = function () {
      clearTimeout(timer);
      timer = setTimeout(function () { self.render(); }, 120);
    };
    if (window.ResizeObserver) new ResizeObserver(settle).observe(this.page);
    else { addEventListener('resize', settle); addEventListener('load', settle); }
  };

  Shaft.prototype.render = function () {
    if (innerWidth <= PLAIN_POST) return;   // the canvas is display:none down there
    var w = this.page.clientWidth, h = this.page.offsetHeight;
    if (w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    var pr = this.page.getBoundingClientRect();
    var cr = this.column.getBoundingClientRect();
    var fr = this.floor.getBoundingClientRect();
    var canvas = this.canvasEl;
    generateAsync(w, h, {
      x0: cr.left - pr.left, x1: cr.right - pr.left,
      y0: CEILING, y1: fr.top - pr.top + FLOOR_AIR
    }, { maxMargin: 560 }, function (g) {
      paint(canvas, g);
      canvas.classList.add('is-painted');   // post.css fades it in
    });
  };

  /* ---- landing page wiring ---- */

  // The whole viewport scrolls, but only the content column moves — eased so a
  // wheel over the rock feels the same as one over the text.
  function initWheelScroll(scroller) {
    var reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    var target = null, easedTop = null, raf = null;

    function stop() {
      target = null; easedTop = null;
      if (raf) { cancelAnimationFrame(raf); raf = null; }
    }

    function tick() {
      raf = null;
      if (target == null) return;
      // if anything else moved the scroller since our last frame — native wheel,
      // touch, keyboard, a link jump — yield to it rather than yanking it back
      if (easedTop != null && Math.abs(scroller.scrollTop - easedTop) > 1.5) { stop(); return; }
      var from = scroller.scrollTop, d = target - from;
      if (Math.abs(d) >= 0.6) scroller.scrollTop = from + d * 0.18;
      // arrived, or the step was too small for the browser to register: land it
      if (Math.abs(d) < 0.6 || scroller.scrollTop === from) { scroller.scrollTop = target; stop(); return; }
      easedTop = scroller.scrollTop;
      raf = requestAnimationFrame(tick);
    }

    if (scroller) {
      addEventListener('wheel', function (e) {
        if (innerWidth <= PLAIN_HOME) return;   // the document scrolls itself down there
        if (e.target instanceof Node && scroller.contains(e.target)) {
          // wheeling over the text scrolls it natively. Drop any eased scroll still
          // in flight from a wheel over the rock, or it fights the native scroll and
          // hauls the reader back to a stale target.
          stop();
          return;
        }
        var px = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * scroller.clientHeight : e.deltaY;
        var max = scroller.scrollHeight - scroller.clientHeight;
        var from = target == null ? scroller.scrollTop : target;
        var to = Math.max(0, Math.min(max, from + px));
        if (reduced) { stop(); scroller.scrollTop = to; return; }
        target = to;
        if (!raf) raf = requestAnimationFrame(tick);
      }, { passive: true });
    }

    return { stop: stop };
  }

  // the ABOUT / POSTS tab switch
  function initTabs(wheel) {
    var tabs = [].slice.call(document.querySelectorAll('[data-tab]'));
    var panels = {
      home: document.getElementById('panel-home'),
      blog: document.getElementById('panel-blog')
    };
    if (!tabs.length) return;

    function go(page, push) {
      if (!panels[page]) return;
      Object.keys(panels).forEach(function (k) {
        if (panels[k]) panels[k].hidden = k !== page;
      });
      tabs.forEach(function (t) {
        var on = t.getAttribute('data-tab') === page;
        t.classList.toggle('is-active', on);
        t.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      var sc = document.getElementById('plate-scroll');
      if (sc) sc.scrollTop = 0;
      if (innerWidth <= PLAIN_HOME) scrollTo(0, 0);
      wheel.stop();
      if (push) history.replaceState(null, '', page === 'blog' ? '#posts' : location.pathname);
    }

    tabs.forEach(function (t) {
      t.addEventListener('click', function (e) {
        e.preventDefault();
        go(t.getAttribute('data-tab'), true);
      });
    });

    go(location.hash === '#posts' ? 'blog' : 'home', false);
    addEventListener('hashchange', function () {
      go(location.hash === '#posts' ? 'blog' : 'home', false);
    });
  }

  function init() {
    var canvas = document.getElementById('cave');
    var plate = document.getElementById('plate');
    if (canvas && plate) {
      new Cave(canvas, plate).start();
      initTabs(initWheelScroll(document.getElementById('plate-scroll')));
    }

    var shaft = document.getElementById('shaft');
    var page = document.getElementById('page');
    var column = document.getElementById('post');
    var floor = document.getElementById('floor');
    if (shaft && page && column && floor) new Shaft(shaft, page, column, floor).start();
  }

  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', init);
  else init();
})();
