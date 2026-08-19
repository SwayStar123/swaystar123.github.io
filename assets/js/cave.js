/*
 * Cave background for the landing page.
 *
 * Ported from the Claude Design artboard `Redesign.dc.html` (project
 * "Interactive bouncing balls redesign"). The design's canvas-editor wrappers
 * — DCLogic/React refs, <sc-if>, <sc-for>, and the editable props panel — are
 * replaced with plain DOM here; the simulation itself is the design's own code.
 *
 * The artboard's editable props are frozen to the values it was published with.
 */
(function () {
  'use strict';

  var CONFIG = {
    wall: 'cave',        // frame | cave | scattered
    fillTarget: 0.75,    // wall density held by the regrow loop
    physics: 'floaty',   // floaty | snappy | heavy
    ballsOnLoad: 8
  };

  var PLAIN_MAX_WIDTH = 720;   // matches the layout's media query

  var CELL = 10, MAXHP = 18;
  var RAMP = ['#f7768e', '#e0af68', '#a9c1ff', '#7aa2f7'];
  // rock strata, cavity mouth -> screen edge: pale slate down to pure black
  var LAYERS = ['#9aa3b2', '#69707d', '#3d434e', '#22262e', '#101318', '#05070a', '#000000'];
  var FLORA = ['#4c6537', '#6f9448', '#9ece6a'];
  var BALL_COLORS = ['#7aa2f7', '#a9c1ff', '#e0af68', '#f7768e'];
  var PHYS = {
    floaty: { speed: 71, r: 3, range: 65, dps: 6 },
    snappy: { speed: 176, r: 2, range: 48, dps: 8.5 },
    heavy: { speed: 101, r: 4.5, range: 83, dps: 11 }
  };

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

  function Cave(canvas, plate, scroller) {
    this.canvasEl = canvas;
    this.plateEl = plate;
    this.scroller = scroller;
    this.balls = [];
    this.sparks = [];
    this.dead = [];
    this.broken = 0;
  }

  // phones drop the cave entirely and read as a plain scrolling page
  function isPlain() { return innerWidth <= PLAIN_MAX_WIDTH; }

  var MOTION_KEY = 'cave-motion';
  function motionPref() {
    try { return localStorage.getItem(MOTION_KEY); } catch (e) { return null; }
  }
  function saveMotionPref(on) {
    try { localStorage.setItem(MOTION_KEY, on ? 'on' : 'off'); } catch (e) {}
  }

  Cave.prototype.start = function () {
    var self = this;
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.plain = isPlain();
    // frozen unless the visitor has motion on: their stored choice wins, and
    // absent one, prefers-reduced-motion decides
    var pref = motionPref();
    this.static = pref === null ? this.reduced : pref === 'off';

    this.loop = function (t) {
      var dt = Math.min(0.05, (t - self.last) / 1000);
      self.last = t;
      self.step(dt);
      self.draw();
      self.raf = requestAnimationFrame(self.loop);
    };

    this.onResize = function () {
      var wasPlain = self.plain;
      self.plain = isPlain();
      if (self.plain) {
        // crossed down into phone layout: stop burning frames on a hidden canvas
        if (self.raf) { cancelAnimationFrame(self.raf); self.raf = null; }
        return;
      }
      self.measure();
      self.buildWall();
      if (self.static) self.draw();
      else if (wasPlain || !self.raf) { self.seed(); self.last = performance.now(); self.raf = requestAnimationFrame(self.loop); }
    };
    addEventListener('resize', this.onResize);

    if (this.plain) return;   // no canvas, no listeners, no loop

    this.measure();
    this.buildWall();
    if (this.static) {
      this.draw();
    } else {
      this.seed();
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.loop);
    }
    // only offer the click-to-spawn cursor while the sim is actually running
    this.canvasEl.style.cursor = this.static ? 'default' : 'crosshair';

    // click anywhere on the rock to add a ball
    this.onDown = function (e) {
      if (self.static) return;
      var r = self.canvasEl.getBoundingClientRect();
      self.spawnAt(e.clientX - r.left, e.clientY - r.top);
    };
    this.canvasEl.addEventListener('pointerdown', this.onDown);

    // the whole viewport scrolls, but only the content column moves — eased so it feels
    // the same as scrolling directly over the text
    this.scrollTarget = null;
    this.onWheel = function (e) {
      var sc = self.scroller;
      if (!sc) return;
      if (e.target instanceof Node && sc.contains(e.target)) {
        // wheeling over the text scrolls it natively. Drop any eased scroll still
        // in flight from a wheel over the rock, or it fights the native scroll and
        // hauls the reader back to a stale target.
        self.scrollTarget = null;
        return;
      }
      var px = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * sc.clientHeight : e.deltaY;
      var max = sc.scrollHeight - sc.clientHeight;
      var from = self.scrollTarget == null ? sc.scrollTop : self.scrollTarget;
      self.scrollTarget = Math.max(0, Math.min(max, from + px));
      if (self.static) sc.scrollTop = self.scrollTarget;
    };
    addEventListener('wheel', this.onWheel, { passive: true });
  };

  // Stopping heals the wall back to a clean cave rather than freezing it
  // mid-chip, so the still frame looks deliberate rather than half-eaten.
  Cave.prototype.setMotion = function (on) {
    if (this.plain) return;
    this.static = !on;
    if (on) {
      this.seed();
      this.last = performance.now();
      if (!this.raf) this.raf = requestAnimationFrame(this.loop);
    } else {
      if (this.raf) { cancelAnimationFrame(this.raf); this.raf = null; }
      this.balls.length = 0;
      this.sparks.length = 0;
      this.buildWall();
      this.draw();
    }
    this.canvasEl.style.cursor = on ? 'crosshair' : 'default';
  };

  Cave.prototype.seed = function () {
    if (this.balls.length) return;
    var n = Math.max(1, Math.min(24, CONFIG.ballsOnLoad));
    for (var i = 0; i < n; i++) this.addBall();
  };

  Cave.prototype.measure = function () {
    var c = this.canvasEl; if (!c) return;
    var dpr = Math.min(2, devicePixelRatio || 1);
    this.w = innerWidth; this.h = innerHeight;
    c.width = Math.round(this.w * dpr); c.height = Math.round(this.h * dpr);
    this.ctx = c.getContext('2d');
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var r = this.plateEl
      ? this.plateEl.getBoundingClientRect()
      : { left: this.w * 0.3, top: 60, right: this.w * 0.7, bottom: this.h - 50 };
    var pad = 20;
    this.safe = { x0: r.left - pad, y0: r.top - pad, x1: r.right + pad, y1: r.bottom + pad };
  };

  Cave.prototype.buildWall = function () {
    var geo = CONFIG.wall;
    this.cols = Math.ceil(this.w / CELL); this.rows = Math.ceil(this.h / CELL);
    this.grid = new Float32Array(this.cols * this.rows);
    this.mask = new Uint8Array(this.cols * this.rows);
    this.dead.length = 0; this.maskCount = 0; this.regAcc = 0;
    this.tier = new Int8Array(this.cols * this.rows);
    this.dArr = new Float32Array(this.cols * this.rows);
    this.plant = new Uint8Array(this.cols * this.rows);
    var maxD = 1, gx, gy;
    for (gy = 0; gy < this.rows; gy++) for (gx = 0; gx < this.cols; gx++) {
      var i = gy * this.cols + gx, cx = gx * CELL + CELL / 2, cy = gy * CELL + CELL / 2;
      if (cx > this.safe.x0 && cx < this.safe.x1 && cy > this.safe.y0 && cy < this.safe.y1) continue;
      var dx = Math.max(this.safe.x0 - cx, cx - this.safe.x1, 0);
      var dy = Math.max(this.safe.y0 - cy, cy - this.safe.y1, 0);
      var dSafe = Math.hypot(dx, dy);
      // how far out this cell sits as a FRACTION of the margin available on its own side,
      // so top/bottom (thin margins) get the same rock as left/right (wide margins)
      var nd = Math.min(1, Math.max(
        cx < this.safe.x0 ? (this.safe.x0 - cx) / Math.max(1, this.safe.x0) : 0,
        cx > this.safe.x1 ? (cx - this.safe.x1) / Math.max(1, this.w - this.safe.x1) : 0,
        cy < this.safe.y0 ? (this.safe.y0 - cy) / Math.max(1, this.safe.y0) : 0,
        cy > this.safe.y1 ? (cy - this.safe.y1) / Math.max(1, this.h - this.safe.y1) : 0
      ));
      var n = fbm(gx * 0.055, gy * 0.055);
      var fill;
      if (geo === 'frame') fill = true;
      else if (geo === 'scattered') fill = n > 0.52 - 0.22 * Math.min(1, dSafe / 220) ? hash2(gx, gy * 7) > 0.35 : false;
      else {
        // cavern: solid rock everywhere, with a jagged cavity carved around the content
        var carve = 0.1 + 0.62 * Math.pow(fbm(gx * 0.032, gy * 0.032), 1.7);
        fill = nd > carve;
      }
      if (!fill) continue;
      this.mask[i] = 1; this.maskCount++;
      this.grid[i] = MAXHP;   // untouched rock always reads as its stratum, never pre-damaged
      this.dArr[i] = geo === 'cave' ? nd : dSafe;
      if (this.dArr[i] > maxD) maxD = this.dArr[i];
    }
    // strata normalised to the margin actually available: pale rock at the cavity mouth,
    // pure black by the screen edge, whatever the window size
    var top = LAYERS.length - 1;
    for (gy = 0; gy < this.rows; gy++) for (gx = 0; gx < this.cols; gx++) {
      var k = gy * this.cols + gx;
      if (!this.mask[k]) continue;
      var j = (fbm(gx * 0.07, gy * 0.07) - 0.5) * 0.22;
      this.tier[k] = Math.max(0, Math.min(top, Math.round((this.dArr[k] / maxD + j) * top)));
    }
    this.live = this.maskCount;
    this.growFlora();
  };

  // Vines off overhangs, shrubs and grass on ledges, brackets off vertical
  // faces, moss on whatever is left — planted straight into the grid as fragile
  // blocks, so they break like anything else. Broken plant cells leave the mask
  // for good.
  Cave.prototype.growFlora = function () {
    var self = this;
    var plant = [];
    var empty = function (gx, gy) {
      if (gx < 0 || gy < 0 || gx >= self.cols || gy >= self.rows) return true;
      return self.grid[gy * self.cols + gx] <= 0;
    };
    // the tall rock columns either side of the content plate. Vines have room to
    // fall the full height here, so they run several times longer than the ones
    // hanging off the ceiling.
    var beside = function (gx, gy) {
      var x = gx * CELL + CELL / 2, y = gy * CELL + CELL / 2;
      return (x < self.safe.x0 || x > self.safe.x1) &&
             y > self.safe.y0 - 80 && y < self.safe.y1 + 80;
    };

    for (var gy = 0; gy < this.rows; gy++) for (var gx = 0; gx < this.cols; gx++) {
      var i = gy * this.cols + gx;
      if (this.grid[i] <= 0) continue;
      var seed = hash2(gx * 13 + 7, gy * 29 + 3);
      var side = beside(gx, gy);
      var px, s, n, h, dx, up, b, bx, bh, reach, blades;

      if (empty(gx, gy + 1) && (side || gy < this.rows * 0.55) && seed > 0.86) {
        // pixel vine: a stem of cells stepping down, leaf cells budding off it
        px = [];
        var ox = 0;
        n = 6 + ((seed * 100) % 10 | 0);
        if (side) n = Math.round(n * (2 + (seed * 37) % 1.4));
        for (s = 1; s <= n; s++) {
          if (hash2(gx + s, gy * 3) > 0.72) ox += hash2(gx, gy + s) > 0.5 ? 1 : -1;
          px.push({ x: ox, y: s, leaf: 0, s: s });
          if (s > 1 && hash2(gx * 5 + s, gy) > 0.66) px.push({ x: ox + (s % 2 ? 1 : -1), y: s, leaf: 1, s: s });
          if (s === n) px.push({ x: ox, y: s + 1, leaf: 2, s: s });
        }
        plant.push({ gx: gx, gy: gy, px: px });
      } else if (empty(gx, gy - 1) && seed > 0.76 && seed <= 0.86) {
        // pixel shrub: a squat tuft of cells rising off the ledge
        px = [{ x: 0, y: -1, leaf: 0, s: 1 }];
        h = 2 + ((seed * 60) % 3 | 0);
        for (s = 2; s <= h; s++) px.push({ x: 0, y: -s, leaf: 0, s: s });
        var dirs = [-1, 1];
        for (var di = 0; di < dirs.length; di++) {
          dx = dirs[di];
          if (hash2(gx + dx * 3, gy) < 0.3) continue;
          up = 1 + ((hash2(gx, gy + dx) * 3) | 0);
          px.push({ x: dx, y: -up, leaf: 1, s: up });
          if (hash2(gx * 7, gy + dx) > 0.5) px.push({ x: dx * 2, y: -up - 1, leaf: 2, s: up + 1 });
        }
        px.push({ x: 0, y: -h - 1, leaf: 2, s: h + 1 });
        plant.push({ gx: gx, gy: gy, px: px });
      } else if (empty(gx, gy - 1) && seed > 0.66 && seed <= 0.76) {
        // grass: a few loose blades of uneven height, no crown
        px = [];
        blades = 2 + ((seed * 90) % 3 | 0);
        for (b = 0; b < blades; b++) {
          bx = b - (blades >> 1);
          bh = 1 + ((hash2(gx + b * 5, gy * 3) * 4) | 0);
          for (s = 1; s <= bh; s++) px.push({ x: bx, y: -s, leaf: s === bh ? 2 : 1, s: s });
        }
        plant.push({ gx: gx, gy: gy, px: px });
      } else if ((empty(gx - 1, gy) || empty(gx + 1, gy)) && seed > 0.56 && seed <= 0.66) {
        // bracket fern off a vertical wall, reaching into the open side
        dx = empty(gx - 1, gy) ? -1 : 1;
        px = [];
        reach = 2 + ((seed * 70) % 3 | 0);
        for (s = 1; s <= reach; s++) {
          px.push({ x: dx * s, y: 0, leaf: 0, s: s });
          if (hash2(gx * 3 + s, gy * 7) > 0.45) px.push({ x: dx * s, y: -1, leaf: 1, s: s });
          if (s === reach) px.push({ x: dx * s, y: 1, leaf: 2, s: s });
        }
        plant.push({ gx: gx, gy: gy, px: px });
      } else if (hash2(gx * 11, gy * 17 + 5) > 0.72) {
        // moss: one speck on whichever face is exposed, so plain rock surfaces
        // pick up some texture instead of reading as bare blocks
        px = [];
        if (empty(gx, gy - 1)) px.push({ x: 0, y: -1, leaf: 0, s: 1 });
        else if (empty(gx - 1, gy)) px.push({ x: -1, y: 0, leaf: 0, s: 1 });
        else if (empty(gx + 1, gy)) px.push({ x: 1, y: 0, leaf: 0, s: 1 });
        else if (empty(gx, gy + 1)) px.push({ x: 0, y: 1, leaf: 0, s: 1 });
        if (px.length) plant.push({ gx: gx, gy: gy, px: px });
      }
    }
    if (plant.length > 2000) plant.length = 2000;
    for (var f = 0; f < plant.length; f++) {
      var fl = plant[f];
      for (var p = 0; p < fl.px.length; p++) {
        var cell = fl.px[p];
        var cx = fl.gx + cell.x, cy = fl.gy + cell.y;
        if (cx < 0 || cy < 0 || cx >= this.cols || cy >= this.rows) continue;
        var jj = cy * this.cols + cx;
        if (this.grid[jj] > 0) continue;
        var c = cx * CELL + CELL / 2, r = cy * CELL + CELL / 2;
        if (c > this.safe.x0 && c < this.safe.x1 && r > this.safe.y0 && r < this.safe.y1) continue;
        this.plant[jj] = cell.leaf + 1;
        this.grid[jj] = MAXHP * 0.16;   // fragile
        if (!this.mask[jj]) { this.mask[jj] = 1; this.maskCount++; this.live++; }
      }
    }
  };

  Cave.prototype.addBall = function (x, y) {
    var p = PHYS[CONFIG.physics] || PHYS.floaty;
    if (x == null) {
      var tries = 0;
      do { x = Math.random() * this.w; y = Math.random() * this.h; tries++; }
      while (tries < 80 && (this.solid(x, y) || (x > this.safe.x0 - 30 && x < this.safe.x1 + 30 && y > this.safe.y0 - 30 && y < this.safe.y1 + 30)));
    }
    var a = Math.random() * Math.PI * 2, sp = p.speed * (0.8 + Math.random() * 0.4);
    this.balls.push({
      x: x, y: y,
      vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
      r: p.r, c: BALL_COLORS[this.balls.length % BALL_COLORS.length], fire: 0
    });
  };

  Cave.prototype.spawnAt = function (x, y) {
    if (this.balls.length >= 48) this.balls.shift();
    // never start buried: walk out to the nearest open cell
    if (this.solid(x, y)) {
      var gx = Math.floor(x / CELL), gy = Math.floor(y / CELL);
      var bd = 1e9, bx = null, by = null;
      for (var r = 1; r <= 12 && bx === null; r++) {
        for (var oy = -r; oy <= r; oy++) for (var ox = -r; ox <= r; ox++) {
          if (Math.max(Math.abs(ox), Math.abs(oy)) !== r) continue;
          var cx = (gx + ox) * CELL + CELL / 2, cy = (gy + oy) * CELL + CELL / 2;
          if (cx < 0 || cy < 0 || cx > this.w || cy > this.h) continue;
          if (this.solid(cx, cy)) continue;
          var d = (cx - x) * (cx - x) + (cy - y) * (cy - y);
          if (d < bd) { bd = d; bx = cx; by = cy; }
        }
      }
      if (bx !== null) { x = bx; y = by; }
    }
    this.addBall(x, y);
  };

  Cave.prototype.easeScroll = function () {
    if (this.scrollTarget == null) return;
    var sc = this.scroller; if (!sc) return;
    // if anything else moved the scroller since our last frame — native wheel,
    // touch, keyboard, a link jump — yield to it rather than yanking it back
    if (this.easedTop != null && Math.abs(sc.scrollTop - this.easedTop) > 1.5) {
      this.scrollTarget = null; this.easedTop = null; return;
    }
    var d = this.scrollTarget - sc.scrollTop;
    if (Math.abs(d) < 0.6) {
      sc.scrollTop = this.scrollTarget; this.scrollTarget = null; this.easedTop = null; return;
    }
    sc.scrollTop += d * 0.18;
    this.easedTop = sc.scrollTop;
  };

  Cave.prototype.step = function (dt) {
    this.easeScroll();
    var p = PHYS[CONFIG.physics] || PHYS.floaty;
    for (var bi = 0; bi < this.balls.length; bi++) {
      var b = this.balls[bi];
      // sweep with axis-separated collision against the rock
      var nx = b.x + b.vx * dt, ny = b.y + b.vy * dt;
      if (this.solid(b.x, b.y)) { b.x = nx; b.y = ny; }  // buried (just spawned) — fly free
      else {
        var hx = this.solid(nx + Math.sign(b.vx) * b.r, b.y);
        var hy = this.solid(b.x, ny + Math.sign(b.vy) * b.r);
        if (hx) { this.chip(nx + Math.sign(b.vx) * b.r, b.y); b.vx = -b.vx; nx = b.x; }
        if (hy) { this.chip(b.x, ny + Math.sign(b.vy) * b.r); b.vy = -b.vy; ny = b.y; }
        b.x = nx; b.y = ny;
      }
      if (b.x < b.r) { b.x = b.r; b.vx = Math.abs(b.vx); }
      if (b.x > this.w - b.r) { b.x = this.w - b.r; b.vx = -Math.abs(b.vx); }
      if (b.y < b.r) { b.y = b.r; b.vy = Math.abs(b.vy); }
      if (b.y > this.h - b.r) { b.y = this.h - b.r; b.vy = -Math.abs(b.vy); }
      b.target = this.nearest(b.x, b.y, p.range);
      if (b.target >= 0) this.damage(b.target, p.dps * dt);
    }
    for (var i = this.sparks.length - 1; i >= 0; i--) {
      var sp = this.sparks[i];
      sp.x += sp.vx * dt; sp.y += sp.vy * dt; sp.life -= dt * 2.2;
      if (sp.life <= 0) this.sparks.splice(i, 1);
    }
    if (this.sparks.length > 220) this.sparks.splice(0, this.sparks.length - 220);
    // closed-loop regrow: hold the wall near a target fill fraction whatever the ball count
    var target = Math.max(0.2, Math.min(1, CONFIG.fillTarget)) * this.maskCount;
    var deficit = target - this.live;
    var rate = deficit > 0 ? 0.5 + deficit * 0.22 : 0;
    this.regAcc = Math.min(600, this.regAcc + rate * dt);
    var budget = 400;
    while (this.regAcc >= 1 && budget-- > 0) { this.regAcc -= 1; this.regrowOne(); }
  };

  Cave.prototype.solid = function (x, y) {
    var gx = Math.floor(x / CELL), gy = Math.floor(y / CELL);
    if (gx < 0 || gy < 0 || gx >= this.cols || gy >= this.rows) return false;
    return this.grid[gy * this.cols + gx] > 0;
  };

  Cave.prototype.chip = function (x, y) {
    var gx = Math.floor(x / CELL), gy = Math.floor(y / CELL);
    if (gx < 0 || gy < 0 || gx >= this.cols || gy >= this.rows) return;
    this.damage(gy * this.cols + gx, 1.6);
  };

  Cave.prototype.damage = function (i, amount) {
    if (this.grid[i] <= 0) return;
    this.grid[i] -= amount;
    if (this.grid[i] > 0) return;
    this.grid[i] = 0; this.broken++; this.live--;
    if (this.plant[i]) { this.plant[i] = 0; this.mask[i] = 0; this.maskCount--; }  // flora never comes back
    else this.dead.push(i);
    var gx = i % this.cols, gy = (i - gx) / this.cols;
    for (var k = 0; k < 4; k++) {
      this.sparks.push({
        x: gx * CELL + CELL / 2, y: gy * CELL + CELL / 2,
        vx: (Math.random() - 0.5) * 90, vy: (Math.random() - 0.5) * 90,
        life: 1, c: RAMP[0]
      });
    }
  };

  // rock creeps back only into empty cells that touch surviving rock, outermost first —
  // so the cavity keeps its carved shape and never refills from the inside
  Cave.prototype.regrowOne = function () {
    var best = -1, bestD = -1;
    for (var t = 0; t < 18 && this.dead.length; t++) {
      var k = (Math.random() * this.dead.length) | 0;
      var i = this.dead[k];
      if (this.grid[i] > 0 || !this.mask[i]) {
        this.dead[k] = this.dead[this.dead.length - 1]; this.dead.pop(); continue;
      }
      if (this.neighbour(i) < 0) continue;
      if (this.dArr[i] > bestD) { best = k; bestD = this.dArr[i]; }
    }
    if (best >= 0) this.fill(best);
  };

  Cave.prototype.neighbour = function (i) {
    var gx = i % this.cols, gy = (i - gx) / this.cols;
    var opts = [];
    var offs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    for (var o = 0; o < offs.length; o++) {
      var cx = gx + offs[o][0], cy = gy + offs[o][1];
      if (cx < 0 || cy < 0 || cx >= this.cols || cy >= this.rows) continue;
      var j = cy * this.cols + cx;
      if (this.grid[j] > 0) opts.push(j);
    }
    return opts.length ? opts[(Math.random() * opts.length) | 0] : -1;
  };

  // new rock inherits the health and stratum of the neighbour it grew from
  Cave.prototype.fill = function (k) {
    var i = this.dead[k], j = this.neighbour(i);
    if (j >= 0) { this.grid[i] = Math.max(this.grid[j], MAXHP * 0.7); this.tier[i] = this.tier[j]; }
    else { this.grid[i] = MAXHP * 0.7; }
    this.live++;
    this.dead[k] = this.dead[this.dead.length - 1]; this.dead.pop();
  };

  Cave.prototype.nearest = function (x, y, range) {
    var gx = Math.floor(x / CELL), gy = Math.floor(y / CELL), R = Math.ceil(range / CELL);
    var best = -1, bd = range * range, bestPlant = -1, bdPlant = range * range;
    for (var oy = -R; oy <= R; oy++) for (var ox = -R; ox <= R; ox++) {
      var cx = gx + ox, cy = gy + oy;
      if (cx < 0 || cy < 0 || cx >= this.cols || cy >= this.rows) continue;
      var i = cy * this.cols + cx;
      if (this.grid[i] <= 0) continue;
      var px = cx * CELL + CELL / 2 - x, py = cy * CELL + CELL / 2 - y, d = px * px + py * py;
      if (this.plant[i]) { if (d < bdPlant) { bdPlant = d; bestPlant = i; } }   // shot only as a last resort
      else if (d < bd) { bd = d; best = i; }
    }
    return best >= 0 ? best : bestPlant;
  };

  Cave.prototype.draw = function () {
    var ctx = this.ctx; if (!ctx) return;
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.fillStyle = '#12141a'; ctx.fillRect(0, 0, this.w, this.h);
    var gap = 0;
    for (var i = 0; i < this.grid.length; i++) {
      var hp = this.grid[i]; if (hp <= 0) continue;
      var gx = i % this.cols, gy = (i - gx) / this.cols;
      var f = hp / MAXHP;
      var pl = this.plant[i];
      if (pl) { ctx.fillStyle = FLORA[pl - 1]; ctx.globalAlpha = f > 0.2 ? 1 : 0.65; }
      else if (f > 0.62) { ctx.fillStyle = LAYERS[this.tier[i]]; ctx.globalAlpha = 1; }
      else if (f > 0.3) { ctx.fillStyle = RAMP[1]; ctx.globalAlpha = 0.8; }
      else { ctx.fillStyle = RAMP[0]; ctx.globalAlpha = 1; }
      ctx.fillRect(gx * CELL, gy * CELL, CELL - gap, CELL - gap);
    }
    ctx.globalAlpha = 1;
    if (this.static) return;
    ctx.lineWidth = 1.2; ctx.lineCap = 'round';
    var b, k;
    for (k = 0; k < this.balls.length; k++) {
      b = this.balls[k];
      if (b.target >= 0) {
        var tgx = b.target % this.cols, tgy = (b.target - tgx) / this.cols;
        var tx = tgx * CELL + CELL / 2, ty = tgy * CELL + CELL / 2;
        ctx.strokeStyle = 'rgba(224,175,104,.95)'; ctx.shadowColor = '#e0af68'; ctx.shadowBlur = 7;
        ctx.beginPath(); ctx.moveTo(b.x, b.y);
        var segs = 4, nx = -(ty - b.y), ny = tx - b.x, len = Math.hypot(nx, ny) || 1;
        for (var s = 1; s <= segs; s++) {
          var t = s / segs, j = s === segs ? 0 : (Math.random() - 0.5) * 9;
          ctx.lineTo(b.x + (tx - b.x) * t + (nx / len) * j, b.y + (ty - b.y) * t + (ny / len) * j);
        }
        ctx.stroke(); ctx.shadowBlur = 0;
      }
    }
    for (var si = 0; si < this.sparks.length; si++) {
      var sp = this.sparks[si];
      ctx.globalAlpha = Math.max(0, sp.life); ctx.fillStyle = sp.c;
      ctx.fillRect(sp.x - 1.5, sp.y - 1.5, 3, 3);
    }
    ctx.globalAlpha = 1;
    for (k = 0; k < this.balls.length; k++) {
      b = this.balls[k];
      ctx.fillStyle = b.c; ctx.shadowColor = b.c; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, 6.2832); ctx.fill();
      ctx.shadowBlur = 0;
    }
  };

  /* ---- page wiring: the tab switch the design drives with <sc-if> ---- */

  function initTabs(cave) {
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
      if (innerWidth <= 720) scrollTo(0, 0);
      if (cave) { cave.scrollTarget = null; cave.easedTop = null; }
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

  function initMotionToggle(cave) {
    var btn = document.getElementById('motion-toggle');
    if (!btn || !cave || cave.plain) return;   // nothing to control on a plain page
    var on = !cave.static;
    btn.hidden = false;

    function paint() {
      btn.textContent = on ? 'PAUSE' : 'RESUME';
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
      btn.setAttribute('aria-label',
        on ? 'Pause the background animation' : 'Resume the background animation');
    }
    paint();

    btn.addEventListener('click', function () {
      on = !on;
      saveMotionPref(on);
      cave.setMotion(on);
      paint();
    });
  }

  function init() {
    var canvas = document.getElementById('cave');
    var plate = document.getElementById('plate');
    var scroller = document.getElementById('plate-scroll');
    var cave = null;
    if (canvas && plate) {
      cave = new Cave(canvas, plate, scroller);
      cave.start();
    }
    initTabs(cave);
    initMotionToggle(cave);
  }

  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', init);
  else init();
})();
