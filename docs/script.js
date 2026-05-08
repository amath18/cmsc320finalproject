/* =========================================================================
   In-browser diabetes risk calculator + tasteful reveal animations.
   No frameworks. No analytics. No data leaves the page.
   ========================================================================= */

// ------- Reveal-on-scroll ---------------------------------------------
(function () {
  const targets = document.querySelectorAll(
    ".prose section, .hero__stats, .contrib-card, .figure, .pullquote, .calc, .metrics, .codeblock, .hypotheses"
  );
  targets.forEach((el) => el.classList.add("reveal"));

  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-visible");
            io.unobserve(e.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );
    targets.forEach((el) => io.observe(el));
  } else {
    targets.forEach((el) => el.classList.add("is-visible"));
  }
})();

// ------- Counter animation in the hero --------------------------------
(function () {
  const el = document.querySelector("[data-counter]");
  if (!el) return;
  const target = Number(el.dataset.counter);
  if (!Number.isFinite(target)) return;

  let started = false;
  const animate = () => {
    if (started) return;
    started = true;
    const start = performance.now();
    const dur = 1400;
    const ease = (t) => 1 - Math.pow(1 - t, 3);
    const tick = (now) => {
      const t = Math.min(1, (now - start) / dur);
      const v = Math.round(ease(t) * target);
      el.textContent = v.toLocaleString("en-US");
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          animate();
          io.disconnect();
        }
      },
      { threshold: 0.6 }
    );
    io.observe(el);
  } else {
    animate();
  }
})();

// ------- Risk calculator ----------------------------------------------
(async function () {
  const root = document.getElementById("calc");
  if (!root) return;

  let model;
  try {
    const res = await fetch("model.json", { cache: "no-cache" });
    if (!res.ok) throw new Error("model.json HTTP " + res.status);
    model = await res.json();
  } catch (err) {
    console.warn("Risk calculator: could not load model.json", err);
    const label = root.querySelector("[data-band]");
    if (label) {
      label.textContent = "model unavailable";
      label.dataset.bandState = "high";
    }
    return;
  }

  const { features, coefficients, intercept, scaler_mean, scaler_scale } = model;

  // Default values for any feature whose input we don't render
  const DEFAULTS = Object.fromEntries(features.map((f, i) => [f, scaler_mean[i]]));

  // Hook up live output text for sliders
  document.querySelectorAll('input[type="range"][data-feature]').forEach((r) => {
    const out =
      document.querySelector(`[data-bmi-out]`) && r.dataset.feature === "BMI"
        ? document.querySelector("[data-bmi-out]")
        : document.querySelector(`[data-out="${r.dataset.feature}"]`);
    if (out) {
      const update = () => (out.textContent = r.value);
      r.addEventListener("input", update);
      update();
    }
  });

  // Read the form into a feature vector ordered the same way the model expects
  function readVector() {
    const v = new Array(features.length).fill(0);
    features.forEach((name, i) => {
      const el = root.querySelector(`[data-feature="${name}"]`);
      if (!el) {
        v[i] = DEFAULTS[name];
        return;
      }
      if (el.type === "checkbox") {
        v[i] = el.checked ? 1 : 0;
      } else {
        const num = Number(el.value);
        v[i] = Number.isFinite(num) ? num : DEFAULTS[name];
      }
    });
    return v;
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function predict(v) {
    let z = intercept;
    for (let i = 0; i < v.length; i++) {
      const std = (v[i] - scaler_mean[i]) / scaler_scale[i];
      z += coefficients[i] * std;
    }
    return sigmoid(z);
  }

  // Render the gauge: arc length, needle rotation, percent, band color
  const arc = document.getElementById("gaugeArc");
  const needle = document.getElementById("gaugeNeedle");
  const pctEl = root.querySelector("[data-pct]");
  const bandEl = root.querySelector("[data-band]");
  const ARC_LEN = 251.3; // 2*pi*r * 0.5 for the half-circle stroke (computed in CSS)

  function band(p) {
    if (p < 0.15) return ["low", "below average"];
    if (p < 0.4) return ["mid", "elevated"];
    return ["high", "high — consider screening"];
  }

  function render() {
    const p = predict(readVector());
    const pct = Math.round(p * 100);
    pctEl.textContent = pct.toString();

    arc.setAttribute("stroke-dashoffset", String(ARC_LEN * (1 - p)));

    // needle rotates from -90 (left, p=0) through 0 (top, p=0.5) to +90 (right, p=1)
    const angle = -90 + 180 * p;
    needle.setAttribute("transform", `rotate(${angle.toFixed(2)} 100 110)`);

    const [state, label] = band(p);
    bandEl.dataset.bandState = state;
    bandEl.textContent = label;
  }

  // Listen on every input the user can change
  root.querySelectorAll("input, select").forEach((el) => {
    el.addEventListener("input", render);
    el.addEventListener("change", render);
  });

  // First paint
  render();
})();
