"use strict";

const form = document.getElementById("login-form");
const error = document.getElementById("error");
const submit = document.getElementById("submit");

fetch("/api/health").then((r) => r.json()).then((health) => {
  if (health.authenticated) window.location.href = "/";
}).catch(() => {});

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  error.textContent = "";
  submit.disabled = true;
  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: document.getElementById("password").value }),
    });
    const body = await res.json().catch(() => ({}));
    if (res.ok) { window.location.href = "/"; return; }
    error.textContent = body.error || `Sign in failed (${res.status})`;
  } catch (err) {
    error.textContent = String(err);
  } finally {
    submit.disabled = false;
  }
});
