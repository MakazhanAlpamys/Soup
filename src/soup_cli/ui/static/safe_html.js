/* Soup Web UI - escaping renderer. Every value interpolated into markup goes
   through escapeHtml unless it is already SafeHtml built by html`...`. */
'use strict';

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

class SafeHtml {
  constructor(value) {
    this.value = String(value);
  }
  toString() {
    return this.value;
  }
}

function trustedHtml(markup) {
  return new SafeHtml(markup);
}

function _renderValue(value) {
  if (value instanceof SafeHtml) return value.value;
  if (Array.isArray(value)) return value.map(_renderValue).join('');
  if (value === null || value === undefined || value === false) return '';
  return escapeHtml(value);
}

function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i += 1) {
    out += _renderValue(values[i]) + strings[i + 1];
  }
  return new SafeHtml(out);
}

function setHtml(element, content) {
  element.innerHTML = content instanceof SafeHtml ? content.value : escapeHtml(content);
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { escapeHtml, SafeHtml, trustedHtml, html, setHtml };
}
