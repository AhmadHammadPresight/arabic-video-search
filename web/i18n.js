// One language at a time: Arabic (right to left, the default) or English (left to right).
// Every string the page shows comes from the table of the current language, web/locales/<code>.json.
//
// A table maps a key to a string with {name} placeholders, or, for a count, to its plural forms
// (zero, one, two, few, many, other, as Intl.PluralRules names them; "other" is required).
// A key such as "colour.red@f" names a grammatical variant: a language that needs it adds it,
// and every other language falls back to "colour.red".

export const LANGUAGES = { ar: "rtl", en: "ltr" };
const DEFAULT_LANGUAGE = "ar";
const STORAGE_KEY = "avs.lang";

let language = DEFAULT_LANGUAGE;
let table = {};
let plurals = new Intl.PluralRules(DEFAULT_LANGUAGE);

function saved() {
  const asked = new URLSearchParams(location.search).get("lang");
  if (asked in LANGUAGES) return asked;
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

// Load the table of the remembered language and set the page's language and direction.
export async function initLanguage() {
  const remembered = saved();
  language = remembered in LANGUAGES ? remembered : DEFAULT_LANGUAGE;
  const response = await fetch(`/locales/${language}.json`);
  table = await response.json();
  plurals = new Intl.PluralRules(language);
  document.documentElement.lang = language;
  document.documentElement.dir = LANGUAGES[language];
  return language;
}

export function currentLanguage() {
  return language;
}

// Remember the choice for this browser and load the page again in that language.
export function switchLanguage(code) {
  const url = new URL(location.href);
  try {
    localStorage.setItem(STORAGE_KEY, code);
    url.searchParams.delete("lang");
  } catch {
    url.searchParams.set("lang", code); // storage is blocked: carry the choice in the address instead
  }
  // Assigning the same address (or one that differs only after #) would not load the page again.
  if (url.href === location.href) location.reload();
  else location.assign(url);
}

function lookup(key) {
  if (Object.hasOwn(table, key)) return table[key];
  const base = key.split("@")[0];
  return base !== key && Object.hasOwn(table, base) ? table[base] : undefined;
}

function format(entry, args) {
  if (typeof entry === "object") entry = entry[plurals.select(Number(args.n))] ?? entry.other;
  return entry.replace(/\{(\w+)\}/g, (_, name) => String(args[name] ?? ""));
}

// The string for a key, with {name} placeholders filled from args (and args.n choosing the plural form).
export function t(key, args = {}) {
  const entry = lookup(key);
  if (entry === undefined) {
    console.warn(`no "${key}" in the ${language} table`);
    return key;
  }
  return format(entry, args);
}

// The string for a key when the table has one, else the fallback (a value the server sent as it is).
export function maybe(key, fallback, args = {}) {
  const entry = lookup(key);
  return entry === undefined ? fallback : format(entry, args);
}

const ATTRIBUTES = ["placeholder", "aria-label", "title", "alt"];

// Fill the elements marked in the HTML: data-i18n sets the text, data-i18n-<attribute> an attribute.
export function translatePage(root = document) {
  const all = (selector) => [...(root.matches?.(selector) ? [root] : []), ...root.querySelectorAll(selector)];
  for (const node of all("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const attribute of ATTRIBUTES) {
    for (const node of all(`[data-i18n-${attribute}]`)) node.setAttribute(attribute, t(node.getAttribute(`data-i18n-${attribute}`)));
  }
}
