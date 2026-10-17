import en from "./en.json";
import te from "./te.json";

export type Locale = "en" | "te";

const catalogs = { en, te } as const;

export function isLocale(value: string | null | undefined): value is Locale {
  return value === "en" || value === "te";
}

function lookup(catalog: unknown, key: string): string | undefined {
  let current: unknown = catalog;
  for (const part of key.split(".")) {
    if (!current || typeof current !== "object" || !(part in current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[part];
  }
  return typeof current === "string" ? current : undefined;
}

export function translate(
  locale: Locale,
  key: string,
  vars?: Record<string, string | number>
): string {
  const raw = lookup(catalogs[locale], key) ?? lookup(catalogs.en, key) ?? key;
  if (!vars) return raw;
  return raw.replace(/\{(\w+)\}/g, (_, name: string) =>
    vars[name] == null ? "" : String(vars[name])
  );
}

export function cropName(locale: Locale, value: string): string {
  const table = catalogs[locale].crops as Record<string, string>;
  if (table[value]) return table[value];
  const match = Object.keys(table).find(
    (key) => key.toLowerCase() === value.toLowerCase()
  );
  if (match) return table[match];
  const english = catalogs.en.crops as Record<string, string>;
  return english[value] ?? value;
}

export function leafKeys(value: unknown, prefix = ""): string[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    if (child && typeof child === "object" && !Array.isArray(child)) {
      return leafKeys(child, path);
    }
    return [path];
  });
}

export { en, te };
