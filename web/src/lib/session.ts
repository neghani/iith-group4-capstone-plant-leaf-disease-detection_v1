import type { CheckSession } from "./types";

const CURRENT_KEY = "cropcare_current_check";
const HISTORY_KEY = "cropcare_check_history";
const MAX_HISTORY = 5;

export function saveCheckSession(session: CheckSession): void {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(CURRENT_KEY, JSON.stringify(session));

  const raw = sessionStorage.getItem(HISTORY_KEY);
  const history: CheckSession[] = raw ? JSON.parse(raw) : [];
  const filtered = history.filter((h) => h.id !== session.id);
  const next = [session, ...filtered].slice(0, MAX_HISTORY);
  sessionStorage.setItem(HISTORY_KEY, JSON.stringify(next));
}

export function getCurrentCheck(): CheckSession | null {
  if (typeof window === "undefined") return null;
  const raw = sessionStorage.getItem(CURRENT_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as CheckSession;
  } catch {
    return null;
  }
}

export function getCheckHistory(): CheckSession[] {
  if (typeof window === "undefined") return [];
  const raw = sessionStorage.getItem(HISTORY_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as CheckSession[];
  } catch {
    return [];
  }
}

export function loadCheckById(id: string): CheckSession | null {
  if (id === getCurrentCheck()?.id) return getCurrentCheck();
  return getCheckHistory().find((h) => h.id === id) ?? null;
}

export function setCurrentFromHistory(session: CheckSession): void {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(CURRENT_KEY, JSON.stringify(session));
}
