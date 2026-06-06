import { ApiError } from "./api";
import type { CatalogQuery, LoadedModel, Theme } from "./types";

const TOKEN_KEY = "pymss.webui.token";
const THEME_KEY = "pymss.webui.theme";

export function createCatalogQuery(): CatalogQuery {
  return {
    supported: "true",
    local: "all",
    category: "",
    q: "",
  };
}

export function loadStoredToken(): string {
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

export function saveStoredToken(token: string): void {
  if (token) {
    sessionStorage.setItem(TOKEN_KEY, token);
  } else {
    sessionStorage.removeItem(TOKEN_KEY);
  }
}

export function clearStoredToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

export function loadStoredTheme(): Theme {
  return localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light";
}

export function saveStoredTheme(theme: Theme): void {
  localStorage.setItem(THEME_KEY, theme);
}

export function appErrorText(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.code}: ${error.message}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unknown error";
}

export function requiresTokenPrompt(error: unknown): boolean {
  return error instanceof ApiError && error.code === "invalid_api_key";
}

export function selectedStemsForModel(current: string[], model: LoadedModel | null): string[] {
  const instruments = model?.pymss.instruments ?? [];
  if (!instruments.length) {
    return [];
  }
  const filtered = current.filter((stem) => instruments.includes(stem));
  return filtered.length ? filtered : [...instruments];
}

export function toggleStemSelection(selectedStems: string[], stem: string, checked: boolean): string[] {
  return checked ? [...new Set([...selectedStems, stem])] : selectedStems.filter((item) => item !== stem);
}
