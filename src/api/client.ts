function configuredApiBaseUrl(): string {
  try {
    const storageValue = typeof window !== "undefined" && typeof window.localStorage?.getItem === "function"
      ? window.localStorage.getItem("cephalon.apiBaseUrl")
      : null;
    return storageValue || import.meta.env.VITE_CEPHALON_API_URL || "http://127.0.0.1:8765";
  } catch {
    return import.meta.env.VITE_CEPHALON_API_URL || "http://127.0.0.1:8765";
  }
}

const API_BASE_URL = configuredApiBaseUrl();

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data.detail || data.error || res.statusText;
  } catch {
    return res.statusText;
  }
}

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    throw new ApiError(await parseError(res), res.status);
  }

  return res.json() as Promise<T>;
}

export async function responseError(res: Response): Promise<ApiError> {
  return new ApiError(await parseError(res), res.status);
}
