import { CaseRecord, DetectionDemoResult, DetectionUploadResult } from '../types';

// Proxied to backend/main.py (FastAPI) by vite.config.ts's server.proxy in
// dev; in a production build, serve this app behind the same origin as the
// backend (or set VITE_API_BASE) so these paths still resolve.
const BASE = (import.meta as any).env?.VITE_API_BASE ?? '';

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function fetchCases(): Promise<CaseRecord[]> {
  return getJson<CaseRecord[]>('/api/cases');
}

export function fetchCase(id: string): Promise<CaseRecord> {
  return getJson<CaseRecord>(`/api/cases/${id}`);
}

export function fetchDetectionDemo(): Promise<DetectionDemoResult> {
  return getJson<DetectionDemoResult>('/api/detection/demo');
}

export async function uploadForDetection(file: File): Promise<DetectionUploadResult> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/api/detect`, { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Detection failed (HTTP ${res.status}): ${text}`);
  }
  return res.json() as Promise<DetectionUploadResult>;
}
