// src/lib/stream.ts
export type Frame = {
  gen: number;
  summary: {
    best_fit: number; mean_fit: number;
    best_on: number; best_off: number;
    best_gc: number; diversity_hamming: number;
    mutation_rate: number; crossover_rate: number;
  };
  top?: Array<{id:string;seq:string;fit:number;on:number;off:number;gc:number;}>;
  pareto?: Array<{id:string;on:number;off:number;}>;
  logo_counts?: Record<string, {A:number;C:number;G:number;T:number}>;
  done: boolean;
  best_sequence?: string;
  validation_fitness?: number;  // Full dataset fitness for final evaluation
  training_fitness?: number;    // Subset fitness consistent with training
};

export function openGAStreamURL(url: string, onFrame: (f: Frame) => void) {
  const es = new EventSource(url);
  es.addEventListener('frame', (ev: MessageEvent) => {
    try { onFrame(JSON.parse(ev.data)); } catch {}
  });
  es.onmessage = (ev) => {
    try { onFrame(JSON.parse(ev.data)); } catch {}
  };
  return es;
}

export function buildStreamURL(base = 'http://localhost:8000/runs/stream', params: Record<string, any>) {
  const q = new URLSearchParams();
  for (const [k,v] of Object.entries(params)) if (v !== undefined && v !== null) q.set(k, String(v));
  return `${base}?${q.toString()}`;
}
