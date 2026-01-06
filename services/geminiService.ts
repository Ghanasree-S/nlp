/**
 * VisualVerse Service
 * Uses FastAPI Backend - NO RATE LIMITS!
 */

import { AnalysisResult } from "../types";

const BACKEND_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const analyzeText = async (
    text: string,
    requestedMode: "auto" | "comic" | "mindmap"
): Promise<AnalysisResult> => {
    const response = await fetch(`${BACKEND_URL}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, mode: requestedMode }),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Error: ${response.status}`);
    }

    const data = await response.json();

    return {
        mode: data.mode as "comic" | "mindmap",
        title: data.title || "Generated Content",
        summary: data.summary || "",
        comicData: data.mode === "comic"
            ? (data.comic_data || []).map((p: any, i: number) => ({
                id: String(i + 1),
                prompt: p.prompt || "",
                caption: p.caption || "",
            }))
            : undefined,
        mindMapData: data.mode === "mindmap"
            ? {
                nodes: (data.mindmap_data?.nodes || []).map((n: any) => ({
                    id: n.id,
                    label: n.label,
                    type: n.type || "concept",
                    nodeType: n.nodeType || n.type || "concept",
                    level: n.level || 2,
                    x: n.x || 600,
                    y: n.y || 350,
                    size: n.size || 45,
                })),
                edges: (data.mindmap_data?.edges || []).map((e: any, i: number) => ({
                    id: e.id || `e${i}`,
                    from: e.source || e.from,
                    to: e.target || e.to,
                    label: e.label || e.relation || "",
                    relation: e.relation || e.label || "RELATES_TO",
                })),
            }
            : undefined,
    };
};

export const generatePanelImage = async (prompt: string): Promise<string> => {
    const colors = ["#e94560", "#0f3460", "#533483", "#16213e"];
    const color = colors[Math.floor(Math.random() * colors.length)];
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">
    <rect width="512" height="512" fill="${color}"/>
    <rect x="8" y="8" width="496" height="496" fill="none" stroke="white" stroke-width="4" rx="8"/>
    <text x="256" y="200" fill="white" font-size="48" text-anchor="middle" font-weight="bold">PANEL</text>
    <text x="256" y="260" fill="white" font-size="16" text-anchor="middle" opacity="0.8">AI Generated Scene</text>
    <text x="256" y="300" fill="white" font-size="12" text-anchor="middle" opacity="0.6">${prompt.slice(0, 40)}...</text>
  </svg>`;
    // Use encodeURIComponent for safe encoding (works with all characters)
    return `data:image/svg+xml,${encodeURIComponent(svg)}`
};
