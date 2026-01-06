/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

export type AppView = 'landing' | 'workspace' | 'about' | 'future' | 'results' | 'nlp';

export type OutputMode = 'comic' | 'mindmap';

export type ProcessStatus = 'idle' | 'analyzing' | 'generating' | 'complete' | 'error';

export interface ComicPanel {
  id: string;
  prompt: string;
  caption: string;
  imageUrl?: string;
}

export interface MindMapNode {
  id: string;
  label: string;
  type: 'concept' | 'entity' | 'action' | 'main' | 'category' | 'detail';
  nodeType?: string;
  level?: number;
  x?: number;
  y?: number;
  size?: number;
}

export interface MindMapEdge {
  id: string;
  from: string;
  to: string;
  label: string;
  relation?: string;
}

export interface MindMapData {
  nodes: MindMapNode[];
  edges: MindMapEdge[];
}

export interface AnalysisResult {
  mode: OutputMode;
  title: string;
  summary: string;
  comicData?: ComicPanel[];
  mindMapData?: MindMapData;
}
