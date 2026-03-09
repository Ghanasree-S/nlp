/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

export type AppView = 'landing' | 'workspace' | 'about' | 'future' | 'results' | 'nlp';

export type OutputMode = 'comic' | 'mindmap' | 'story';

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

export interface ClassificationData {
  text_type: string;
  confidence: number;
  features?: Record<string, any>;
  language?: string;
}

export interface StoryPanel {
  id: string;
  panel_number: number;
  caption: string;
  prompt: string;
  image_url?: string;
}

export interface StoryData {
  title: string;
  summary: string;
  story: string;
  panels: StoryPanel[];
  keywords_used: string[];
  nlp_info: {
    technique: string;
    corpus: string;
    keyword_roles: Record<string, string>;
  };
}

export interface AnalysisResult {
  mode: OutputMode;
  title: string;
  summary: string;
  language?: string;
  classification?: ClassificationData;
  comicData?: ComicPanel[];
  mindMapData?: MindMapData;
  storyData?: StoryData;
}
