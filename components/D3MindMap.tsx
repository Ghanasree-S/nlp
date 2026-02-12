import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { MindMapNode, MindMapEdge } from '../types';

interface D3MindMapProps {
  nodes: MindMapNode[];
  edges: MindMapEdge[];
}

interface TreeNode {
  id: string;
  label: string;
  level: number;
  nodeType: string;
  children: TreeNode[];
}

// Color schemes per branch (assigned to categories)
const BRANCH_COLORS = [
  { fill: '#6366f1', stroke: '#4f46e5', light: '#e0e7ff' },  // Indigo
  { fill: '#8b5cf6', stroke: '#7c3aed', light: '#ede9fe' },  // Purple
  { fill: '#ec4899', stroke: '#db2777', light: '#fce7f3' },  // Pink
  { fill: '#06b6d4', stroke: '#0891b2', light: '#cffafe' },  // Cyan
  { fill: '#10b981', stroke: '#059669', light: '#d1fae5' },  // Emerald
  { fill: '#f59e0b', stroke: '#d97706', light: '#fef3c7' },  // Amber
  { fill: '#ef4444', stroke: '#dc2626', light: '#fee2e2' },  // Red
  { fill: '#3b82f6', stroke: '#2563eb', light: '#dbeafe' },  // Blue
];

function truncate(text: string, maxChars: number) {
  return text.length > maxChars ? text.slice(0, maxChars - 1) + '…' : text;
}

/** Convert flat nodes + edges into a hierarchical tree for D3 */
function buildTree(nodes: MindMapNode[], edges: MindMapEdge[]): TreeNode | null {
  const mainNode = nodes.find(n => n.level === 0 || n.nodeType === 'main');
  if (!mainNode) return null;

  const catNodes = nodes.filter(n => n.level === 1 || n.nodeType === 'category');
  const detailNodes = nodes.filter(n => n.level === 2 || n.nodeType === 'detail');

  // Build parent-child from edges
  const childMap = new Map<string, string[]>();
  edges.forEach(e => {
    if (!childMap.has(e.from)) childMap.set(e.from, []);
    childMap.get(e.from)!.push(e.to);
  });

  const tree: TreeNode = {
    id: mainNode.id,
    label: mainNode.label,
    level: 0,
    nodeType: 'main',
    children: catNodes.map((cat, ci) => {
      // Find detail children of this category from edges
      const catChildIds = childMap.get(cat.id) || [];
      const catDetails = detailNodes.filter(d => catChildIds.includes(d.id));

      // If no edges link details to categories, fall back to id prefix matching
      const fallbackDetails = catDetails.length > 0 ? catDetails :
        detailNodes.filter(d => d.id?.startsWith(`det_${ci}_`));

      return {
        id: cat.id,
        label: cat.label,
        level: 1,
        nodeType: 'category',
        children: fallbackDetails.map(d => ({
          id: d.id,
          label: d.label,
          level: 2,
          nodeType: 'detail',
          children: [],
        })),
      };
    }),
  };

  return tree;
}

const D3MindMap: React.FC<D3MindMapProps> = ({ nodes: rawNodes, edges }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 800 });

  // Measure container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width: Math.max(width, 600), height: Math.max(height, 500) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // D3 render — radial tree mindmap
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    if (!svg.node() || rawNodes.length === 0) return;

    svg.selectAll('*').remove();

    const { width, height } = dimensions;

    // ---- Build hierarchy ----
    const treeData = buildTree(rawNodes, edges);
    if (!treeData) return;

    const root = d3.hierarchy<TreeNode>(treeData);

    // Compute layout radius based on container
    const radius = Math.min(width, height) / 2 - 80;

    // D3 tree layout in radial mode
    const treeLayout = d3.tree<TreeNode>()
      .size([2 * Math.PI, radius])
      .separation((a, b) => {
        // Give more space between branches at root level
        if (a.parent === b.parent) {
          const childCount = Math.max((a.children?.length || 0), (b.children?.length || 0), 1);
          return (1 + childCount * 0.15) / a.depth;
        }
        return 2 / a.depth;
      });

    treeLayout(root);

    // ---- Zoom & pan ----
    const g = svg.append('g');

    const zoomBehaviour = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => g.attr('transform', event.transform));

    const svgEl = svg as unknown as d3.Selection<SVGSVGElement, unknown, null, undefined>;
    svgEl.call(zoomBehaviour);

    // Center the tree
    const initialScale = Math.min(width / (radius * 2.6), height / (radius * 2.6), 1);
    const initialTransform = d3.zoomIdentity
      .translate(width / 2, height / 2)
      .scale(initialScale);
    svgEl.call(zoomBehaviour.transform, initialTransform);

    // ---- Defs ----
    const defs = g.append('defs');

    // Drop shadow
    const shadowFilter = defs.append('filter').attr('id', 'nodeShadow')
      .attr('x', '-40%').attr('y', '-40%').attr('width', '180%').attr('height', '180%');
    shadowFilter.append('feDropShadow')
      .attr('dx', 0).attr('dy', 2).attr('stdDeviation', 4).attr('flood-opacity', 0.15);

    // Glow for main node
    const glowFilter = defs.append('filter').attr('id', 'mainGlow')
      .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    glowFilter.append('feGaussianBlur').attr('in', 'SourceAlpha').attr('stdDeviation', 12).attr('result', 'blur');
    glowFilter.append('feFlood').attr('flood-color', '#6366f1').attr('flood-opacity', 0.35).attr('result', 'color');
    glowFilter.append('feComposite').attr('in', 'color').attr('in2', 'blur').attr('operator', 'in').attr('result', 'glow');
    const glowMerge = glowFilter.append('feMerge');
    glowMerge.append('feMergeNode').attr('in', 'glow');
    glowMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // ---- Radial coordinate helper ----
    function radialPoint(angle: number, r: number): [number, number] {
      return [(r) * Math.cos(angle - Math.PI / 2), (r) * Math.sin(angle - Math.PI / 2)];
    }

    // Assign branch color index to each category
    const categoryColorMap = new Map<string, number>();
    root.children?.forEach((child, i) => {
      categoryColorMap.set(child.data.id, i % BRANCH_COLORS.length);
    });

    // Get branch color for any node (inherit from category ancestor)
    function getBranchColor(d: d3.HierarchyNode<TreeNode>) {
      if (d.depth === 0) return { fill: '#6366f1', stroke: '#4f46e5', light: '#e0e7ff' };
      let current: d3.HierarchyNode<TreeNode> | null = d;
      while (current && current.depth > 1) current = current.parent;
      if (current) {
        const idx = categoryColorMap.get(current.data.id) ?? 0;
        return BRANCH_COLORS[idx];
      }
      return BRANCH_COLORS[0];
    }

    // ---- Draw links (organic curved) ----
    const linkGroup = g.append('g').attr('class', 'links').attr('fill', 'none');

    const allLinks = root.links();
    allLinks.forEach((link, i) => {
      const sourceAngle = link.source.x ?? 0;
      const sourceRadius = link.source.y ?? 0;
      const targetAngle = link.target.x ?? 0;
      const targetRadius = link.target.y ?? 0;

      const [sx, sy] = radialPoint(sourceAngle, sourceRadius);
      const [tx, ty] = radialPoint(targetAngle, targetRadius);

      const branchColor = getBranchColor(link.target);
      const depth = link.target.depth;
      const strokeWidth = depth === 1 ? 3.5 : 2;
      const opacity = depth === 1 ? 0.7 : 0.45;

      // Organic curved path using quadratic bezier
      const midR = (sourceRadius + targetRadius) / 2;
      const [mx, my] = radialPoint((sourceAngle + targetAngle) / 2, midR);

      linkGroup.append('path')
        .attr('d', `M${sx},${sy} Q${mx},${my} ${tx},${ty}`)
        .attr('stroke', branchColor.fill)
        .attr('stroke-width', strokeWidth)
        .attr('stroke-opacity', 0)
        .attr('stroke-linecap', 'round')
        .transition()
        .duration(700)
        .delay(i * 30 + 200)
        .attr('stroke-opacity', opacity);
    });

    // ---- Draw nodes ----
    const nodeGroup = g.append('g').attr('class', 'nodes');

    const allNodes = root.descendants();

    allNodes.forEach((d, i) => {
      const [x, y] = radialPoint(d.x ?? 0, d.y ?? 0);
      const branchColor = getBranchColor(d);
      const isMain = d.depth === 0;
      const isCat = d.depth === 1;

      const ng = nodeGroup.append('g')
        .attr('class', 'node')
        .attr('cursor', 'pointer');

      // Entrance animation: pop out from center
      ng.attr('transform', 'translate(0,0) scale(0)')
        .transition()
        .duration(600)
        .delay(i * 50 + 100)
        .ease(d3.easeBackOut.overshoot(1.3))
        .attr('transform', `translate(${x},${y}) scale(1)`);

      if (isMain) {
        // Main topic — large circle with glow
        ng.attr('filter', 'url(#mainGlow)');
        ng.append('circle')
          .attr('r', 52)
          .attr('fill', '#6366f1')
          .attr('stroke', '#4f46e5')
          .attr('stroke-width', 3);

        // Decorative inner ring
        ng.append('circle')
          .attr('r', 44)
          .attr('fill', 'none')
          .attr('stroke', 'rgba(255,255,255,0.25)')
          .attr('stroke-width', 1.5);

        // Main label — wrap text
        const label = d.data.label;
        const words = label.split(/\s+/);
        const lines: string[] = [];
        let currentLine = '';
        words.forEach(word => {
          if ((currentLine + ' ' + word).trim().length > 12) {
            if (currentLine) lines.push(currentLine);
            currentLine = word;
          } else {
            currentLine = (currentLine + ' ' + word).trim();
          }
        });
        if (currentLine) lines.push(currentLine);

        lines.forEach((line, li) => {
          ng.append('text')
            .attr('x', 0)
            .attr('y', (li - (lines.length - 1) / 2) * 16)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('fill', '#ffffff')
            .attr('font-size', 14)
            .attr('font-weight', '700')
            .attr('font-family', 'system-ui, -apple-system, sans-serif')
            .text(truncate(line, 16));
        });

      } else if (isCat) {
        // Category — rounded rectangle pill
        const labelText = truncate(d.data.label, 20);
        const pillWidth = Math.max(labelText.length * 8.5 + 28, 90);
        const pillHeight = 36;

        ng.attr('filter', 'url(#nodeShadow)');

        ng.append('rect')
          .attr('x', -pillWidth / 2)
          .attr('y', -pillHeight / 2)
          .attr('width', pillWidth)
          .attr('height', pillHeight)
          .attr('rx', pillHeight / 2)
          .attr('fill', branchColor.fill)
          .attr('stroke', branchColor.stroke)
          .attr('stroke-width', 2);

        ng.append('text')
          .attr('x', 0)
          .attr('y', 1)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'central')
          .attr('fill', '#ffffff')
          .attr('font-size', 12)
          .attr('font-weight', '600')
          .attr('font-family', 'system-ui, -apple-system, sans-serif')
          .text(labelText);

      } else {
        // Detail — smaller pill with light fill
        const labelText = truncate(d.data.label, 24);
        const pillWidth = Math.max(labelText.length * 7 + 20, 70);
        const pillHeight = 28;

        ng.attr('filter', 'url(#nodeShadow)');

        ng.append('rect')
          .attr('x', -pillWidth / 2)
          .attr('y', -pillHeight / 2)
          .attr('width', pillWidth)
          .attr('height', pillHeight)
          .attr('rx', pillHeight / 2)
          .attr('fill', branchColor.light)
          .attr('stroke', branchColor.fill)
          .attr('stroke-width', 1.5);

        ng.append('text')
          .attr('x', 0)
          .attr('y', 1)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'central')
          .attr('fill', branchColor.stroke)
          .attr('font-size', 10.5)
          .attr('font-weight', '500')
          .attr('font-family', 'system-ui, -apple-system, sans-serif')
          .text(labelText);
      }

      // ---- Hover effects ----
      ng.on('mouseenter', function () {
        d3.select(this)
          .transition().duration(200)
          .attr('transform', `translate(${x},${y}) scale(1.12)`);

        // Show full-label tooltip
        const tip = g.append('g').attr('class', 'tooltip');
        const padX = 10, padY = 6;
        const tipY = y - (isMain ? 66 : isCat ? 30 : 24);
        const tipText = tip.append('text')
          .attr('x', x)
          .attr('y', tipY)
          .attr('text-anchor', 'middle')
          .attr('font-size', 11)
          .attr('fill', '#ffffff')
          .attr('font-family', 'system-ui, sans-serif')
          .text(d.data.label);
        const bbox = tipText.node()?.getBBox();
        if (bbox) {
          tip.insert('rect', 'text')
            .attr('x', bbox.x - padX)
            .attr('y', bbox.y - padY)
            .attr('width', bbox.width + padX * 2)
            .attr('height', bbox.height + padY * 2)
            .attr('rx', 6)
            .attr('fill', '#1f2937')
            .attr('opacity', 0.92);
        }
      });

      ng.on('mouseleave', function () {
        d3.select(this)
          .transition().duration(200)
          .attr('transform', `translate(${x},${y}) scale(1)`);
        g.selectAll('.tooltip').remove();
      });
    });

  }, [rawNodes, edges, dimensions]);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="w-full h-full"
        style={{ background: 'transparent' }}
      />
      {/* Legend */}
      <div className="absolute bottom-4 left-4 flex items-center gap-4 px-4 py-2 bg-white/90 dark:bg-black/90 backdrop-blur-xl rounded-xl text-xs z-40 shadow border border-zinc-200 dark:border-zinc-700">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-indigo-500" />
          <span className="font-semibold">Main Topic</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-purple-500" />
          <span className="font-semibold">Category</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full border-2 border-cyan-400 bg-indigo-50" />
          <span className="font-semibold">Detail</span>
        </div>
      </div>
      {/* Instruction */}
      <div className="absolute top-4 right-4 px-3 py-1.5 bg-white/90 dark:bg-black/90 backdrop-blur-xl rounded-full text-[11px] font-medium text-zinc-500 z-40 shadow">
        Scroll to zoom · Drag to pan
      </div>
    </div>
  );
};

export default D3MindMap;
