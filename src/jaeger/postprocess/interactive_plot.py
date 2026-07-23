"""Interactive HTML genome plots for Jaeger prophage prediction.

Generates self-contained HTML files with D3.js for scrollable, zoomable
genome visualisation including gene annotations, tRNA markers, phage scores,
and prophage region highlighting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("jaeger")


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


# Phage hallmark genes for false positive filtering
PHAGE_HALLMARK_GENES = [
    "integrase",
    "terminase",
    "capsid",
    "tail",
    "portal",
    "protease",
    "lysis",
    "holin",
    "endolysin",
    "repressor",
    "cro",
    "excisionase",
    "phage",
    "prophage",
]


def _is_phage_hallmark(product: str) -> bool:
    """Check if a gene product is a phage hallmark gene."""
    product_lower = product.lower()
    return any(hallmark in product_lower for hallmark in PHAGE_HALLMARK_GENES)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {{
    --bg: #ffffff;
    --text: #1a1a18;
    --text-muted: #666;
    --border: #d3d1c7;
    --chart-bg: #ffffff;
    --chart-border: #eee;
    --chart-axis: #333;
    --chart-axis-text: #555;
    --chart-text: #333;
    --gene-stroke: #333;
    --tooltip-bg: rgba(255, 255, 255, 0.95);
    --tooltip-border: #ccc;
    --tooltip-text: #1a1a18;
    --tooltip-shadow: rgba(0,0,0,0.2);
    --table-bg: #fff;
    --table-header-bg: #f0efea;
    --table-header-text: #666;
    --table-row-even: #fafaf9;
    --table-row-hover: rgba(0,0,0,0.03);
    --table-row-highlight: rgba(230,159,0,0.15);
    --overview-viewport-fill: rgba(70,130,180,0.15);
    --overview-viewport-stroke: #4682b4;
  }}
  body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 20px;
    background: var(--bg);
    color: var(--text);
  }}
  #controls {{ margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  #controls button, #controls label {{
    padding: 4px 8px;
    background: var(--bg);
    color: var(--text);
    border: 0.5px solid var(--border);
    border-radius: 6px;
    font-size: 12px;
  }}
  #controls button {{ cursor: pointer; }}
  #controls button:hover {{ background: #f0efea; }}
  #overview {{ width: 100%; height: 55px; margin-bottom: 6px; border: 1px solid var(--border); background: #fafafa; }}
  .overview-axis text {{ font-size: 10px; fill: var(--chart-axis-text); }}
  #chart {{ width: 100%; overflow: hidden; border: 1px solid var(--chart-border); background: var(--chart-bg); cursor: grab; touch-action: none; }}
  #chart:active {{ cursor: grabbing; }}
  .axis path, .axis line {{ fill: none; stroke: var(--chart-axis); shape-rendering: crispEdges; }}
  .axis text {{ font-size: 12px; fill: var(--chart-axis-text); }}
  .axis-label {{ font-size: 14px; fill: var(--chart-axis-text); }}
  .track-label {{ font-size: 12px; font-weight: bold; fill: var(--chart-text); }}
  .gene {{ stroke: none; cursor: pointer; opacity: 0.85; }}
  .gene:hover {{ opacity: 1; }}
  .gene.selected {{ opacity: 1; }}
  .trna-marker {{ stroke: #e377c2; stroke-width: 2px; fill: #e377c2; opacity: 0.9; }}
  .prophage-region {{ fill: #e69f00; opacity: 0.2; stroke: #e69f00; stroke-width: 2px; stroke-dasharray: 4,2; cursor: pointer; }}
  .prophage-region:hover {{ opacity: 0.35; stroke-width: 3px; }}
  .prophage-region.selected {{ opacity: 0.5; stroke-width: 4px; stroke: #d55e00; }}
  .label {{ font-size: 10px; fill: var(--chart-text); pointer-events: none; font-family: monospace; }}
  .label.hidden {{ display: none; }}
  .tooltip {{
    position: absolute;
    text-align: left;
    padding: 8px;
    font-size: 12px;
    background: var(--tooltip-bg);
    border: 1px solid var(--tooltip-border);
    border-radius: 4px;
    pointer-events: none;
    box-shadow: 2px 2px 6px var(--tooltip-shadow);
    color: var(--tooltip-text);
    z-index: 1000;
  }}
  .legend {{ font-size: 12px; margin-top: 18px; }}
  .legend-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 6px 14px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text);
  }}
  .legend-swatch {{
    width: 12px;
    height: 12px;
    border: 0.5px solid var(--chart-axis);
    flex-shrink: 0;
  }}
  #annotation-table, #prophage-table {{ margin-top: 16px; }}
  #annotation-table h4, #prophage-table h4 {{
    margin: 0 0 10px 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  .table-wrap {{
    border: 0.5px solid var(--border);
    border-radius: 10px;
    overflow-x: auto;
    background: var(--table-bg);
  }}
  #annotation-table table, #prophage-table table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
    background: transparent;
  }}
  #annotation-table th, #prophage-table th {{
    background: var(--table-header-bg);
    font-weight: 600;
    font-size: 11px;
    color: var(--table-header-text);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 8px 10px;
    text-align: left;
    border-bottom: 0.5px solid var(--border);
    white-space: nowrap;
  }}
  #annotation-table td, #prophage-table td {{
    padding: 7px 10px;
    text-align: left;
    border-bottom: 0.5px solid var(--border);
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: top;
    cursor: pointer;
    color: var(--text);
  }}
  #annotation-table tr:nth-child(even), #prophage-table tr:nth-child(even) {{ background-color: var(--table-row-even); }}
  #annotation-table tr:hover td, #prophage-table tr:hover td {{ background: var(--table-row-hover); }}
  #annotation-table tr.highlight td, #prophage-table tr.highlight td {{ background: var(--table-row-highlight); font-weight: 600; }}
  .pagination {{
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-muted);
  }}
  .pagination button {{
    padding: 4px 10px;
    font-size: 12px;
    border: 0.5px solid var(--border);
    border-radius: 6px;
    background: var(--bg);
    cursor: pointer;
    color: var(--text);
  }}
  .pagination button:hover:not(:disabled) {{ background: #f0efea; }}
  .pagination button:disabled {{ opacity: 0.4; cursor: default; }}
  .zoom-hint {{ font-size: 11px; color: var(--text-muted); margin-left: auto; }}
  .header {{
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 20px;
    background: var(--bg);
    border: 0.5px solid var(--border);
    padding: 16px 20px;
  }}
  .header h2 {{ margin: 0 0 10px 0; font-size: 18px; }}
  .header .metric {{ display: inline-block; margin-right: 20px; font-size: 13px; }}
  .header .metric .label {{ color: var(--text-muted); font-size: 11px; text-transform: uppercase; }}
  .header .metric .value {{ font-weight: 600; }}
</style>
</head>
<body>

<div class="header">
  <h2>{contig_id}</h2>
  <div class="metric"><span class="label">Length</span><br><span class="value">{length:,} bp</span></div>
  <div class="metric"><span class="label">Host</span><br><span class="value">{host}</span></div>
  <div class="metric"><span class="label">Prophages</span><br><span class="value">{prophage_count}</span></div>
  <div class="metric"><span class="label">Genes</span><br><span class="value">{gene_count}</span></div>
  <div class="metric"><span class="label">tRNAs</span><br><span class="value">{trna_count}</span></div>
  <div class="metric"><span class="label">Gene Density</span><br><span class="value">{gene_density} genes/kb</span></div>
</div>

<div id="controls">
  <button id="pan-left">&larr; Pan</button>
  <button id="pan-right">Pan &rarr;</button>
  <button id="zoom-in">Zoom in</button>
  <button id="zoom-out">Zoom out</button>
  <button id="zoom-reset">Reset</button>
  <label><input type="checkbox" id="show-labels" checked> Show labels</label>
  <span class="zoom-hint">Wheel = zoom &middot; Shift+wheel = pan</span>
</div>
<div id="overview"></div>
<div id="chart"></div>
<div id="legend"></div>
<div id="prophage-table"></div>
<div id="annotation-table"></div>

<script>
(function() {{
  const data = {data_json};
  const contigLength = data.length;
  const genes = data.genes;
  const trnas = data.trnas;
  const prophages = data.prophages;
  const scores = data.scores;

  // Merge genes and tRNAs for display
  const allFeatures = genes.map(function(d) {{ return {{...d, type: "cds"}}; }})
    .concat(trnas.map(function(d) {{ return {{...d, type: "trna"}}; }}));
  allFeatures.forEach(function(d, i) {{ d._idx = i; }});

  const margin = {{top: 55, right: 20, bottom: 55, left: 80}};
  const trackHeight = 55;
  const trackGap = 35;
  const featureHeight = 22;
  const laneGap = 6;
  const laneStep = featureHeight + laneGap;
  const scoreTrackHeight = 60;

  function assignLanes(features) {{
    const byStrand = {{1: [], "-1": []}};
    features.forEach(function(d) {{
      byStrand[d.strand === 1 ? 1 : -1].push(d);
    }});
    [1, -1].forEach(function(strand) {{
      const feats = byStrand[strand].sort(function(a, b) {{
        return a.start - b.start || a.end - b.end;
      }});
      const laneEnds = [];
      feats.forEach(function(d) {{
        let lane = -1;
        for (let i = 0; i < laneEnds.length; i++) {{
          if (d.start >= laneEnds[i]) {{
            lane = i;
            break;
          }}
        }}
        if (lane === -1) {{
          lane = laneEnds.length;
          laneEnds.push(0);
        }}
        d.lane = lane;
        laneEnds[lane] = Math.max(laneEnds[lane] || 0, d.end);
      }});
    }});
  }}
  assignLanes(allFeatures);

  function featureCenterY(d) {{
    const base = d.strand === 1 ? forwardY : reverseY;
    const offset = (d.lane || 0) * laneStep;
    return d.strand === 1 ? base - offset : base + offset;
  }}

  const maxForwardLane = d3.max(allFeatures.filter(function(d) {{ return d.strand === 1; }}), function(d) {{ return d.lane || 0; }}) || 0;
  const maxReverseLane = d3.max(allFeatures.filter(function(d) {{ return d.strand === -1; }}), function(d) {{ return d.lane || 0; }}) || 0;
  const maxLaneOffset = Math.max(maxForwardLane, maxReverseLane) * laneStep;
  const lanePadding = maxLaneOffset + 10;

  // Track layout: score track on top, then forward genes, then reverse genes, then coordinate bar
  const scoreTrackY = 0;
  const forwardY = scoreTrackHeight + lanePadding + trackHeight / 2;
  const reverseY = forwardY + trackHeight + trackGap;
  const coordBarY = reverseY + trackHeight / 2 + lanePadding;
  const coordBarHeight = 30;
  const trackHeightTotal = coordBarY + coordBarHeight;
  const plotHeight = trackHeightTotal;
  const overviewHeight = 55;

  const chartDiv = document.getElementById("chart");
  let width = Math.max(600, chartDiv.clientWidth) - margin.left - margin.right;

  const x = d3.scaleLinear().domain([0, contigLength]).range([0, width]);
  let currentXScale = x.copy();
  let currentTransform = d3.zoomIdentity;

  let showLabels = true;
  const minZoom = 1;
  const maxZoom = 50;
  const zoomStep = 1.3;
  const panStep = 0.2;

  const svg = d3.select("#chart")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", plotHeight + margin.top + margin.bottom)
    .style("display", "block");

  const plotWrapper = svg.append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  plotWrapper.append("defs").append("clipPath")
      .attr("id", "plot-clip")
    .append("rect")
      .attr("width", width)
      .attr("height", plotHeight);

  const plotArea = plotWrapper.append("g")
      .attr("class", "plot-area")
      .attr("clip-path", "url(#plot-clip)");

  const staticGroup = plotWrapper.append("g").attr("class", "static-group");

  const zoom = d3.zoom()
    .scaleExtent([minZoom, maxZoom])
    .extent([[0, 0], [width, plotHeight]])
    .translateExtent([[0, 0], [width, plotHeight]])
    .on("zoom", function(event) {{
      currentTransform = event.transform;
      currentXScale = event.transform.rescaleX(x);
      updateView();
    }});

  const overlay = svg;
  svg.call(zoom);

  // Score track (top)
  const scoreScale = d3.scaleLinear().domain([0, 4]).range([scoreTrackHeight, 0]);
  const scoreGroup = plotArea.append("g").attr("class", "score-group");

  // Score toggles
  const scoreColors = {{
    "phage": "#e69f00",
    "bacteria": "#0072b2",
    "archaea": "#009e73",
    "eukarya": "#cc79a7",
    "plasmid": "#d55e00",
    "virus": "#56b4e9"
  }};
  const visibleScores = new Set(["phage"]);

  // Create toggle buttons
  const toggleContainer = d3.select("#controls").append("span").attr("id", "score-toggles");
  toggleContainer.append("span").text("Scores: ").style("margin-left", "10px");
  const classCols = data.class_cols || ["phage"];
  classCols.forEach(function(col) {{
    const color = scoreColors[col] || "#999";
    const btn = toggleContainer.append("button")
      .attr("class", "score-toggle")
      .attr("data-score", col)
      .style("background", visibleScores.has(col) ? color : "#fff")
      .style("color", visibleScores.has(col) ? "#fff" : color)
      .style("border", "1px solid " + color)
      .style("margin-left", "4px")
      .text(col)
      .on("click", function() {{
        if (visibleScores.has(col)) {{
          visibleScores.delete(col);
          d3.select(this).style("background", "#fff").style("color", color);
        }} else {{
          visibleScores.add(col);
          d3.select(this).style("background", color).style("color", "#fff");
        }}
        drawScores();
      }});
  }});

  // Prophage regions
  const prophageGroup = plotArea.append("g").attr("class", "prophage-group");

  // Gene track
  const geneGroup = plotArea.append("g").attr("class", "gene-group");
  const labelGroup = plotArea.append("g").attr("class", "label-group");
  const axisGroup = plotArea.append("g").attr("class", "axis");
  const coordBarGroup = plotArea.append("g").attr("class", "coord-bar");

  staticGroup.append("text")
    .attr("class", "track-label")
    .attr("x", -10)
    .attr("y", scoreTrackY + scoreTrackHeight / 2)
    .attr("text-anchor", "end")
    .attr("dominant-baseline", "middle")
    .text("Phage score");

  staticGroup.append("text")
    .attr("class", "track-label")
    .attr("x", -10)
    .attr("y", forwardY)
    .attr("text-anchor", "end")
    .attr("dominant-baseline", "middle")
    .text("Forward (+)");

  staticGroup.append("text")
    .attr("class", "track-label")
    .attr("x", -10)
    .attr("y", reverseY)
    .attr("text-anchor", "end")
    .attr("dominant-baseline", "middle")
    .text("Reverse (-)");

  const axisLabel = plotArea.append("text")
    .attr("class", "axis-label")
    .attr("y", plotHeight + 40)
    .style("text-anchor", "middle")
    .text("Position (bp)");

  const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

  // Overview bar
  const overviewDiv = d3.select("#overview");
  const overviewWidth = overviewDiv.node().clientWidth;
  const overviewSvg = overviewDiv.append("svg")
    .attr("width", overviewWidth)
    .attr("height", overviewHeight);
  const overviewX = d3.scaleLinear().domain([0, contigLength]).range([0, overviewWidth]);
  const overviewGroup = overviewSvg.append("g");

  const overviewAxis = d3.axisTop(overviewX)
    .ticks(Math.max(2, Math.floor(overviewWidth / 80)))
    .tickSize(4)
    .tickFormat(d3.format("~s"));
  overviewGroup.append("g")
    .attr("class", "overview-axis")
    .attr("transform", "translate(0,22)")
    .call(overviewAxis)
    .selectAll("text")
    .attr("dy", "-2px");

  const overviewGeneHeight = 8;
  const overviewForwardY = 26;
  const overviewReverseY = 38;

  // Overview genes
  overviewGroup.selectAll(".overview-gene")
    .data(allFeatures)
    .enter()
    .append("rect")
    .attr("class", "overview-gene")
    .attr("x", function(d) {{ return overviewX(d.start); }})
    .attr("y", function(d) {{ return d.strand === 1 ? overviewForwardY : overviewReverseY; }})
    .attr("width", function(d) {{ return Math.max(1, overviewX(d.end) - overviewX(d.start)); }})
    .attr("height", overviewGeneHeight)
    .attr("fill", function(d) {{ return d.type === "trna" ? "#cc79a7" : (d.strand === 1 ? "#0072b2" : "#009e73"); }})
    .attr("stroke", "none");

  // Overview prophage regions
  overviewGroup.selectAll(".overview-prophage")
    .data(prophages)
    .enter()
    .append("rect")
    .attr("class", "overview-prophage")
    .attr("x", function(d) {{ return overviewX(d.start); }})
    .attr("y", 20)
    .attr("width", function(d) {{ return Math.max(1, overviewX(d.end) - overviewX(d.start)); }})
    .attr("height", 20)
    .attr("fill", "#e69f00")
    .attr("opacity", 0.2)
    .attr("stroke", "#e69f00")
    .attr("stroke-width", 1);

  const viewportRect = overviewGroup.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", overviewWidth)
    .attr("height", overviewHeight)
    .attr("fill", "var(--overview-viewport-fill)")
    .attr("stroke", "var(--overview-viewport-stroke)")
    .attr("stroke-width", 2)
    .style("pointer-events", "none");

  function updateOverview() {{
    const visibleMin = Math.max(0, currentXScale.invert(0));
    const visibleMax = Math.min(contigLength, currentXScale.invert(width));
    viewportRect
      .attr("x", overviewX(visibleMin))
      .attr("width", Math.max(2, overviewX(visibleMax) - overviewX(visibleMin)));
  }}

  overviewSvg.on("click", function(event) {{
    const [mx] = d3.pointer(event, overviewGroup.node());
    const bp = Math.max(0, Math.min(contigLength, overviewX.invert(mx)));
    const targetK = Math.max(1, currentTransform.k);
    const tx = width / 2 - targetK * x(bp);
    const newTransform = d3.zoomIdentity.translate(tx, 0).scale(targetK);
    overlay.call(zoom.transform, clampTransform(newTransform));
  }});

  function genePath(d) {{
    const start = currentXScale(d.start);
    const end = currentXScale(d.end);
    const w = Math.max(0, end - start);
    const headPixels = Math.min(7, w / 2);
    const cy = featureCenterY(d);
    const y0 = cy - featureHeight / 2;
    const y1 = cy + featureHeight / 2;
    const mid = cy;

    if (d.strand === 1) {{
      return "M" + start + "," + y0 +
             "H" + (end - headPixels) +
             "L" + end + "," + mid +
             "L" + (end - headPixels) + "," + y1 +
             "H" + start + "Z";
    }} else {{
      return "M" + (start + headPixels) + "," + y0 +
             "H" + end +
             "V" + y1 +
             "H" + (start + headPixels) +
             "L" + start + "," + mid + "Z";
    }}
  }}

  function fillColor(d) {{
    if (d.type === "trna") return "#cc79a7";
    if (d.phage_hallmark) return "#d55e00"; // Orange for phage hallmark genes
    return d.strand === 1 ? "#0072b2" : "#009e73";
  }}

  function drawScores() {{
    if (!scores || scores.length === 0) return;
    scoreGroup.selectAll("*").remove();
    classCols.forEach(function(col) {{
      if (!visibleScores.has(col)) return;
      const color = scoreColors[col] || "#999";
      const line = d3.line()
        .x(function(d) {{ return currentXScale(d.position); }})
        .y(function(d) {{ return scoreScale(d[col] || 0); }});
      const area = d3.area()
        .x(function(d) {{ return currentXScale(d.position); }})
        .y0(scoreTrackHeight)
        .y1(function(d) {{ return scoreScale(d[col] || 0); }});

      scoreGroup.append("path")
        .datum(scores)
        .attr("d", area)
        .attr("fill", color)
        .attr("opacity", 0.2);
      scoreGroup.append("path")
        .datum(scores)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2);
    }});
  }}

  function drawProphages() {{
    prophageGroup.selectAll("*").remove();
    prophageGroup.selectAll(".prophage-region")
      .data(prophages)
      .enter()
      .append("rect")
      .attr("class", "prophage-region")
      .attr("x", function(d) {{ return currentXScale(d.start); }})
      .attr("y", scoreTrackY)
      .attr("width", function(d) {{ return Math.max(0, currentXScale(d.end) - currentXScale(d.start)); }})
      .attr("height", plotHeight)
      .attr("data-index", function(d, i) {{ return i; }})
      .on("click", function(event, d) {{
        const idx = parseInt(d3.select(this).attr("data-index"));
        d3.selectAll(".prophage-region").classed("selected", false);
        d3.select(this).classed("selected", true);
        highlightProphageTableRow(idx);
      }})
      .on("mouseover", function(event, d) {{
        tooltip.transition().duration(150).style("opacity", 0.95);
        tooltip.html("<b>Prophage region</b>" +
          "<br>start: " + d.start.toLocaleString() +
          "<br>end: " + d.end.toLocaleString() +
          "<br>score: " + d.score.toFixed(2))
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      }})
      .on("mousemove", function(event) {{
        tooltip.style("left", (event.pageX + 10) + "px")
               .style("top", (event.pageY - 28) + "px");
      }})
      .on("mouseout", function() {{
        tooltip.transition().duration(300).style("opacity", 0);
      }});
  }}

  function drawGenes() {{
    const geneSel = geneGroup.selectAll(".gene").data(allFeatures);
    const genesEnter = geneSel.enter().append("path").attr("class", "gene");
    genesEnter.merge(geneSel)
      .attr("d", genePath)
      .attr("fill", fillColor)
      .attr("data-index", function(d) {{ return d._idx; }})
      .on("click", function(event, d) {{
        d3.selectAll(".gene").classed("selected", false);
        d3.select(this).classed("selected", true);
        highlightTableRow(d._idx);
      }})
      .on("mouseover", function(event, d) {{
        tooltip.transition().duration(150).style("opacity", 0.95);
        tooltip.html("<b>" + (d.label || "feature") + "</b>" +
          (d.locus_tag ? "<br>locus_tag: " + d.locus_tag : "") +
          (d.product ? "<br>product: " + d.product : "") +
          (d.protein_id ? "<br>protein_id: " + d.protein_id : "") +
          (d.inference ? "<br>inference: " + d.inference : "") +
          (d.note ? "<br>note: " + d.note : "") +
          (d.phage_hallmark ? "<br><b>PHAGE HALLMARK</b>" : "") +
          "<br>type: " + (d.type || "n/a") +
          "<br>start: " + d.start +
          "<br>end: " + d.end +
          "<br>strand: " + (d.strand === 1 ? "+" : "-"))
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      }})
      .on("mousemove", function(event) {{
        tooltip.style("left", (event.pageX + 10) + "px")
               .style("top", (event.pageY - 28) + "px");
      }})
      .on("mouseout", function() {{
        tooltip.transition().duration(300).style("opacity", 0);
      }});
    geneSel.exit().remove();
  }}

  function hideOverlappingLabels() {{
    if (!showLabels) return;
    const labels = labelGroup.selectAll(".label").nodes();
    const padding = 4;
    const items = labels.map(function(node) {{
      const bbox = node.getBBox();
      return {{node: node, left: bbox.x, right: bbox.x + bbox.width}};
    }}).sort(function(a, b) {{ return a.left - b.left; }});

    let lastRight = -Infinity;
    items.forEach(function(item) {{
      if (item.left < lastRight + padding) {{
        d3.select(item.node).classed("hidden", true);
      }} else {{
        d3.select(item.node).classed("hidden", false);
        lastRight = item.right;
      }}
    }});
  }}

  function drawLabels() {{
    if (!showLabels) {{
      labelGroup.selectAll(".label").remove();
      return;
    }}
    // Only show labels for features that are large enough to be visible
    const labelData = allFeatures.filter(function(d) {{
      const widthPx = currentXScale(d.end) - currentXScale(d.start);
      return d.label && d.label.toLowerCase() !== "unknown" && widthPx > 30;
    }});
    const labels = labelGroup.selectAll(".label").data(labelData);
    const labelsEnter = labels.enter().append("text").attr("class", "label");
    labelsEnter.merge(labels)
      .attr("x", function(d) {{ return (currentXScale(d.start) + currentXScale(d.end)) / 2; }})
      .attr("y", function(d) {{
        const cy = featureCenterY(d);
        return d.strand === 1 ? cy - featureHeight / 2 - 4 : cy + featureHeight / 2 + 12;
      }})
      .attr("text-anchor", "middle")
      .classed("hidden", false)
      .text(function(d) {{ return d.label; }});
    labels.exit().remove();

    if (labelsEnter.size() > 0 || labels.size() > 0) {{
      requestAnimationFrame(hideOverlappingLabels);
    }}
  }}

  function drawLegend() {{
    const legendItems = [
      ["CDS (+)", "#0072b2"],
      ["CDS (-)", "#009e73"],
      ["Phage hallmark", "#d55e00"],
      ["tRNA", "#cc79a7"],
      ["Prophage region", "#e69f00"]
    ];
    const legend = d3.select("#legend");
    legend.selectAll("*").remove();
    legend.append("div")
      .attr("class", "legend-grid-title")
      .text("Legend");
    const grid = legend.append("div").attr("class", "legend-grid");
    const items = grid.selectAll(".legend-item")
      .data(legendItems)
      .enter()
      .append("div")
      .attr("class", "legend-item");
    items.append("div")
      .attr("class", "legend-swatch")
      .style("background-color", function(d) {{ return d[1]; }});
    items.append("div")
      .text(function(d) {{ return d[0]; }});
  }}

  function updateView() {{
    axisGroup.attr("transform", "translate(0," + (plotHeight + 15) + ")")
      .call(d3.axisBottom(currentXScale).ticks(10).tickFormat(function(d) {{
        if (d >= 1000000) return (d / 1000000).toFixed(1) + "M";
        if (d >= 1000) return (d / 1000).toFixed(d % 1000 === 0 ? 0 : 1) + "k";
        return d;
      }}));

    axisLabel.attr("x", width / 2);

    // Update coordinate bar (dynamic ruler, shows visible region ticks)
    const visibleMin = Math.max(0, currentXScale.invert(0));
    const visibleMax = Math.min(contigLength, currentXScale.invert(width));
    const isFullZoomOut = visibleMin <= 0 && visibleMax >= contigLength;
    coordBarGroup.selectAll("*").remove();
    coordBarGroup.attr("transform", "translate(0," + coordBarY + ")");
    if (!isFullZoomOut) {{
      // Draw ruler background
      coordBarGroup.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", coordBarHeight)
        .attr("fill", "#f5f5f3")
        .attr("stroke", "#d3d1c7")
        .attr("stroke-width", 0.5);
      // Draw ruler ticks with coordinates (multiples of 10)
      const visibleSpan = visibleMax - visibleMin;
      const tickCount = Math.max(3, Math.floor(width / 120));
      // Calculate tick step as a multiple of 10 appropriate for the zoom level
      let tickStep = Math.pow(10, Math.floor(Math.log10(visibleSpan / tickCount)));
      // Adjust to be a multiple of 1, 2, or 5 times the power of 10
      const candidates = [1, 2, 5, 10];
      for (const c of candidates) {{
        if (tickStep * c * tickCount >= visibleSpan) {{
          tickStep *= c;
          break;
        }}
      }}
      // Start at the first multiple of tickStep within the visible range
      const firstTick = Math.ceil(visibleMin / tickStep) * tickStep;
      for (let bp = firstTick; bp <= visibleMax; bp += tickStep) {{
        const xPos = currentXScale(bp);
        // Tick line
        coordBarGroup.append("line")
          .attr("x1", xPos)
          .attr("y1", 0)
          .attr("x2", xPos)
          .attr("y2", 8)
          .attr("stroke", "#666")
          .attr("stroke-width", 1);
        // Tick label
        coordBarGroup.append("text")
          .attr("x", xPos)
          .attr("y", coordBarHeight - 4)
          .attr("text-anchor", "middle")
          .attr("font-size", "10px")
          .attr("fill", "#666")
          .text(Math.round(bp).toLocaleString());
      }}
    }}

    drawScores();
    drawProphages();
    geneGroup.selectAll(".gene").attr("d", genePath);
    drawLabels();
    updateOverview();
  }}

  function setZoom(level, centerPixel) {{
    const k = Math.max(minZoom, Math.min(maxZoom, level));
    const cx = centerPixel === undefined ? width / 2 : centerPixel;
    const tx = cx - k * (cx - currentTransform.x) / currentTransform.k;
    const newTransform = d3.zoomIdentity.translate(tx, 0).scale(k);
    overlay.call(zoom.transform, clampTransform(newTransform));
  }}

  function clampTransform(t) {{
    const tx = Math.min(0, Math.max(width * (1 - t.k), t.x));
    return d3.zoomIdentity.translate(tx, 0).scale(t.k);
  }}

  function panBy(fraction) {{
    const visibleSpan = currentXScale.invert(width) - currentXScale.invert(0);
    const shiftBp = fraction * visibleSpan;
    const tx = currentTransform.x - currentTransform.k * x(shiftBp);
    const newTransform = d3.zoomIdentity.translate(tx, 0).scale(currentTransform.k);
    overlay.call(zoom.transform, clampTransform(newTransform));
  }}

  d3.select("#zoom-in").on("click", function() {{
    setZoom(currentTransform.k * zoomStep, width / 2);
  }});
  d3.select("#zoom-out").on("click", function() {{
    setZoom(currentTransform.k / zoomStep, width / 2);
  }});
  d3.select("#zoom-reset").on("click", function() {{
    overlay.call(zoom.transform, d3.zoomIdentity);
  }});
  d3.select("#pan-left").on("click", function() {{ panBy(-panStep); }});
  d3.select("#pan-right").on("click", function() {{ panBy(panStep); }});
  d3.select("#show-labels").on("change", function() {{
    showLabels = this.checked;
    drawLabels();
  }});

  overlay.on("wheel", function(event) {{
    event.preventDefault();
    const isPan = event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY);
    const point = d3.pointer(event, overlay.node());
    if (isPan) {{
      const dx = event.deltaX || event.deltaY;
      const tx = currentTransform.x - dx;
      const newTransform = d3.zoomIdentity.translate(tx, 0).scale(currentTransform.k);
      overlay.call(zoom.transform, clampTransform(newTransform));
    }} else {{
      const factor = event.deltaY < 0 ? zoomStep : 1 / zoomStep;
      const k = Math.max(minZoom, Math.min(maxZoom, currentTransform.k * factor));
      const tx = point[0] - k * (point[0] - currentTransform.x) / currentTransform.k;
      const newTransform = d3.zoomIdentity.translate(tx, 0).scale(k);
      overlay.call(zoom.transform, clampTransform(newTransform));
    }}
  }});

  // Prophage predictions table (sorted by phage score)
  const prophageTableContainer = d3.select("#prophage-table");
  prophageTableContainer.append("h4").text("Prophage Predictions");
  const prophageTableWrap = prophageTableContainer.append("div").attr("class", "table-wrap");
  const prophageTable = prophageTableWrap.append("table");
  const prophageTheadRow = prophageTable.append("thead").append("tr");
  const prophageColumns = [
    {{"key": "index", "header": "#"}},
    {{"key": "start", "header": "Start"}},
    {{"key": "end", "header": "End"}},
    {{"key": "length", "header": "Length"}},
    {{"key": "score", "header": "Phage Score"}},
    {{"key": "hallmark_genes", "header": "Hallmark Genes"}},
    {{"key": "phage_fraction", "header": "Phage Fraction"}},
    {{"key": "gene_count", "header": "Genes"}},
    {{"key": "gene_density", "header": "Gene Density"}}
  ];
  prophageColumns.forEach(function(col) {{
    prophageTheadRow.append("th").text(col.header);
  }});
  const prophageTbody = prophageTable.append("tbody");

  // Sort prophages by score (descending)
  const sortedProphages = prophages.map(function(d, i) {{
    // Find phage hallmark genes in this region
    const regionGenes = genes.filter(function(g) {{
      return g.start >= d.start && g.end <= d.end;
    }});
    const hallmarkGenes = regionGenes.filter(function(g) {{
      return g.phage_hallmark;
    }});
    // Extract hallmark gene types (integrase, terminase, tail, etc.)
    const hallmarkTypes = [...new Set(hallmarkGenes.map(function(g) {{
      const product = (g.product || "").toLowerCase();
      for (const hallmark of ["integrase", "terminase", "capsid", "tail", "portal", "protease", "lysis", "holin", "endolysin", "repressor", "cro", "excisionase"]) {{
        if (product.includes(hallmark)) return hallmark;
      }}
      return null;
    }}).filter(Boolean))];
    const phageFraction = regionGenes.length > 0 ? hallmarkGenes.length / regionGenes.length : 0;
    const geneDensity = regionGenes.length / (d.end - d.start) * 1000; // genes per kb
    return {{
      ...d,
      index: i,
      length: d.end - d.start,
      hallmark_genes: hallmarkTypes.join(", ") || "-",
      phage_fraction: phageFraction.toFixed(2),
      gene_count: regionGenes.length,
      gene_density: geneDensity.toFixed(2)
    }};
  }}).sort(function(a, b) {{ return b.score - a.score; }});

  function renderProphageTable() {{
    const rows = prophageTbody.selectAll("tr").data(sortedProphages, function(d) {{ return d.index; }});
    rows.exit().remove();
    const rowsEnter = rows.enter().append("tr").style("cursor", "pointer");
    const allRows = rowsEnter.merge(rows);
    allRows.each(function(d) {{
      const row = d3.select(this);
      row.selectAll("td").remove();
      row.append("td").text(d.index + 1);
      row.append("td").text(d.start.toLocaleString());
      row.append("td").text(d.end.toLocaleString());
      row.append("td").text(d.length.toLocaleString());
      row.append("td").text(d.score.toFixed(2));
      row.append("td").text(d.hallmark_genes || "-");
      row.append("td").text(d.phage_fraction);
      row.append("td").text(d.gene_count);
      row.append("td").text(d.gene_density);
      row.attr("data-index", d.index);
      row.on("click", function() {{
        highlightProphageRegion(d.index);
      }});
    }});
  }}

  function highlightProphageRegion(index) {{
    d3.selectAll(".prophage-region").classed("selected", false);
    d3.select('.prophage-region[data-index="' + index + '"]').classed("selected", true);
    // Center on the prophage region
    const feature = prophages[index];
    const centerBp = (feature.start + feature.end) / 2;
    const visibleSpan = currentXScale.invert(width) - currentXScale.invert(0);
    if (centerBp < currentXScale.invert(0) || centerBp > currentXScale.invert(width) || visibleSpan > contigLength * 0.9) {{
      const targetK = currentTransform.k > 2 ? currentTransform.k : 3;
      const tx = width / 2 - targetK * x(centerBp);
      const newTransform = d3.zoomIdentity.translate(tx, 0).scale(targetK);
      overlay.call(zoom.transform, clampTransform(newTransform));
    }}
    highlightProphageTableRow(index);
  }}

  function highlightProphageTableRow(index) {{
    prophageTbody.selectAll("tr").classed("highlight", function(d) {{
      return d.index === index;
    }});
    const rowNode = prophageTbody.selectAll("tr").filter(function(d) {{ return d.index === index; }}).node();
    if (rowNode) rowNode.scrollIntoView({{ behavior: "smooth", block: "center" }});
  }}

  // Annotation table
  const tableContainer = d3.select("#annotation-table");
  tableContainer.append("h4").text("Annotations");
  const tableWrap = tableContainer.append("div").attr("class", "table-wrap");
  const table = tableWrap.append("table");
  const theadRow = table.append("thead").append("tr");
  const tableColumns = [
    {{"key": "label", "header": "Label"}},
    {{"key": "locus_tag", "header": "Locus Tag"}},
    {{"key": "product", "header": "Product"}},
    {{"key": "protein_id", "header": "Protein ID"}},
    {{"key": "type", "header": "Type"}},
    {{"key": "start", "header": "Start"}},
    {{"key": "end", "header": "End"}},
    {{"key": "strand", "header": "Strand"}},
    {{"key": "phage_hallmark", "header": "Phage Hallmark"}}
  ];
  tableColumns.forEach(function(col) {{
    theadRow.append("th").text(col.header);
  }});
  const tbody = table.append("tbody");
  const paginationDiv = tableContainer.append("div").attr("class", "pagination");
  paginationDiv.append("button")
    .attr("id", "page-prev")
    .text("Previous")
    .on("click", function() {{
      if (currentPage > 0) {{
        currentPage--;
        renderTable();
        renderPagination();
      }}
    }});
  paginationDiv.append("span").attr("class", "page-info");
  paginationDiv.append("button")
    .attr("id", "page-next")
    .text("Next")
    .on("click", function() {{
      if ((currentPage + 1) * rowsPerPage < allFeatures.length) {{
        currentPage++;
        renderTable();
        renderPagination();
      }}
    }});

  const rowsPerPage = 10;
  let currentPage = 0;

  function renderTable() {{
    const pageData = allFeatures.slice(currentPage * rowsPerPage, (currentPage + 1) * rowsPerPage);
    const rows = tbody.selectAll("tr").data(pageData, function(d) {{ return d._idx; }});
    rows.exit().remove();
    const rowsEnter = rows.enter().append("tr").style("cursor", "pointer");
    const allRows = rowsEnter.merge(rows);
    allRows.each(function(d) {{
      const row = d3.select(this);
      row.selectAll("td").remove();
      tableColumns.forEach(function(col) {{
        let val = d[col.key];
        if (val === undefined || val === null || val === "") val = "-";
        row.append("td").text(String(val));
      }});
      row.on("click", function() {{
        highlightFeature(d._idx);
      }});
    }});
  }}

  function renderPagination() {{
    const totalPages = Math.ceil(allFeatures.length / rowsPerPage) || 1;
    if (currentPage >= totalPages) currentPage = totalPages - 1;
    paginationDiv.select(".page-info")
      .text("Page " + (currentPage + 1) + " of " + totalPages + " (" + allFeatures.length + " features)");
    paginationDiv.select("#page-prev").property("disabled", currentPage === 0);
    paginationDiv.select("#page-next").property("disabled", (currentPage + 1) >= totalPages);
  }}

  function highlightFeature(index) {{
    const feature = allFeatures[index];
    const centerBp = (feature.start + feature.end) / 2;
    const visibleSpan = currentXScale.invert(width) - currentXScale.invert(0);
    if (centerBp < currentXScale.invert(0) || centerBp > currentXScale.invert(width) || visibleSpan > contigLength * 0.9) {{
      const targetK = currentTransform.k > 2 ? currentTransform.k : 3;
      const tx = width / 2 - targetK * x(centerBp);
      const newTransform = d3.zoomIdentity.translate(tx, 0).scale(targetK);
      overlay.call(zoom.transform, clampTransform(newTransform));
    }}
    d3.selectAll(".gene").classed("selected", false);
    d3.select('.gene[data-index="' + index + '"]').classed("selected", true);
    highlightTableRow(index);
  }}

  function highlightTableRow(index) {{
    const localIndex = allFeatures.findIndex(function(d) {{ return d._idx === index; }});
    if (localIndex >= 0) {{
      const page = Math.floor(localIndex / rowsPerPage);
      if (page !== currentPage) {{
        currentPage = page;
        renderTable();
        renderPagination();
      }}
      tbody.selectAll("tr").classed("highlight", function(d) {{
        return d._idx === index;
      }});
      const rowNode = tbody.selectAll("tr").filter(function(d) {{ return d._idx === index; }}).node();
      if (rowNode) rowNode.scrollIntoView({{ behavior: "smooth", block: "center" }});
    }}
  }}

  drawScores();
  drawProphages();
  drawGenes();
  drawLegend();
  updateView();
  renderProphageTable();
  renderTable();
  renderPagination();
}})();
</script>
</body>
</html>
"""


def _prepare_plot_data(
    contig_id: str,
    logits_df: Any,
    phage_cordinates: dict,
    annotations: dict | None,
    fsize: int,
    stride: int | None = None,
    fasta_path: Path | None = None,
    precomputed_genes: dict | None = None,
    precomputed_trnas: dict | None = None,
) -> dict[str, Any]:
    """Prepare JSON data for one contig's interactive plot.

    Args:
        contig_id: Contig identifier.
        logits_df: DataFrame with window-level scores for this contig.
        phage_cordinates: Prophage coordinates from segmentation.
        annotations: Optional GenBank annotations dict.
        fsize: Fragment size in bp.
        stride: Sliding-window stride in bp (default: ``fsize``).
        fasta_path: Path to FASTA file for gene prediction when annotations
            are not available.
        precomputed_genes: Optional precomputed gene predictions.
        precomputed_trnas: Optional precomputed tRNA predictions.

    Returns:
        Dict ready for JSON serialisation.
    """
    step = stride or fsize
    tmp, host, length = logits_df[contig_id]

    # Scores (extract all class scores)
    scores = []
    class_cols = [c for c in tmp.columns if c not in ("length", "gc", "gc_skew")]
    for _, row in tmp.iterrows():
        score_entry = {
            "position": int(row["length"]),
            "gc": float(row["gc"]),
            "gc_skew": float(row["gc_skew"]),
        }
        for col in class_cols:
            score_entry[col] = float(row[col])
        scores.append(score_entry)

    # Prophage regions (convert window indices to bp)
    prophages = []
    cords, scores_list = phage_cordinates.get(contig_id, [[], []])
    for (start_idx, end_idx), score in zip(cords, scores_list):
        raw_start = int(start_idx * step)
        raw_end = int((end_idx - 1) * step + fsize)
        prophages.append(
            {
                "start": raw_start,
                "end": min(raw_end, length),
                "score": float(score),
            }
        )

    # Genes and tRNAs from annotations, precomputed, or predict with pyrodigal-gv/tRNAscan-SE
    genes = []
    trnas = []
    phage_hallmark_genes = []
    if annotations and contig_id in annotations:
        ann = annotations[contig_id]
        genes = [
            {
                "start": c["start"],
                "end": c["end"],
                "strand": c["strand"],
                "label": c["label"] or c["product"] or "unknown",
                "locus_tag": c.get("locus_tag", ""),
                "protein_id": c.get("protein_id", ""),
                "inference": c.get("inference", ""),
                "note": c.get("note", ""),
                "product": c.get("product", ""),
                "phage_hallmark": _is_phage_hallmark(c.get("product", "")),
            }
            for c in ann["cds"]
        ]
        # Extract phage hallmark genes
        phage_hallmark_genes = [
            {
                "start": c["start"],
                "end": c["end"],
                "strand": c["strand"],
                "label": c["label"] or c["product"] or "unknown",
                "product": c.get("product", ""),
                "locus_tag": c.get("locus_tag", ""),
            }
            for c in ann["cds"]
            if _is_phage_hallmark(c.get("product", ""))
        ]
        trnas = [
            {
                "start": t["start"],
                "end": t["end"],
                "strand": t["strand"],
                "label": f"tRNA-{t['type']}",
                "anticodon": t.get("anticodon", ""),
            }
            for t in ann["trna"]
        ]
    elif precomputed_genes and contig_id in precomputed_genes:
        genes = precomputed_genes[contig_id]
        trnas = precomputed_trnas.get(contig_id, []) if precomputed_trnas else []
    elif fasta_path is not None:
        # Predict genes with pyrodigal-gv and tRNAs with tRNAscan-SE for FASTA input
        try:
            import pyfastx
            from jaeger.postprocess.prophage_boundaries import (
                find_genes_with_strand,
                find_trnas,
            )

            fa = pyfastx.Fasta(str(fasta_path), build_index=False)
            for record in fa:
                header = record[0].strip().replace(",", "___")
                if header == contig_id:
                    sequence = str(record[1])
                    # Predict genes with strand information
                    intervals = find_genes_with_strand(sequence)
                    genes = [
                        {
                            "start": start,
                            "end": end,
                            "strand": strand,
                            "label": f"gene_{i + 1}",
                            "phage_hallmark": False,
                        }
                        for i, (start, end, strand) in enumerate(intervals)
                    ]
                    # Predict tRNAs
                    trna_intervals = find_trnas(sequence)
                    trnas = [
                        {
                            "start": start,
                            "end": end,
                            "strand": strand,
                            "label": f"tRNA-{trna_type}",
                            "anticodon": "",
                        }
                        for start, end, strand, trna_type in trna_intervals
                    ]
                    break
        except Exception as e:
            logger.warning(f"Failed to predict genes/tRNAs for {contig_id}: {e}")

    return {
        "contig_id": contig_id,
        "length": length,
        "host": host,
        "scores": scores,
        "class_cols": class_cols,
        "genes": genes,
        "trnas": trnas,
        "prophages": prophages,
        "phage_hallmark_genes": phage_hallmark_genes,
    }


def _predict_genes_trnas_worker(
    fasta_path: str,
    contig_ids: list[str],
) -> tuple[dict, dict]:
    """Worker function to predict genes and tRNAs for a list of contigs."""
    import pyfastx
    from jaeger.postprocess.prophage_boundaries import (
        find_genes_with_strand,
        find_trnas,
    )

    genes_dict = {}
    trnas_dict = {}

    fa = pyfastx.Fasta(fasta_path, build_index=False)
    for record in fa:
        header = record[0].strip().replace(",", "___")
        if header in contig_ids:
            sequence = str(record[1])
            # Predict genes
            intervals = find_genes_with_strand(sequence)
            genes_dict[header] = [
                {
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "label": f"gene_{i + 1}",
                }
                for i, (start, end, strand) in enumerate(intervals)
            ]
            # Predict tRNAs
            trna_intervals = find_trnas(sequence)
            trnas_dict[header] = [
                {
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "label": f"tRNA-{trna_type}",
                }
                for start, end, strand, trna_type in trna_intervals
            ]

    return genes_dict, trnas_dict


def plot_scores_html(
    logits_df: dict,
    phage_cordinates: dict,
    annotations: dict | None,
    outdir: Path,
    infile_base: str,
    fsize: int,
    stride: int | None = None,
    fasta_path: Path | None = None,
    n_workers: int = 4,
) -> None:
    """Generate a self-contained interactive HTML genome plot per contig.

    Args:
        logits_df: Output of ``logits_to_df_v2``.
        phage_cordinates: Output of ``segment()``.
        annotations: Optional GenBank annotations from
            ``jaeger.seqops.genbank.parse_genbank``.
        outdir: Directory where HTML files are written.
        infile_base: Base filename prefix.
        fsize: Fragment size in bp.
        stride: Sliding-window stride in bp (default: ``fsize``).
        fasta_path: Path to FASTA file for gene prediction when annotations
            are not available.
        n_workers: Number of parallel workers for gene/tRNA prediction.
    """
    outdir = Path(outdir)

    # Precompute gene/tRNA predictions in parallel if FASTA input and no annotations
    precomputed_genes = None
    precomputed_trnas = None
    if fasta_path is not None and annotations is None:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        contig_ids = list(logits_df.keys())
        logger.info(
            f"predicting genes/tRNAs for {len(contig_ids)} contigs with {n_workers} workers"
        )

        # Split contigs into chunks for parallel processing
        chunk_size = max(1, len(contig_ids) // n_workers)
        chunks = [
            contig_ids[i : i + chunk_size]
            for i in range(0, len(contig_ids), chunk_size)
        ]

        precomputed_genes = {}
        precomputed_trnas = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _predict_genes_trnas_worker, str(fasta_path), chunk
                ): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                genes_dict, trnas_dict = future.result()
                precomputed_genes.update(genes_dict)
                precomputed_trnas.update(trnas_dict)

        logger.info(
            f"predicted {sum(len(v) for v in precomputed_genes.values())} genes and {sum(len(v) for v in precomputed_trnas.values())} tRNAs"
        )

    for contig_id in logits_df:
        data = _prepare_plot_data(
            contig_id,
            logits_df,
            phage_cordinates,
            annotations,
            fsize,
            stride,
            fasta_path=fasta_path,
            precomputed_genes=precomputed_genes,
            precomputed_trnas=precomputed_trnas,
        )
        # Compute genome-wide gene density
        gene_density = (
            len(data["genes"]) / data["length"] * 1000 if data["length"] > 0 else 0
        )
        html = _HTML_TEMPLATE.format(
            title=f"Jaeger prophage plot: {contig_id}",
            contig_id=contig_id,
            length=data["length"],
            host=data["host"],
            prophage_count=len(data["prophages"]),
            gene_count=len(data["genes"]),
            trna_count=len(data["trnas"]),
            gene_density=f"{gene_density:.2f}",
            data_json=json.dumps(_to_native(data)),
        )
        out_path = (
            outdir / f"{infile_base}_jaeger_{contig_id.split(' ')[0]}_interactive.html"
        )
        out_path.write_text(html)
        logger.info(f"interactive prophage plot saved at {out_path}")
