// Loopback project page: the paper's result tables as data (arXiv version, 2026-10-27). Tables 1-2 of the page are rendered
// from this file by page.js; the static tables (ablations, readers, inference memory, configurations) live in page.html.
// Sources: tab:downstream-main (sections/tables_downstream_main.tex: 191M, 374M at 32 layers, 778M at 48 layers, nine
// models, BOS prompts), tab:scale-3b (3B), the 1.4B lm-eval JSONs behind fig:scale-1p4b (figure/scaling/data_1p4b; the
// paper prints only their means) with tab:scale-large's params and loss, and tab:compute. Means are the paper's printed
// values; page.js checks them against the per-task numbers.
window.LOOPBACK_DATA = {
  models: [
    { key: 'dense',     name: 'Standard',  family: 'control', note: 'baseline' },
    { key: 'plain',     name: 'Plain',     family: 'plain',   note: 'plain loop' },
    { key: 'mhc',       name: 'mHC',       family: 'hc' },
    { key: 'hc',        name: 'HC',        family: 'hc' },
    { key: 'ihc',       name: 'iHC',       family: 'hc' },
    { key: 'hyperloop', name: 'Hyperloop', family: 'hc' },
    { key: 'mhar',      name: 'MHAR',      family: 'attn' },
    { key: 'attnres',   name: 'AttnRes',   family: 'attn' },
    { key: 'loopback',  name: 'Loopback',  family: 'ours',    note: 'ours' }
  ],
  families: { control: 'Baseline', plain: 'Loop', hc: 'Hyper-connection family', attn: 'Attention family', ours: 'Ours' },
  // metric groups, in row order. lower: lower is better. mark: whether a best-per-row mark is computed. fmt: decimals.
  groups: [
    { name: 'Setting', metrics: [
      { key: 'params', label: 'Transformer params (M)', fmt: 2, mark: false },
      { key: 'loss', label: 'ClimbMix loss', unit: '↓', fmt: 4, lower: true, strong: true } ] },
    { name: 'Commonsense and reasoning', metrics: [
      { key: 'piqa', label: 'PIQA', fmt: 2 }, { key: 'hellaswag', label: 'HellaSwag', fmt: 2 }, { key: 'winogrande', label: 'WinoGrande', fmt: 2 },
      { key: 'openbookqa', label: 'OpenBookQA', fmt: 2 }, { key: 'commonsense_qa', label: 'CommonsenseQA', fmt: 2 } ] },
    { name: 'Reading comprehension', metrics: [ { key: 'boolq', label: 'BoolQ', fmt: 2 }, { key: 'race', label: 'RACE', fmt: 2 } ] },
    { name: 'Science', metrics: [ { key: 'arc_easy', label: 'ARC-Easy', fmt: 2 }, { key: 'arc_challenge', label: 'ARC-Challenge', fmt: 2 } ] },
    { name: 'Knowledge', metrics: [ { key: 'mmlu', label: 'MMLU', unit: '57 subjects', fmt: 2 } ] },
    { name: 'Natural language inference', metrics: [ { key: 'anli_r1', label: 'ANLI-R1', fmt: 2 }, { key: 'rte', label: 'RTE', fmt: 2 } ] },
    { name: 'Language modeling', metrics: [
      { key: 'lambada', label: 'LAMBADA', fmt: 2 }, { key: 'wikitext', label: 'WikiText word ppl', unit: '↓', fmt: 2, lower: true } ] },
    { name: 'Mean', metrics: [
      { key: 'mean13', label: 'All 13 tasks', fmt: 2, strong: true, meanOf: ['piqa', 'hellaswag', 'winogrande', 'openbookqa', 'commonsense_qa', 'boolq', 'race', 'arc_easy', 'arc_challenge', 'mmlu', 'anli_r1', 'rte', 'lambada'] } ] }
  ],
  // per-scale values; the arrays follow the model order listed in `models` for that scale
  scales: [
    { key: '191M', label: '191M', title: '191M, 20 layers, [2 → 4×4 → 2], 40.0% of the standard Transformer’s parameters, 3.82B tokens (20 TPP)', topology: '[2 → 4×4 → 2]', blocks: 20,
      models: ['dense', 'plain', 'mhc', 'hc', 'ihc', 'hyperloop', 'mhar', 'attnres', 'loopback'],
      values: {
        params:         [141.59, 56.64, 56.93, 56.65, 57.04, 56.79, 56.64, 56.66, 56.65],
        loss:           [2.6234, 2.6802, 2.6250, 2.6449, 2.6564, 2.6489, 2.6564, 2.6404, 2.6187],
        piqa:           [65.51, 64.96, 65.89, 64.74, 65.34, 65.07, 65.13, 64.47, 65.78],
        hellaswag:      [33.26, 32.14, 33.79, 33.18, 32.74, 33.04, 33.12, 33.07, 33.83],
        winogrande:     [51.70, 51.38, 52.64, 51.22, 51.70, 49.80, 50.75, 52.72, 51.14],
        openbookqa:     [32.60, 29.40, 31.40, 30.20, 32.80, 31.80, 31.80, 29.20, 32.40],
        commonsense_qa: [21.54, 19.98, 19.90, 18.67, 19.98, 21.95, 18.92, 20.48, 21.13],
        boolq:          [62.72, 46.82, 61.99, 59.60, 62.17, 60.58, 61.22, 60.73, 61.44],
        race:           [28.04, 29.09, 27.75, 29.86, 29.67, 28.61, 30.14, 29.95, 28.33],
        arc_easy:       [47.90, 46.93, 46.51, 47.14, 46.42, 45.66, 46.21, 44.28, 47.22],
        arc_challenge:  [26.45, 25.94, 25.68, 25.94, 25.09, 26.11, 26.62, 26.11, 25.77],
        mmlu:           [23.56, 24.57, 24.01, 24.13, 23.64, 24.02, 23.25, 26.57, 23.68],
        anli_r1:        [30.40, 32.20, 27.30, 32.40, 31.30, 32.90, 33.00, 31.60, 30.70],
        rte:            [49.46, 50.90, 55.60, 53.43, 53.43, 53.79, 53.79, 53.07, 54.15],
        lambada:        [24.47, 22.01, 25.46, 25.38, 24.92, 24.72, 23.91, 24.18, 26.80],
        wikitext:       [46.30, 50.30, 45.86, 47.52, 48.72, 47.87, 48.30, 47.01, 45.74],
        mean13:         [38.28, 36.64, 38.30, 38.15, 38.40, 38.31, 38.30, 38.19, 38.64]
      } },
    { key: '374M', label: '374M', title: '374M, 32 layers, [2 → 14×2 → 2], 56.3% of the standard Transformer’s parameters, 7.47B tokens (20 TPP)', topology: '[2 → 14×2 → 2]', blocks: 32,
      models: ['dense', 'plain', 'mhc', 'hc', 'ihc', 'hyperloop', 'mhar', 'attnres', 'loopback'],
      values: {
        params:         [323.02, 181.70, 181.87, 181.71, 182.52, 181.79, 181.70, 181.75, 181.72],
        loss:           [2.4395, 2.4371, 2.4118, 2.4273, 2.4199, 2.4147, 2.4310, 2.4166, 2.4008],
        piqa:           [68.82, 68.23, 68.39, 68.88, 69.48, 69.21, 69.10, 68.72, 69.10],
        hellaswag:      [38.72, 39.47, 40.44, 39.89, 40.36, 40.48, 39.33, 40.42, 41.11],
        winogrande:     [51.93, 52.96, 52.88, 52.17, 52.01, 52.49, 52.88, 51.14, 52.09],
        openbookqa:     [30.80, 34.00, 33.60, 32.80, 31.20, 32.00, 33.60, 33.20, 34.60],
        commonsense_qa: [19.98, 20.23, 19.90, 21.05, 20.31, 21.21, 20.64, 21.21, 19.74],
        boolq:          [47.13, 60.89, 58.04, 58.65, 62.42, 56.73, 55.99, 58.23, 58.90],
        race:           [29.28, 31.00, 30.81, 29.38, 30.14, 28.42, 30.91, 30.81, 29.67],
        arc_easy:       [51.26, 52.90, 53.37, 51.43, 52.65, 54.63, 52.44, 51.73, 52.15],
        arc_challenge:  [27.65, 28.58, 29.01, 28.41, 27.82, 28.41, 29.69, 28.16, 28.07],
        mmlu:           [25.25, 23.64, 25.32, 25.76, 24.85, 25.81, 26.25, 25.37, 26.24],
        anli_r1:        [33.30, 33.60, 32.00, 30.70, 32.60, 31.30, 33.40, 33.90, 32.40],
        rte:            [54.87, 53.79, 53.43, 51.26, 51.99, 50.18, 53.07, 46.57, 53.07],
        lambada:        [30.29, 31.69, 31.94, 31.85, 32.54, 32.76, 31.19, 33.05, 34.43],
        wikitext:       [34.79, 34.57, 33.34, 34.05, 33.69, 33.45, 34.23, 33.59, 32.61],
        mean13:         [39.18, 40.84, 40.70, 40.17, 40.64, 40.28, 40.65, 40.19, 40.89]
      } },
    { key: '778M', label: '778M', title: '778M, 48 layers, [3 → 21×2 → 3], 56.3% of the standard Transformer’s parameters, 15.60B tokens (20 TPP)', topology: '[3 → 21×2 → 3]', blocks: 48,
      models: ['dense', 'plain', 'mhc', 'hc', 'ihc', 'hyperloop', 'mhar', 'attnres', 'loopback'],
      values: {
        params:         [708.68, 398.63, 398.84, 398.65, 400.11, 398.74, 398.64, 398.73, 398.68],
        loss:           [2.2644, 2.2737, 2.2454, 2.2642, 2.2523, 2.2539, 2.2646, 2.2529, 2.2404],
        piqa:           [71.44, 71.55, 72.20, 71.76, 70.57, 71.16, 71.49, 71.71, 71.16],
        hellaswag:      [47.07, 47.07, 49.03, 47.74, 48.56, 48.08, 48.10, 48.72, 48.97],
        winogrande:     [52.09, 49.80, 53.43, 53.04, 53.12, 53.28, 53.51, 55.64, 53.91],
        openbookqa:     [34.80, 35.40, 35.20, 35.00, 35.60, 36.00, 34.00, 36.40, 36.20],
        commonsense_qa: [19.16, 19.41, 18.10, 19.82, 18.76, 20.97, 20.15, 21.95, 20.80],
        boolq:          [59.97, 58.99, 55.17, 60.28, 58.93, 60.37, 57.98, 57.92, 58.93],
        race:           [31.48, 32.54, 31.96, 32.06, 32.34, 32.44, 33.49, 33.21, 33.30],
        arc_easy:       [60.44, 57.58, 61.20, 60.56, 60.98, 58.63, 61.41, 58.42, 58.92],
        arc_challenge:  [31.91, 31.40, 31.83, 32.25, 33.19, 31.06, 32.59, 31.74, 32.08],
        mmlu:           [25.77, 25.10, 25.70, 25.37, 24.53, 24.84, 25.49, 25.44, 25.30],
        anli_r1:        [31.30, 32.00, 33.70, 33.20, 30.70, 33.30, 33.20, 33.30, 35.20],
        rte:            [56.32, 57.40, 52.71, 51.26, 53.79, 53.79, 50.18, 53.07, 51.99],
        lambada:        [37.76, 37.84, 39.71, 38.87, 39.78, 38.07, 38.54, 37.92, 39.74],
        wikitext:       [26.39, 26.78, 25.68, 26.49, 25.84, 26.01, 26.35, 25.95, 25.33],
        mean13:         [43.04, 42.78, 43.07, 43.17, 43.14, 43.23, 43.09, 43.50, 43.58]
      } },
    { key: '1.4B', label: '1.4B', title: '1.4B, 25 layers, [3 → 8×2 → 6], 68.0% of the standard Transformer’s parameters, 28.31B tokens (20 TPP)', topology: '[3 → 8×2 → 6]', blocks: 25,
      models: ['dense', 'loopback'],
      values: {
        params:         [1284.61, 873.57],
        loss:           [2.1283, 2.1073],
        piqa:           [73.88, 74.48],
        hellaswag:      [55.02, 56.43],
        winogrande:     [56.51, 56.99],
        openbookqa:     [37.80, 36.60],
        commonsense_qa: [23.34, 18.67],
        boolq:          [62.51, 63.24],
        race:           [34.64, 33.49],
        arc_easy:       [66.16, 67.00],
        arc_challenge:  [36.35, 37.03],
        mmlu:           [24.70, 26.81],
        anli_r1:        [31.30, 34.80],
        rte:            [57.40, 55.23],
        lambada:        [42.64, 43.72],
        wikitext:       [21.09, 20.29],
        mean13:         [46.33, 46.50]
      } },
    { key: '3B', label: '3B', title: '3B, 24 layers, [6 → 6×2 → 6], 75.0% of the standard Transformer’s parameters, 59.77B tokens (20 TPP)', topology: '[6 → 6×2 → 6]', blocks: 24,
      models: ['dense', 'loopback'],
      values: {
        params:         [2718.06, 2038.58],
        loss:           [2.0005, 1.9849],
        piqa:           [77.04, 75.90],
        hellaswag:      [62.84, 63.72],
        winogrande:     [57.22, 58.48],
        openbookqa:     [40.40, 41.80],
        commonsense_qa: [19.90, 21.87],
        boolq:          [62.23, 64.95],
        race:           [35.12, 36.17],
        arc_easy:       [69.19, 71.13],
        arc_challenge:  [40.53, 40.96],
        mmlu:           [26.26, 28.86],
        anli_r1:        [31.00, 32.90],
        rte:            [61.01, 56.68],
        lambada:        [48.50, 49.45],
        wikitext:       [17.28, 16.74],
        mean13:         [48.56, 49.45]
      } }
  ],
  // Table 2: compute and efficiency (Transformer-only parameters; extra forward FLOPs relative to the standard Transformer; per-GPU training
  // throughput and peak memory; prefill on one H100)
  compute: {
    columns: [
      { key: 'params', label: 'Params (M)', fmt: 2 },
      { key: 'flops', label: 'Extra FLOPs', text: true },
      { key: 'tps', label: 'Train tok/s/GPU', text: true, higher: true },
      { key: 'mem', label: 'Peak GiB/GPU', fmt: 0, lower: true },
      { key: 'prefill', label: 'Prefill tok/s', text: true, higher: true }
    ],
    scales: [
      { key: '191M', title: '191M, [2 → 4×4 → 2], 3.82B tokens, 16 sequences per GPU', rows: [
        ['dense',     141.59, '—',   '222k', 38, '885k'],
        ['plain',     56.64,  '+0.00%', '226k', 38, '886k'],
        ['mhc',       56.93,  '+0.15%', '169k', 42, '594k'],
        ['hc',        56.65,  '+0.07%', '186k', 42, '671k'],
        ['ihc',       57.04,  '+0.17%', '116k', 67, '370k'],
        ['hyperloop', 56.79,  '+0.08%', '186k', 42, '743k'],
        ['mhar',      56.64,  '+0.01%', '204k', 41, '810k'],
        ['loopback',  56.65,  '+0.08%', '167k', 42, '601k'] ] },
      { key: '374M', title: '374M, [2 → 10×2 → 2], 7.47B tokens, 8 sequences per GPU', rows: [
        ['dense',     308.33, '—',   '128k', 26, '492k'],
        ['plain',     179.86, '+0.00%', '130k', 26, '491k'],
        ['mhc',       180.06, '+0.05%', '115k', 29, '422k'],
        ['hc',        179.87, '+0.02%', '120k', 30, '439k'],
        ['ihc',       180.53, '+0.15%', '68k',  52, '220k'],
        ['hyperloop', 179.96, '+0.03%', '120k', 29, '457k'],
        ['mhar',      179.86, '+0.00%', '126k', 27, '480k'],
        ['loopback',  179.88, '+0.08%', '98k',  32, '352k'] ] },
      { key: '778M', title: '778M, [2 → 8×2 → 6], 15.60B tokens, 8 sequences per GPU', rows: [
        ['dense',    679.55, '—',   '75k', 38, '269k'],
        ['plain',    453.04, '+0.00%', '76k', 37, '269k'],
        ['mhc',      453.33, '+0.04%', '68k', 42, '240k'],
        ['loopback', 453.06, '+0.04%', '63k', 44, '217k'] ] }
    ]
  }
};
