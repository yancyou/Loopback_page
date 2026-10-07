// Loopback project page: the paper's result tables as data (arXiv version). Tables 1-2 of the page are rendered from this
// file by page.js; the static tables (ablations, readers, inference memory, configurations) live in page.html.
// Sources: tables_downstream.tex (191M, 13 tasks), tables_downstream_300m.tex (374M), tables_downstream_760m.tex (778M, 1.4B),
// tables_compute.tex. Means are the paper's printed values; page.js checks them against the per-task numbers.
window.LOOPBACK_DATA = {
  models: [
    { key: 'dense',     name: 'Dense',     family: 'control', note: 'control' },
    { key: 'plain',     name: 'Plain',     family: 'plain',   note: 'plain loop' },
    { key: 'mhc',       name: 'mHC',       family: 'hc' },
    { key: 'hc',        name: 'HC',        family: 'hc' },
    { key: 'ihc',       name: 'iHC',       family: 'hc' },
    { key: 'hyperloop', name: 'Hyperloop', family: 'hc' },
    { key: 'mhar',      name: 'MHAR',      family: 'attn' },
    { key: 'attnres',   name: 'AttnRes',   family: 'attn' },
    { key: 'loopback',  name: 'Loopback',  family: 'ours',    note: 'ours' }
  ],
  families: { control: 'Control', plain: 'Loop', hc: 'Hyper-connection family', attn: 'Attention family', ours: 'Ours' },
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
    { name: 'Means', metrics: [
      { key: 'mean7', label: 'Mean, 7 main tasks', fmt: 2, strong: true, meanOf: ['piqa', 'hellaswag', 'winogrande', 'openbookqa', 'arc_easy', 'arc_challenge', 'lambada'] },
      { key: 'mean13', label: 'Mean, 13 tasks', fmt: 2, strong: true, meanOf: ['piqa', 'hellaswag', 'winogrande', 'openbookqa', 'commonsense_qa', 'boolq', 'race', 'arc_easy', 'arc_challenge', 'mmlu', 'anli_r1', 'rte', 'lambada'] } ] }
  ],
  // per-scale values; the arrays follow the model order listed in `models` for that scale
  scales: [
    { key: '191M', label: '191M', title: '191M, [2 → 4×4 → 2], 40% of the control, 3.82B tokens (20 TPP)', topology: '[2 → 4×4 → 2]', blocks: 20,
      models: ['dense', 'plain', 'mhc', 'hc', 'ihc', 'hyperloop', 'mhar', 'attnres', 'loopback'],
      values: {
        params:         [141.59, 56.64, 56.93, 56.65, 57.04, 56.79, 56.64, 56.67, 56.65],
        loss:           [2.6234, 2.6802, 2.6250, 2.6449, 2.6564, 2.6489, 2.6564, 2.6647, 2.6187],
        piqa:           [65.45, 64.91, 65.29, 64.58, 65.34, 64.47, 64.91, 65.29, 65.56],
        hellaswag:      [33.11, 31.93, 33.59, 32.90, 32.70, 32.94, 32.83, 32.48, 33.62],
        winogrande:     [51.46, 51.30, 51.62, 51.85, 52.64, 48.70, 49.72, 53.04, 51.62],
        openbookqa:     [31.60, 29.80, 31.20, 29.00, 32.40, 32.20, 31.00, 30.60, 31.00],
        commonsense_qa: [21.46, 18.67, 20.88, 20.15, 19.82, 21.62, 19.98, 20.23, 20.64],
        boolq:          [62.20, 47.03, 62.20, 58.32, 62.20, 60.83, 61.38, 60.52, 61.50],
        race:           [28.61, 28.13, 28.04, 29.47, 28.71, 28.71, 29.28, 27.46, 28.23],
        arc_easy:       [47.18, 46.89, 46.68, 46.13, 46.42, 45.41, 45.62, 46.17, 46.93],
        arc_challenge:  [26.54, 26.45, 25.26, 25.94, 24.66, 26.11, 26.11, 26.19, 26.54],
        mmlu:           [23.61, 25.46, 24.42, 24.20, 24.41, 24.13, 23.69, 25.09, 23.78],
        anli_r1:        [29.80, 31.80, 29.90, 32.60, 31.10, 31.30, 31.10, 31.10, 30.90],
        rte:            [49.82, 48.01, 53.43, 53.43, 54.87, 52.71, 51.62, 51.99, 53.79],
        lambada:        [23.31, 21.97, 25.19, 23.99, 23.50, 23.99, 23.05, 24.35, 25.89],
        wikitext:       [46.30, 50.30, 45.86, 47.52, 48.72, 47.87, 48.30, 48.31, 45.74],
        mean7:          [39.81, 39.03, 39.83, 39.20, 39.67, 39.12, 39.04, 39.73, 40.16],
        mean13:         [38.01, 36.33, 38.28, 37.89, 38.37, 37.93, 37.72, 38.04, 38.46]
      } },
    { key: '374M', label: '374M', title: '374M, [2 → 10×2 → 2], 58.3% of the control, 7.47B tokens (20 TPP)', topology: '[2 → 10×2 → 2]', blocks: 24,
      models: ['dense', 'plain', 'mhc', 'hc', 'ihc', 'hyperloop', 'mhar', 'attnres', 'loopback'],
      values: {
        params:         [308.33, 179.86, 180.06, 179.87, 180.53, 179.96, 179.86, 179.91, 179.88],
        loss:           [2.4198, 2.4415, 2.4062, 2.4249, 2.4217, 2.4179, 2.4223, 2.4601, 2.3971],
        piqa:           [68.44, 68.93, 67.08, 68.93, 69.31, 68.82, 68.66, 68.50, 69.04],
        hellaswag:      [39.17, 38.88, 40.32, 39.52, 39.59, 39.33, 39.46, 37.85, 40.62],
        winogrande:     [51.07, 51.22, 52.80, 50.59, 49.88, 52.25, 52.25, 51.93, 53.28],
        openbookqa:     [32.00, 33.60, 33.00, 32.00, 33.20, 33.00, 32.20, 33.60, 33.00],
        commonsense_qa: [20.15, 20.31, 20.97, 18.76, 19.16, 20.15, 19.16, 18.43, 20.48],
        boolq:          [55.50, 58.90, 59.88, 55.23, 55.17, 61.59, 58.93, 52.54, 61.71],
        race:           [29.57, 29.95, 31.20, 31.00, 30.14, 30.72, 30.43, 30.53, 32.34],
        arc_easy:       [52.74, 51.43, 54.92, 54.50, 52.48, 53.91, 53.37, 51.64, 52.78],
        arc_challenge:  [28.84, 29.52, 30.12, 29.78, 28.16, 29.52, 27.73, 28.41, 28.16],
        mmlu:           [24.46, 26.46, 23.13, 26.28, 24.51, 25.11, 24.59, 25.73, 24.80],
        anli_r1:        [32.70, 31.40, 33.30, 33.90, 31.90, 31.60, 30.20, 31.20, 33.20],
        rte:            [55.23, 53.79, 51.26, 54.51, 49.10, 54.87, 52.35, 53.43, 50.18],
        lambada:        [30.68, 29.98, 30.95, 30.70, 30.84, 31.61, 31.81, 28.27, 33.01],
        wikitext:       [33.71, 34.88, 32.77, 33.94, 33.78, 33.54, 33.83, 35.46, 32.41],
        mean7:          [43.28, 43.37, 44.17, 43.72, 43.35, 44.07, 43.64, 42.89, 44.27],
        mean13:         [40.04, 40.34, 40.69, 40.44, 39.50, 40.96, 40.09, 39.39, 40.97]
      } },
    { key: '778M', label: '778M', title: '778M, [2 → 8×2 → 6], 66.7% of the control, 15.60B tokens (20 TPP)', topology: '[2 → 8×2 → 6]', blocks: 24,
      models: ['dense', 'loopback'],
      values: {
        params: [679.55, 453.06], loss: [2.2521, 2.2334],
        piqa: [72.20, 71.76], hellaswag: [47.49, 48.82], winogrande: [52.49, 53.91], openbookqa: [35.20, 35.00], commonsense_qa: [18.51, 21.21],
        boolq: [57.49, 55.20], race: [31.10, 31.77], arc_easy: [59.47, 60.52], arc_challenge: [31.74, 33.11], mmlu: [25.59, 25.22],
        anli_r1: [33.20, 34.70], rte: [54.51, 54.87], lambada: [36.21, 37.96], wikitext: [25.80, 25.03],
        mean7: [47.83, 48.73], mean13: [42.71, 43.39]
      } },
    { key: '1.4B', label: '1.4B', title: '1.4B, [3 → 8×2 → 6], 68.0% of the control, 28.31B tokens (20 TPP)', topology: '[3 → 8×2 → 6]', blocks: 25,
      models: ['dense', 'loopback'],
      values: {
        params: [1284.61, 873.57], loss: [2.1283, 2.1073],
        piqa: [73.61, 74.21], hellaswag: [54.28, 56.36], winogrande: [55.09, 54.54], openbookqa: [37.20, 36.60], commonsense_qa: [21.95, 20.72],
        boolq: [62.39, 63.39], race: [34.35, 33.11], arc_easy: [66.08, 67.76], arc_challenge: [37.12, 38.99], mmlu: [25.12, 26.84],
        anli_r1: [30.50, 33.90], rte: [54.87, 53.43], lambada: [41.39, 42.40], wikitext: [21.09, 20.29],
        mean7: [52.11, 52.98], mean13: [45.69, 46.33]
      } }
  ],
  // Table 2: compute and efficiency (Transformer-only parameters; extra forward FLOPs relative to Dense; per-GPU training
  // throughput and peak memory; hours estimated from throughput; prefill on one H100)
  compute: {
    columns: [
      { key: 'params', label: 'Params (M)', fmt: 2 },
      { key: 'flops', label: 'Extra FLOPs', text: true },
      { key: 'tps', label: 'Train tok/s/GPU', text: true, higher: true },
      { key: 'hours', label: 'Hours per 20 TPP', fmt: 1, lower: true },
      { key: 'mem', label: 'Peak GiB/GPU', fmt: 0, lower: true },
      { key: 'prefill', label: 'Prefill tok/s', text: true, higher: true }
    ],
    scales: [
      { key: '191M', title: '191M, [2 → 4×4 → 2], 3.82B tokens, 16 sequences per GPU', rows: [
        ['dense',     141.59, '—',   '222k', 0.6, 38, '885k'],
        ['plain',     56.64,  '+0.00%', '226k', 0.6, 38, '886k'],
        ['mhc',       56.93,  '+0.15%', '169k', 0.8, 42, '594k'],
        ['hc',        56.65,  '+0.07%', '186k', 0.7, 42, '671k'],
        ['ihc',       57.04,  '+0.17%', '116k', 1.1, 67, '370k'],
        ['hyperloop', 56.79,  '+0.08%', '186k', 0.7, 42, '743k'],
        ['mhar',      56.64,  '+0.01%', '204k', 0.6, 41, '810k'],
        ['attnres',   56.67,  '+0.45%', '80k',  1.7, 50, '509k'],
        ['loopback',  56.65,  '+0.08%', '167k', 0.8, 42, '601k'] ] },
      { key: '374M', title: '374M, [2 → 10×2 → 2], 7.47B tokens, 8 sequences per GPU', rows: [
        ['dense',     308.33, '—',   '128k', 2.0, 26, '492k'],
        ['plain',     179.86, '+0.00%', '130k', 2.0, 26, '491k'],
        ['mhc',       180.06, '+0.05%', '115k', 2.3, 29, '422k'],
        ['hc',        179.87, '+0.02%', '120k', 2.2, 30, '439k'],
        ['ihc',       180.53, '+0.15%', '68k',  3.8, 52, '220k'],
        ['hyperloop', 179.96, '+0.03%', '120k', 2.2, 29, '457k'],
        ['mhar',      179.86, '+0.00%', '126k', 2.1, 27, '480k'],
        ['attnres',   179.91, '+0.46%', '47k',  5.5, 40, '294k'],
        ['loopback',  179.88, '+0.08%', '98k',  2.6, 32, '352k'] ] },
      { key: '778M', title: '778M, [2 → 8×2 → 6], 15.60B tokens, 8 sequences per GPU', rows: [
        ['dense',    679.55, '—',   '75k', 7.2, 38, '269k'],
        ['plain',    453.04, '+0.00%', '76k', 7.2, 37, '269k'],
        ['mhc',      453.33, '+0.04%', '68k', 7.9, 42, '240k'],
        ['loopback', 453.06, '+0.04%', '63k', 8.6, 44, '217k'] ] }
    ]
  }
};
