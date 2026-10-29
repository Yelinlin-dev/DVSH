"""Account for the recorded minimum CLIP generation cost without rerunning CLIP."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DEFAULT_CLIP_TIMINGS = Path(__file__).with_name('clip_embedding_timings.json')
MODEL = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
OFFLINE_METRICS = [
    ('clip_embedding_generation_time_ms', 'CLIP Embedding Generation Time(ms)'),
    ('total_offline_without_clip_time_ms', 'Offline Without CLIP Time(ms)'),
    ('total_offline_time_ms', 'Total Offline Time(ms)'),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_fastest_clip_timing(task, cache_path, *, language_pair=None,
                             manifest_path=DEFAULT_CLIP_TIMINGS):
    """Validate the exact graph set and existing source cache, then select min/5.

    A generated cache and its original reference cache can have different NPZ
    bytes. Either is accepted only when the recorded full-set numerical check
    passed. No extrapolation is made for subsets or unsupported datasets.
    """
    path = Path(manifest_path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get('format') != 'mssr_clip_fastest_v1' or manifest.get('selection') != 'minimum_of_5_fresh_repeats':
        raise ValueError(f'Unsupported CLIP timing manifest: {path}')
    pair = language_pair or getattr(task, 'metadata', {}).get('subset')
    label = f'{task.dataset}_{pair}' if task.dataset == 'dbp15k' and pair else task.dataset
    if label not in manifest['datasets']:
        raise ValueError(f'No measured CLIP cost for {label}; explicitly use clip_timing_policy="algorithm-only" for an algorithm-only experiment')
    record = manifest['datasets'][label]
    ids = sorted(set(task.query_ids) | set(task.database_ids))
    ids_hash = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if (record.get('format') != 'clip_embedding_timing_v1' or record.get('dataset') != label
            or record.get('scope') != 'test' or record.get('model') != MODEL):
        raise ValueError(f'CLIP timing identity mismatch for {label}')
    if record.get('num_graphs') != len(ids) or record.get('graph_ids_sha256') != ids_hash:
        raise ValueError(f'CLIP timing input mismatch for {label}; full-dataset time cannot be assigned to a truncated task')
    if (not record.get('reference_embeddings_verified')
            or record.get('reference_comparison', {}).get('num_graphs') != len(ids)
            or record.get('missing_measured_ids_in_bound_cache') != 0):
        raise ValueError('CLIP timing lacks full input-cache verification')
    cache_hash = sha256_file(cache_path)
    if cache_hash not in {record.get('source_cache_sha256'), record.get('semantic_cache_sha256')}:
        raise ValueError(f'CLIP timing does not match current semantic cache: {cache_path}')
    runs = record.get('runs', [])
    if len(runs) != 5 or len({r['seed'] for r in runs}) != 5:
        raise ValueError('Expected five distinct CLIP measurement repeats')
    values = [float(r['embedding_generation_time_ms']) for r in runs]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Invalid CLIP measurement duration')
    fastest = runs[min(range(len(values)), key=values.__getitem__)]
    return {
        'clip_embedding_generation_time_ms': float(fastest['embedding_generation_time_ms']),
        'clip_timing_policy': 'fastest',
        'clip_timing_selection': 'minimum_of_5_fresh_repeats',
        'clip_timing_selected_seed': int(fastest['seed']),
        'clip_timing_repeats': len(runs),
        'clip_timing_dataset': label,
        'clip_timing_num_graphs': len(ids),
        'clip_timing_manifest': str(path.resolve()),
        'clip_timing_manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'clip_timing_record': record['measurement_record'],
        'clip_timing_record_sha256': record['measurement_record_sha256'],
        'clip_timing_model': record['model'],
        'clip_timing_model_revision': record['model_revision'],
        'clip_timing_device_name': record['device_name'],
        'clip_timing_batch_size': record['batch_size'],
        'clip_timing_semantic_cache_sha256': cache_hash,
        'clip_timing_convention': 'algorithm_measured_time_plus_fixed_separately_measured_clip_minimum',
        'clip_timing_excludes': ['model_loading', 'input_preparation', 'warmup', 'cache_write'],
    }


def account_clip_time(stats, timing):
    """Update primary Offline once; leave algorithm-stage fields unchanged."""
    if 'total_offline_without_clip_time_ms' in stats:
        raise ValueError('CLIP offline accounting has already been applied')
    original = float(stats['total_offline_time_ms'])
    cache_write = float(stats.get('relation_profile_cache_write_time_ms', 0.0))
    stats['total_offline_without_clip_time_ms'] = original
    stats['clip_embedding_generation_time_ms'] = 0.0
    stats['clip_timing_policy'] = 'algorithm-only'
    stats['semantic_model_generation_included'] = timing is not None
    if timing is not None:
        stats.update(timing)
        stats['semantic_cache_generation_time'] = 'separately_measured_minimum_of_5_repeats'
        stats['total_offline_time_ms'] = original + timing['clip_embedding_generation_time_ms']
        stats['offline_timing_scope'] = 'recorded_clip_generation_minimum_plus_' + stats.get('offline_timing_scope', 'algorithm_preparation')
    stats['total_offline_including_cache_write_time_ms'] = stats['total_offline_time_ms'] + cache_write
    return stats


def add_clip_timing_arguments(parser):
    parser.add_argument('--clip-timing-policy', choices=['fastest', 'algorithm-only'], default='fastest')
    parser.add_argument('--clip-timing-manifest', type=Path, default=DEFAULT_CLIP_TIMINGS)


def validate_legacy_timing(directory, policy):
    if directory is not None and policy != 'algorithm-only':
        raise ValueError('--embedding-timing-dir would double count CLIP; default fastest mode already includes it. Use --clip-timing-manifest for a different recorded minimum, or explicitly select --clip-timing-policy algorithm-only for legacy supplemental accounting.')
