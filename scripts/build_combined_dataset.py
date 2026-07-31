"""Merge the structured-opponent dataset with the DAgger-round dataset.

Combines two training-data sources into one file, keeping provenance so the
mixture is traceable from the code alone:
- STRUCTURED: expert-vs-structured-opponent games (see
  generate_structured_opponent_dataset.py).
- DAGGER: additional states/corrections collected in a DAgger-style round
  (the trained agent's own visited states, re-labeled with expert moves).
Splitting is still done by game_id (see hex_ml.data.splitting) to avoid
leakage between the merged sources.
"""

STRUCTURED = Path('data/processed/structured_11x11_expanded_5000_imitation_records.jsonl')
DAGGER = Path('data/processed/dagger_round2_smartopen_3000_records.jsonl')
OUTPUT = Path('data/processed/structured_plus_dagger_round2_records.jsonl')
SPLIT = Path('data/splits/structured_plus_dagger_round2_split_manifest.json')
SUMMARY_JSON = Path('reports/data/structured_plus_dagger_round2_summary.json')
SUMMARY_MD = Path('reports/data/structured_plus_dagger_round2_summary.md')

def read_records(path, source_name, source_type):
    records = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                item = dict(json.loads(line))
                item.setdefault('source_dataset', source_name)
                item.setdefault('source_type', source_type)
                records.append(item)
    return records

def main():
    for path in (STRUCTURED, DAGGER):
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {OUTPUT}')
    structured = read_records(STRUCTURED, 'structured_expanded5000', 'structured_expert')
    dagger = read_records(DAGGER, 'dagger_round2_smartopen3000', 'dagger_correction')
    records = structured + dagger
    game_ids = sorted({r['game_id'] for r in records})
    random.Random(42).shuffle(game_ids)
    a = int(len(game_ids) * .70)
    b = a + int(len(game_ids) * .15)
    splits = {'train': sorted(game_ids[:a]), 'validation': sorted(game_ids[a:b]), 'test': sorted(game_ids[b:])}
    if len(set().union(*map(set, splits.values()))) != len(game_ids) or sum(map(len, splits.values())) != len(game_ids):
        raise RuntimeError('Split overlap or missing game IDs')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8') as handle:
        for r in records:
            handle.write(json.dumps(r) + '\n')
    manifest = {'schema_version':'split_manifest_v1','split_method':'by_game_id','seed':42,'source_records':str(OUTPUT),'source_datasets':{'structured_expanded5000':len(structured),'dagger_round2_smartopen3000':len(dagger)},'splits':splits,'counts':{k:len(v) for k,v in splits.items()}}
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = {'records':len(records),'games':len(game_ids),'source_counts':manifest['source_datasets'],'split_counts':manifest['counts'],'records_path':str(OUTPUT),'split_path':str(SPLIT),'inputs_preserved':[str(STRUCTURED),str(DAGGER)]}
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    SUMMARY_MD.write_text('# Structured plus DAgger Round 2 Dataset Summary\n\n' + '\n'.join(f'- {k}: {v}' for k,v in summary.items()) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()