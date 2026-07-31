from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def load_game_logs(log_dir: Path):
    logs = []
    for path in sorted(log_dir.glob('*.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(payload, dict) and 'records' in payload:
            logs.append((path, payload))
    return logs


def main() -> None:
    log_dir = Path('logs/games/structured_opponents_stage2')
    out_json = Path('logs/development/structured_stage2_quality_summary.json')
    out_md = Path('reports/data/structured_stage2_quality_summary.md')
    fig_dir = Path('reports/figures')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logs = load_game_logs(log_dir)
    by_opponent = defaultdict(lambda: {
        'games': 0,
        'turns': [],
        'winner_agent_examples': 0,
        'illegal_moves': 0,
        'winner_distribution': Counter(),
        'expert_p1_games': 0,
        'expert_p2_games': 0,
    })
    move_heat = Counter()
    all_turns = []

    for _path, payload in logs:
        records = payload['records']
        if not records:
            continue
        opponent_names = [r['agent_name'] for r in records if r.get('agent_type') != 'winner_agent']
        opponent = opponent_names[0] if opponent_names else 'unknown'
        data = by_opponent[opponent]
        data['games'] += 1
        data['turns'].append(len(records))
        all_turns.append(len(records))
        data['illegal_moves'] += sum(1 for r in records if not r.get('was_legal', True))

        final_winner = None
        for record in reversed(records):
            if record.get('final_winner') is not None:
                final_winner = record['final_winner']
                break
        data['winner_distribution'][str(final_winner)] += 1

        first_expert_record = next((r for r in records if r.get('agent_type') == 'winner_agent'), None)
        if first_expert_record and first_expert_record.get('player_to_move') == 1:
            data['expert_p1_games'] += 1
        if first_expert_record and first_expert_record.get('player_to_move') == 2:
            data['expert_p2_games'] += 1

        for record in records:
            if record.get('agent_type') == 'winner_agent':
                data['winner_agent_examples'] += 1
                move_heat[tuple(record['selected_move'])] += 1

    summary = {
        'log_dir': str(log_dir),
        'game_logs': len(logs),
        'total_turns': sum(all_turns),
        'avg_turns_per_game': mean(all_turns) if all_turns else 0,
        'by_opponent': {},
    }

    for opponent, data in sorted(by_opponent.items()):
        summary['by_opponent'][opponent] = {
            'games': data['games'],
            'winner_agent_examples': data['winner_agent_examples'],
            'avg_turns': mean(data['turns']) if data['turns'] else 0,
            'min_turns': min(data['turns']) if data['turns'] else 0,
            'max_turns': max(data['turns']) if data['turns'] else 0,
            'illegal_moves': data['illegal_moves'],
            'winner_distribution': dict(data['winner_distribution']),
            'expert_p1_games': data['expert_p1_games'],
            'expert_p2_games': data['expert_p2_games'],
        }

    out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    names = list(summary['by_opponent'].keys())
    if names:
        examples = [summary['by_opponent'][name]['winner_agent_examples'] for name in names]
        avg_turns = [summary['by_opponent'][name]['avg_turns'] for name in names]

        plt.figure(figsize=(9, 4))
        plt.bar(names, examples)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Winner-agent imitation examples')
        plt.tight_layout()
        plt.savefig(fig_dir / 'structured_stage2_examples_by_opponent.png', dpi=150)
        plt.close()

        plt.figure(figsize=(9, 4))
        plt.bar(names, avg_turns, color='orange')
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Average turns per game')
        plt.tight_layout()
        plt.savefig(fig_dir / 'structured_stage2_avg_turns_by_opponent.png', dpi=150)
        plt.close()

    if move_heat:
        size = 3
        grid = [[move_heat[(row, col)] for col in range(size)] for row in range(size)]
        plt.figure(figsize=(4, 4))
        plt.imshow(grid, cmap='viridis')
        plt.colorbar(label='winner-agent move count')
        plt.title('Stage 2 winner move heatmap')
        plt.xticks(range(size))
        plt.yticks(range(size))
        plt.tight_layout()
        plt.savefig(fig_dir / 'structured_stage2_winner_move_heatmap.png', dpi=150)
        plt.close()

    lines = [
        '# Structured Stage 2 Data Quality Summary',
        '',
        f'- game logs analyzed: {summary["game_logs"]}',
        f'- total turns: {summary["total_turns"]}',
        f'- average turns/game: {summary["avg_turns_per_game"]:.2f}',
        '',
        '## By opponent',
        '',
        '| Opponent | Games | Winner examples | Avg turns | Min-Max turns | Illegal moves | Expert P1/P2 | Winner distribution |',
        '|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for opponent, data in summary['by_opponent'].items():
        lines.append(
            f'| {opponent} | {data["games"]} | {data["winner_agent_examples"]} | '
            f'{data["avg_turns"]:.2f} | {data["min_turns"]}-{data["max_turns"]} | '
            f'{data["illegal_moves"]} | {data["expert_p1_games"]}/{data["expert_p2_games"]} | '
            f'{data["winner_distribution"]} |'
        )
    lines += [
        '',
        '## Timing note',
        '',
        'The Stage 2 logs do not yet store wall-clock elapsed seconds per game. This summary uses average turns per game as a rough size/speed proxy. Before choosing between 100, 500, or more games per agent, the next generation run should record elapsed seconds by opponent.',
        '',
        '## Figures',
        '',
        '- `reports/figures/structured_stage2_examples_by_opponent.png`',
        '- `reports/figures/structured_stage2_avg_turns_by_opponent.png`',
        '- `reports/figures/structured_stage2_winner_move_heatmap.png`',
    ]
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'wrote={out_json}')
    print(f'wrote={out_md}')


if __name__ == '__main__':
    main()
