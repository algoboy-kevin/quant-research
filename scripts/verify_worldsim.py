"""Read-only verification of worldsim artifacts; no network or git."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from quant_research.worldsim.portfolio import breadth


def verify(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    markets = pl.read_parquet(path / "markets.parquet")
    worlds = pl.read_parquet(path / "worlds.parquet")
    curves = pl.read_parquet(path / "breadth.parquet")
    expected_markets = sum(c['markets'] for c in config['groups']) * config['worlds'] * len(config['arms'])
    expected_worlds = len(config['groups']) * config['worlds'] * len(config['arms'])
    assert markets.height == expected_markets
    assert worlds.height == expected_worlds
    for table in (markets, worlds):
        assert table.select((pl.col('gross_nano') == pl.col('pnl_nano') + pl.col('fees_nano') - pl.col('rebates_nano')).all()).item()
    totals = markets.group_by('group', 'arm', 'world').agg(pl.col('pnl_nano').sum()).sort('group', 'arm', 'world')
    assert totals.equals(worlds.select('group', 'arm', 'world', 'pnl_nano').sort('group', 'arm', 'world'))
    for cfg in config['groups']:
        for arm in config['arms']:
            selected = markets.filter((pl.col('group') == cfg['group']) & (pl.col('arm') == arm)).sort('world', 'market')
            pnl = selected['pnl_nano'].to_numpy().reshape(config['worlds'], cfg['markets'])
            expected = breadth(pnl, config['seed'])
            actual = curves.filter((pl.col('group') == cfg['group']) & (pl.col('arm') == arm)).select(expected.columns)
            assert actual.equals(expected)
    for name in ('markets', 'worlds', 'breadth'):
        csv = pl.read_csv(path / f'{name}.csv', infer_schema_length=None)
        parquet = pl.read_parquet(path / f'{name}.parquet')
        assert csv.height == parquet.height and csv.columns == parquet.columns
        for column in parquet.columns:
            if parquet[column].dtype.is_integer():
                assert np.array_equal(csv[column].to_numpy(), parquet[column].to_numpy())
    assert (path / 'summary.md').stat().st_size > 0
    print(f'verified market_rows={markets.height} world_rows={worlds.height} breadth_rows={curves.height}')
    print('exact reconciliation: PASS (markets and worlds)')
    print('market-to-world aggregation: PASS')
    print('breadth recomputation: PASS')
    print('CSV/parquet integer parity: PASS')
    print('artifacts=' + ','.join(sorted(p.name for p in path.iterdir() if p.is_file())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    verify(parser.parse_args().path)
