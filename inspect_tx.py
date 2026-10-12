"""
Task 2 - semantics check on one extracted Zenodo file.

Usage:  python inspect_tx.py data\\raw\\sample_a\\btc-tx-480000-489999-aa

Checks:
  1. First/last transaction key (does the file boundary cut a transaction in half?)
  2. Every key missing BTC-IN or BTC-OUT, with its raw rows
  3. Coinbase handling: what do tx_index 0 rows look like?
  4. Unit check: block reward at these heights is 12.5 BTC = 1,250,000,000 satoshi
  5. Fee = sum(IN) - sum(OUT): distribution and outliers
  6. Endpoint strings that are neither a 64-hex hash nor a normal address
"""
import collections, re, sys

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ADDR = re.compile(r"^([13][1-9A-HJ-NP-Za-km-z]{25,34}|bc1[0-9a-z]{11,87})$")


def pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]

    s_in, s_out = collections.Counter(), collections.Counter()
    n_in, n_out = collections.Counter(), collections.Counter()
    first = last = None
    cb_rows = collections.defaultdict(list)       # tx_index 0 rows per block
    endpoints = collections.Counter()             # endpoints that are in the address slot
    odd = collections.Counter()
    order_ok = True
    prev = None

    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            t, b, i, frm, to, v = line.split()
            key, v = (int(b), int(i)), int(v)
            first = first or key
            if prev and key < prev:
                order_ok = False
            prev = last = key
            if t == "BTC-IN":
                s_in[key] += v; n_in[key] += 1; addr = frm
            else:
                s_out[key] += v; n_out[key] += 1; addr = to
            endpoints[addr] += 1
            if not ADDR.match(addr) and not HEX64.match(addr):
                odd[addr] += 1
            if key[1] == 0 and len(cb_rows[key[0]]) < 6:
                cb_rows[key[0]].append(line.rstrip())

    keys = set(n_in) | set(n_out)
    print(f"transaction keys: {len(keys):,}   sorted by (block, tx): {order_ok}")
    print(f"first key: {first}   last key: {last}")

    incomplete = [k for k in keys if n_in[k] == 0 or n_out[k] == 0]
    print(f"\n[2] keys missing IN or OUT: {len(incomplete)}")
    wanted = set(incomplete[:10])
    for k in sorted(wanted):
        print(f"  key={k} IN rows={n_in[k]} OUT rows={n_out[k]} "
              f"(is last key in file: {k == last}, is first: {k == first})")
    if wanted:
        with open(path, "r", encoding="ascii", errors="replace") as f:
            shown = 0
            for line in f:
                p = line.split()
                if (int(p[1]), int(p[2])) in wanted and shown < 20:
                    print("   ", line.rstrip()); shown += 1

    print("\n[3] coinbase (tx_index 0), first 3 blocks:")
    for b in sorted(cb_rows)[:3]:
        for r in cb_rows[b]:
            print("   ", r)

    cb_out = sorted(s_out[(b, 0)] for b in cb_rows if (b, 0) in s_out)
    if cb_out:
        print(f"\n[4] tx 0 OUT total per block (satoshi): min={cb_out[0]:,} "
              f"median={pct(cb_out, .5):,} max={cb_out[-1]:,}")
        print("    expect >= 1,250,000,000 (12.5 BTC reward) if values are satoshi")
    cb_in_missing = sum(1 for b in cb_rows if n_in[(b, 0)] == 0)
    print(f"    blocks whose tx 0 has no BTC-IN row: {cb_in_missing} of {len(cb_rows)}")

    fees = []
    for k in keys:
        if k[1] != 0 and n_in[k] and n_out[k]:
            fees.append(s_in[k] - s_out[k])
    fees.sort()
    neg = sum(1 for x in fees if x < 0)
    zero = sum(1 for x in fees if x == 0)
    print(f"\n[5] fee = IN - OUT over {len(fees):,} complete txs: negative={neg:,} zero={zero:,}")
    print(f"    min={fees[0]:,} p50={pct(fees, .5):,} p99={pct(fees, .99):,} max={fees[-1]:,}")
    print(f"    txs with fee > 1 BTC (1e8 sat): {sum(1 for x in fees if x > 100_000_000):,}")
    print("    note: tx 0 'fee' is the block's fees if its IN row carries the reward, so exclude tx 0 from fee stats")

    print(f"\n[6] address-slot strings that are neither address nor 64-hex: {sum(odd.values()):,}")
    for s, c in odd.most_common(10):
        print(f"    {s!r}: {c:,}")
    print("    most frequent address-slot strings:")
    for s, c in endpoints.most_common(5):
        print(f"    {s}: {c:,}")


if __name__ == "__main__":
    main()
