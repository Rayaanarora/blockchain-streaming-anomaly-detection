"""
Task 1 - BitcoinHeist label overlap against Zenodo btc-tx files.

Usage (PowerShell, from the project root):
  python bitcoinheist_overlap.py data\\raw\\BitcoinHeistData.csv ^
      data\\raw\\sample_a\\btc-tx-480000-489999-aa  [more extracted files ...]

Record format (confirmed on sample):  TYPE BLOCK TX_INDEX FROM TO VALUE
  BTC-IN  : FROM = address, TO = tx hash
  BTC-OUT : FROM = tx hash, TO = address
Labels: label != 'white' -> known ransomware address (positive).
        everything else (white or not in BitcoinHeist) -> UNLABELED, not clean.
Writes data\\overlap_report.json next to the console output.
"""
import csv, json, os, re, sys, collections

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load_heist(path):
    fam_addrs = collections.defaultdict(set)   # family -> addresses
    years = collections.defaultdict(set)       # ransomware address -> years active
    white = set()
    rows = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            a, lab = row["address"], row["label"]
            if lab == "white":
                white.add(a)
            else:
                fam_addrs[lab].add(a)
                years[a].add(int(row["year"]))
    pos = set().union(*fam_addrs.values()) if fam_addrs else set()
    white -= pos  # an address listed as both counts as positive
    return rows, pos, white, fam_addrs, years


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    heist_path, tx_paths = sys.argv[1], sys.argv[2:]

    rows, pos, white, fam_addrs, years = load_heist(heist_path)
    print(f"BitcoinHeist rows (address-day): {rows:,}")
    print(f"  distinct ransomware addresses: {len(pos):,}")
    print(f"  distinct 'white' addresses:    {len(white):,}")
    print("  ransomware addresses by family:")
    for fam, s in sorted(fam_addrs.items(), key=lambda kv: -len(kv[1])):
        print(f"    {fam:<28}{len(s):>8,}")
    by_year = collections.Counter(y for ys in years.values() for y in ys)
    print("  ransomware addresses active per year (an address can count in several years):")
    for y in sorted(by_year):
        print(f"    {y}: {by_year[y]:,}")
    pos_1718 = sum(1 for ys in years.values() if ys & {2017, 2018})
    print(f"  ransomware addresses active in 2017-2018: {pos_1718:,}")

    seen = set()
    matched_pos, matched_white = set(), set()
    rows_total = rows_touching_pos = 0
    bad_addr_field = bad_other_field = malformed = 0
    pos_by_family = collections.Counter()
    fam_of = {a: fam for fam, s in fam_addrs.items() for a in s}
    prefix = collections.Counter()

    for p in tx_paths:
        with open(p, "r", encoding="ascii", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 6:
                    malformed += 1
                    continue
                t, _b, _i, frm, to, _v = parts
                rows_total += 1
                if t == "BTC-IN":
                    addr, other = frm, to
                elif t == "BTC-OUT":
                    addr, other = to, frm
                else:
                    malformed += 1
                    continue
                if HEX64.match(addr):
                    bad_addr_field += 1
                if not HEX64.match(other):
                    bad_other_field += 1
                if addr not in seen:
                    seen.add(addr)
                    prefix[addr[:3] if addr.startswith("bc1") else addr[:1]] += 1
                if addr in pos:
                    matched_pos.add(addr)
                    rows_touching_pos += 1
                elif addr in white:
                    matched_white.add(addr)

    for a in matched_pos:
        pos_by_family[fam_of[a]] += 1

    print(f"\nZenodo input files: {len(tx_paths)}  rows: {rows_total:,}  malformed: {malformed}")
    print(f"  address field that looks like a tx hash: {bad_addr_field:,} (expect 0)")
    print(f"  tx-hash field that is NOT 64-hex:        {bad_other_field:,} (expect 0)")
    print(f"  distinct addresses seen: {len(seen):,}   first-char/prefix mix: {dict(prefix.most_common(6))}")
    print(f"\nOVERLAP")
    print(f"  ransomware positives found in these files: {len(matched_pos):,}")
    print(f"  'white' addresses found:                   {len(matched_white):,}")
    print(f"  rows touching a positive:                  {rows_touching_pos:,}")
    if seen:
        print(f"  positives / distinct addresses: {len(matched_pos) / len(seen):.6%}")
    print(f"  positives / all BitcoinHeist ransomware addresses: {len(matched_pos) / max(len(pos), 1):.4%}")
    if pos_by_family:
        print("  positives by family:", dict(pos_by_family.most_common()))

    report = {
        "heist_rows": rows, "heist_ransomware_addresses": len(pos),
        "heist_white_addresses": len(white),
        "heist_ransomware_addresses_2017_2018": pos_1718,
        "files": [os.path.basename(p) for p in tx_paths],
        "zenodo_rows": rows_total, "zenodo_distinct_addresses": len(seen),
        "matched_positives": len(matched_pos), "matched_white": len(matched_white),
        "rows_touching_positive": rows_touching_pos,
        "positives_by_family": dict(pos_by_family),
    }
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "overlap_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved data\\overlap_report.json")


if __name__ == "__main__":
    main()
