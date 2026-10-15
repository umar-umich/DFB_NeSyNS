#!/usr/bin/env python
"""Every number quoted in the ChatGPT notes must trace to a table or a logged measurement.

The notes are instructions a human will act on, so a number left over from a reversed
decision is worse than a missing one: it reads as current. This catches that class.
Exits non-zero when a number has no source, so it can gate a commit.
"""
import re, sys, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPORT = os.path.join(ROOT, 'tbiom/reports/branch_fusion/PROPOSAL_TABLES_3.md')
NOTES = sorted(glob.glob(os.path.join(ROOT, 'tbiom/docs/NOTES_FOR_CHATGPT_*.md')))

# Measurements that are real but live outside the paper tables. Each needs a provenance
# line: where it was measured, so a reader can go and check it.
OFF_TABLE = {
    '0.8933': 'DF40 41-set per-branch, VD face swap',
    '0.9452': 'DF40 VD reenactment / also the 12-set ES-PCR6+ mean',
    '0.9684': 'DF40 LFS face swap',   '0.7855': 'DF40 LFS reenactment',
    '0.9829': 'DF40 STT face swap',   '0.7963': 'DF40 STT reenactment',
    '0.9894': 'DF40 CROSS-EXAM face swap', '0.9335': 'DF40 CROSS-EXAM reenactment',
    '0.0519': 'DF40 VD family gap',   '0.1828': 'DF40 LFS family gap',
    '0.1867': 'DF40 STT family gap',  '0.0559': 'DF40 CROSS-EXAM family gap',
    '0.0117': 'VD minus fusion on reenactment',
    '0.4750': 'branch correlation VD-LFS', '0.5280': 'branch correlation VD-STT',
    '0.7420': 'branch correlation LFS-STT',
    '0.8019': 'PE-Core block-16 frozen probe', '0.7356': 'PE-Core final-layer probe',
    '0.7311': 'CLIP mid-layer probe', '0.7478': 'CLIP last-layer probe',
    '0.9209': 'end-to-end from scratch, 5 matched sets',
    '0.9558': 'Stage 1 on the same 5 sets', '0.0349': 'e2e deficit',
    '0.8588': 'Stage 2 off, VALmix', '0.8596': 'Stage 2 one-sided, VALmix',
    '0.8598': 'Stage 2 original IBDC, VALmix', '0.0145': 'Stage 2 gain',
    '0.0002': 'one-sided minus original', '0.0350': 'IBDC deferral fraction',
    '0.8453': 'Stage 1 through the DISCERN_Ext path', '0.8897': 'Stage 1 cached-opinion path',
    '0.9420': 'plain PCR6+ AUROC', '0.1581': 'plain PCR6+ FPR@90', '0.0082': 'plain PCR6+ vacuity',
    '0.0032': 'ES-PCR6+ AUROC gain', '0.0074': 'ES-PCR6+ FPR gain', '0.0126': 'seed noise',
    '0.9682': 'STT reproduction check', '0.7383': 'VD CDFv2 (also in Table 1)',
    '0.7354': 'VD reproduction check', '0.0029': 'VD reproduction delta',
    '0.9644': 'DiCoME CDFv2 (also in Table 3)', '0.9729': 'released dicome-best.ckpt',
    '0.9679': 'DiCoME epoch 5', '0.9196': 'DiCoME 5-set matched support, old draft',
    '0.0546': 'KoDF adv-real FPR', '0.0323': 'KoDF ordinary-real FPR',
    '0.0397': 'KoDF FPR@90, v3.1 census', '0.3834': 'KoDF FPR@90, with adv',
    '0.9612': 'CDFv3 v3.1', '0.9614': 'CDFv3 widened', '0.9436': 'KoDF v3.1',
    '0.9494': 'KoDF without adv', '0.9112': 'KoDF with adv',
    '0.9514': '12-set mean without adv', '0.9482': '12-set mean with adv',
    '0.4128': 'adv-fake mean score',
}
# Differences and gaps the notes compute from table cells.
DERIVED = {'0.0008', '0.0017', '0.0016'}

def main():
    table_nums = set(re.findall(r'0\.\d{4}', open(REPORT).read()))
    known = table_nums | set(OFF_TABLE) | DERIVED
    bad = 0
    for f in NOTES:
        hits = []
        for i, line in enumerate(open(f), 1):
            for n in re.findall(r'0\.\d{4}', line):
                if n not in known:
                    hits.append((i, n, line.strip()[:100]))
        rel = os.path.relpath(f, ROOT)
        if hits:
            bad += len(hits)
            print(f'{rel}: {len(hits)} number(s) with no source')
            for i, n, l in hits:
                print(f'  L{i}  {n}  {l}')
        else:
            print(f'{rel}: OK')
    print()
    print('VERDICT: ' + ('COMPLETE, every number traces to a source'
                        if not bad else f'{bad} UNTRACEABLE NUMBER(S)'))
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(main())
