#!/usr/bin/env python3
"""
validate_suspects.py — LLM-Powered Web Validation Pipeline
===========================================================
For each top-scoring suspected fraud provider:
1. Enriches with CMS data (why flagged, key metrics)
2. Runs Brave API web searches (news, disciplinary actions, OIG press releases)
3. Feeds results to Claude for structured fraud analysis
4. Outputs JSON + HTML report

Usage:
    python validate_suspects.py                         # top 50
    python validate_suspects.py --limit 20              # top 20
    python validate_suspects.py --npi 1234567890        # single provider
    python validate_suspects.py --min-score 0.85        # score threshold
    python validate_suspects.py --dry-run               # no API calls
    python validate_suspects.py --output report.html    # custom output

Requirements:
    export ANTHROPIC_API_KEY=...
    export BRAVE_API_KEY=...      # get free key at https://brave.com/search/api/
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
SUSPECTS_CSV  = "/home/dataops/fraud-detector/physician_suspects.csv"
RESULTS_DIR   = "/home/dataops/fraud-detector/validation"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BRAVE_KEY     = os.environ.get("BRAVE_API_KEY", "")

SUBSCORE_LABELS = {
    'sub_spb':   'Services per beneficiary (extreme billing volume)',
    'sub_ppb':   'Payment per beneficiary (excess Medicare payments)',
    'sub_la':    'Long-acting opioid prescribing rate (potential pill mill)',
    'sub_cpb':   'Drug cost per beneficiary (extreme drug spending)',
    'sub_pay':   'Industry payments (pharma/device company relationships)',
    'sub_pecos': 'PECOS enrollment gap (billing without Medicare enrollment)',
}


def brave_search(query: str, count: int = 5) -> list[dict]:
    """Search Brave API, return list of {title, url, description}."""
    if not BRAVE_KEY:
        return [{"title": "[DRY RUN]", "url": "", "description": "No BRAVE_API_KEY set"}]
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY},
            params={"q": query, "count": count, "result_filter": "web"},
            timeout=10
        )
        r.raise_for_status()
        results = r.json().get("web", {}).get("results", [])
        return [{"title": x.get("title",""), "url": x.get("url",""),
                 "description": x.get("description","")} for x in results]
    except Exception as e:
        return [{"title": "Search failed", "url": "", "description": str(e)}]


def build_search_queries(provider: dict) -> list[str]:
    """Generate targeted search queries for a provider."""
    name = provider.get('display_name', '').strip()
    specialty = provider.get('specialty', '')
    state = provider.get('state', '')
    npi = provider.get('npi', '')

    queries = [
        f'"{name}" {specialty} Medicare fraud',
        f'"{name}" MD OIG exclusion fraud indictment',
        f'"{name}" {state} physician disciplinary medical board',
        f'NPI {npi} Medicare fraud investigation',
        f'"{name}" {specialty} settlement conviction healthcare',
    ]
    return queries


def run_llm_analysis(provider: dict, search_results: list[dict]) -> dict:
    """Feed provider data + search results to Claude for structured analysis."""
    if not ANTHROPIC_KEY:
        return {
            "verdict": "UNKNOWN (no API key)",
            "fraud_confidence": "N/A",
            "fraud_indicators": ["No ANTHROPIC_API_KEY set — dry run"],
            "legitimate_explanations": [],
            "key_findings": "Dry run — no LLM analysis performed",
            "recommended_action": "Set ANTHROPIC_API_KEY to enable",
            "sources": []
        }

    # Format CMS data context
    sub_descriptions = []
    for col, label in SUBSCORE_LABELS.items():
        score = float(provider.get(col, 0))
        if score > 0.7:
            sub_descriptions.append(f"  ⚠️  {label}: {score:.3f} (HIGH)")
        elif score > 0.5:
            sub_descriptions.append(f"  •  {label}: {score:.3f} (moderate)")

    cms_context = f"""
PROVIDER PROFILE:
- Name: {provider.get('display_name', 'Unknown')}
- NPI: {provider.get('npi')}
- Specialty: {provider.get('specialty', 'Unknown')}
- Location: {provider.get('city_state', 'Unknown')}
- Fraud Score: {float(provider.get('score', 0)):.3f} / 1.000
- Already on OIG LEIE exclusion list: {'YES' if provider.get('is_leie') else 'No'}

CMS DATA FLAGS (why our model flagged this provider):
{chr(10).join(sub_descriptions) if sub_descriptions else '  • Multiple moderate signals'}

RAW CMS METRICS:
- Total Medicare beneficiaries: {int(float(provider.get('total_benes', 0))):,}
- Total Medicare services: {int(float(provider.get('total_svc', 0))):,}
- Services per beneficiary: {float(provider.get('total_svc', 0))/max(float(provider.get('total_benes',1)),1):.1f}
- Total Medicare payments: ${float(provider.get('total_payment', 0)):,.0f}
- Open Payments (industry $): ${float(provider.get('total_payments_usd', 0)):,.0f}
- Driving fraud signal: {SUBSCORE_LABELS.get(provider.get('driving_sub',''), provider.get('driving_sub',''))}
"""

    # Format search results
    search_context = "\n\nWEB SEARCH RESULTS:\n"
    for i, result in enumerate(search_results, 1):
        search_context += f"\n[{i}] {result['title']}\n"
        search_context += f"    URL: {result['url']}\n"
        search_context += f"    {result['description'][:300]}\n"

    prompt = f"""You are a healthcare fraud investigator reviewing a provider flagged by a Medicare claims analysis system.

{cms_context}
{search_context}

Based on the CMS data patterns AND web search results, provide a structured analysis:

1. VERDICT: One of: LIKELY_FRAUD | POSSIBLE_FRAUD | LEGITIMATE | INSUFFICIENT_DATA
2. FRAUD_CONFIDENCE: HIGH | MEDIUM | LOW
3. FRAUD_INDICATORS: List any specific fraud red flags found (billing patterns, news, legal actions)
4. LEGITIMATE_EXPLANATIONS: List any legitimate explanations for the high scores (specialty type, academic center, high-complexity patient population)
5. KEY_FINDINGS: 2-3 sentence summary of most important finding
6. RECOMMENDED_ACTION: What a fraud investigator should do next (e.g., "Request claims audit", "No action needed", "Refer to OIG")
7. SOURCES: List any specific URLs that contained relevant information

Be precise and evidence-based. Distinguish between statistical anomalies (which may be legitimate) and actual fraud evidence.

Respond in valid JSON only, no markdown."""

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        r.raise_for_status()
        content = r.json()["content"][0]["text"].strip()
        # Strip markdown if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        return {
            "verdict": "ERROR",
            "fraud_confidence": "N/A",
            "fraud_indicators": [str(e)],
            "legitimate_explanations": [],
            "key_findings": f"LLM analysis failed: {e}",
            "recommended_action": "Retry",
            "sources": []
        }


def generate_html_report(results: list[dict], output_path: str):
    """Generate a styled HTML report of validation results."""
    verdict_colors = {
        'LIKELY_FRAUD':       ('#fff5f5', '#c53030', '🚨'),
        'POSSIBLE_FRAUD':     ('#fffff0', '#d69e2e', '⚠️'),
        'LEGITIMATE':         ('#f0fff4', '#276749', '✅'),
        'INSUFFICIENT_DATA':  ('#ebf8ff', '#2b6cb0', '❓'),
        'UNKNOWN (no API key)': ('#f7fafc', '#4a5568', '—'),
        'ERROR':              ('#fff5f5', '#c53030', '💥'),
    }

    rows_html = ""
    for r in results:
        p = r['provider']
        a = r['analysis']
        verdict = a.get('verdict', 'INSUFFICIENT_DATA')
        bg, color, icon = verdict_colors.get(verdict, ('#f7fafc','#4a5568','?'))

        # Subscore bars
        sub_bars = ""
        for col, label in SUBSCORE_LABELS.items():
            val = float(p.get(col, 0))
            bar_color = '#c53030' if val > 0.75 else '#d69e2e' if val > 0.5 else '#276749'
            short_label = label.split('(')[0].strip()
            sub_bars += f"""
            <div style="margin:4px 0;font-size:0.8rem">
              <span style="display:inline-block;width:200px;color:#4a5568">{short_label}</span>
              <span style="display:inline-block;width:{int(val*120)}px;height:10px;
                background:{bar_color};border-radius:3px;vertical-align:middle"></span>
              <span style="margin-left:4px;font-family:monospace">{val:.3f}</span>
            </div>"""

        fraud_items = ''.join(f'<li>{x}</li>' for x in a.get('fraud_indicators',[]))
        legit_items = ''.join(f'<li>{x}</li>' for x in a.get('legitimate_explanations',[]))

        rows_html += f"""
        <div style="background:{bg};border:1px solid {color};border-left:5px solid {color};
                    border-radius:6px;margin:1.5rem 0;padding:1.25rem">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <h3 style="color:{color};margin:0">{icon} {p.get('display_name','Unknown')}</h3>
              <div style="color:#4a5568;font-size:0.9rem;margin-top:0.25rem">
                NPI: {p.get('npi')} &nbsp;|&nbsp; {str(p.get('specialty','')).title()[:40]}
                &nbsp;|&nbsp; {p.get('city_state','')}
              </div>
            </div>
            <div style="text-align:right">
              <div style="font-size:1.6rem;font-weight:bold;color:{color}">{float(p.get('score',0)):.3f}</div>
              <div style="font-size:0.75rem;color:#4a5568">Fraud Score</div>
              <div style="margin-top:0.25rem"><span style="background:{color};color:white;
                padding:2px 8px;border-radius:12px;font-size:0.8rem">{verdict}</span></div>
            </div>
          </div>

          <div style="margin-top:1rem;display:grid;grid-template-columns:1fr 1fr;gap:1rem">
            <div>
              <strong style="font-size:0.85rem;color:#1a365d">CMS Signal Scores</strong>
              {sub_bars}
              <div style="margin-top:0.5rem;font-size:0.8rem;color:#4a5568">
                Medicare: {int(float(p.get('total_benes',0))):,} benes &nbsp;|&nbsp;
                {int(float(p.get('total_svc',0))):,} services &nbsp;|&nbsp;
                ${float(p.get('total_payment',0)):,.0f} paid
              </div>
            </div>
            <div>
              <strong style="font-size:0.85rem;color:#1a365d">LLM Analysis
                ({a.get('fraud_confidence','?')} confidence)</strong>
              <p style="font-size:0.85rem;margin:0.5rem 0">{a.get('key_findings','')}</p>
              {f'<div><strong style="font-size:0.8rem;color:#c53030">⚠️ Fraud Indicators:</strong><ul style="font-size:0.8rem;margin:0.25rem 0 0 1rem">{fraud_items}</ul></div>' if fraud_items else ''}
              {f'<div><strong style="font-size:0.8rem;color:#276749">✓ Legitimate Explanations:</strong><ul style="font-size:0.8rem;margin:0.25rem 0 0 1rem">{legit_items}</ul></div>' if legit_items else ''}
              <div style="margin-top:0.5rem;font-size:0.8rem;background:white;
                          padding:0.5rem;border-radius:4px;border:1px solid {color}">
                <strong>Recommended Action:</strong> {a.get('recommended_action','')}
              </div>
            </div>
          </div>
        </div>"""

    summary_counts = {}
    for r in results:
        v = r['analysis'].get('verdict','UNKNOWN')
        summary_counts[v] = summary_counts.get(v, 0) + 1
    summary_html = ' &nbsp;|&nbsp; '.join(
        f'<strong>{v}:</strong> {n}' for v,n in sorted(summary_counts.items())
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Fraud Validation Report — {datetime.now().strftime('%Y-%m-%d')}</title>
  <style>
    body {{font-family:Georgia,serif;background:#f7fafc;color:#2d3748;margin:0;padding:0}}
    .header {{background:#1a365d;color:white;padding:1.5rem}}
    .header h1 {{margin:0;font-size:1.6rem}}
    .container {{max-width:1000px;margin:0 auto;padding:1.5rem}}
    ul {{margin:0.25rem 0;padding-left:1.25rem}}
  </style>
</head>
<body>
  <div class="header">
    <h1>🔬 Healthcare Fraud Validation Report</h1>
    <div style="opacity:0.85;margin-top:0.5rem">
      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
      Providers analyzed: {len(results)} &nbsp;|&nbsp; {summary_html}
    </div>
  </div>
  <div class="container">
    <div style="background:white;border:1px solid #e2e8f0;border-radius:6px;
                padding:1rem;margin:1rem 0;font-size:0.9rem">
      <strong>How this works:</strong> Each provider is scored by our CMS fraud detection model,
      then web-searched and analyzed by Claude AI to distinguish billing anomalies (potentially
      legitimate) from actual fraud indicators (news coverage, legal actions, OIG press releases,
      state medical board actions). This is a <em>research tool</em>, not a legal determination.
    </div>
    {rows_html}
  </div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"[report] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--npi', type=str, default=None)
    parser.add_argument('--min-score', type=float, default=0.0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', default=None)
    parser.add_argument('--search-delay', type=float, default=1.0,
                        help='Seconds between Brave API calls')
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Load suspects
    if not Path(SUSPECTS_CSV).exists():
        print(f"ERROR: {SUSPECTS_CSV} not found — run feature_importance.py first")
        sys.exit(1)

    suspects = pd.read_csv(SUSPECTS_CSV, dtype={'npi': str})

    if args.npi:
        suspects = suspects[suspects['npi'] == args.npi]
    if args.min_score > 0:
        suspects = suspects[suspects['score'] >= args.min_score]
    suspects = suspects.head(args.limit)

    print(f"=== LLM FRAUD VALIDATION PIPELINE ===")
    print(f"Providers to analyze: {len(suspects)}")
    print(f"Brave API: {'✓' if BRAVE_KEY else '✗ missing (set BRAVE_API_KEY)'}")
    print(f"Anthropic API: {'✓' if ANTHROPIC_KEY else '✗ missing (set ANTHROPIC_API_KEY)'}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    results = []
    for i, (_, provider) in enumerate(suspects.iterrows(), 1):
        name = provider.get('display_name', f"NPI {provider['npi']}")
        print(f"[{i}/{len(suspects)}] {name} (score: {float(provider['score']):.3f})")

        if args.dry_run:
            search_results = [{"title": "Dry run", "url": "", "description": ""}]
        else:
            # Run multiple targeted searches
            queries = build_search_queries(provider.to_dict())
            search_results = []
            for q in queries[:3]:  # top 3 queries to limit API usage
                results_q = brave_search(q, count=3)
                search_results.extend(results_q)
                time.sleep(args.search_delay)
            # Deduplicate by URL
            seen = set()
            deduped = []
            for r in search_results:
                if r['url'] not in seen:
                    seen.add(r['url'])
                    deduped.append(r)
            search_results = deduped[:10]  # max 10 unique results per provider

        analysis = run_llm_analysis(provider.to_dict(), search_results)

        result = {
            'provider': provider.to_dict(),
            'search_results': search_results,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)

        verdict = analysis.get('verdict', '?')
        confidence = analysis.get('fraud_confidence', '?')
        print(f"         → {verdict} ({confidence} confidence)")

        # Save intermediate JSON after each provider
        json_path = f"{RESULTS_DIR}/validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Small delay between providers
        if not args.dry_run and i < len(suspects):
            time.sleep(0.5)

    # Generate HTML report
    ts = datetime.now().strftime('%Y-%m-%d')
    out_path = args.output or f"{RESULTS_DIR}/validation_report_{ts}.html"
    generate_html_report(results, out_path)

    # Summary
    verdicts = [r['analysis'].get('verdict','?') for r in results]
    print(f"\n=== SUMMARY ===")
    for v in sorted(set(verdicts)):
        print(f"  {v}: {verdicts.count(v)}")
    print(f"\nHTML report: {out_path}")
    print(f"JSON data:   {RESULTS_DIR}/validation_results.json")


if __name__ == '__main__':
    main()
