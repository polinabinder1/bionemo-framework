import React, { useEffect, useMemo, useState } from 'react'
import FeatureDetailPage from './FeatureDetailPage'

// One-page snapshot of the NS.1 SAE evaluation state. Mix of real numbers
// (training quality from the layer-22 winner config — these are NOT synthetic)
// and synthetic placeholders for downstream metrics that need real eval to land.
// Each tile marks its provenance so the page can't be misread as fully real.

const COLORS = {
  pass: '#5a9c3f',
  fail: '#c34a4a',
  bar: '#4E79A7',
  baseline: '#9C755F',
  sae: '#76B7B2',
  brca1: '#F28E2B',
  exon: '#B07AA1',
  pending: '#aaa',
}


export default function SAESummary() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/sae_qc_summary.json').then((r) => r.json()).then(setData).catch((e) => setError(e.message))
  }, [])

  if (error) return <div style={styles.error}>Failed to load sae_qc_summary.json: {error}</div>
  if (!data) return <div style={styles.loading}>Loading SAE summary…</div>

  return (
    <div style={styles.container}>
      <div style={styles.banner}>
        <b>PARTIAL MOCKUP</b> — training-quality numbers (Section 2) are real, taken from the
        layer-22 winner sweep that finished today. Sections 3–7 are synthetic placeholders for
        evaluation work that hasn't run yet. The methodology checklist (Section 8) is{' '}
        <i>planned</i> work, not completed.
      </div>

      <Section1Metadata meta={data.sae_metadata} />
      <Section2TrainingTiles tq={data.training_quality} />
      <Section3SixProperty sp={data.six_property_qc} />
      <Section4BioQC bq={data.bio_qc} />
      <Section5DownstreamProbes dp={data.downstream_probes} />
      <Section7CausalValidation cv={data.causal_validation} />
      <Section9TopFeatures features={data.top_features} />
    </div>
  )
}


function Section1Metadata({ meta }) {
  const parts = [
    meta.model,
    `layer ${meta.layer}`,
    `width ${meta.width.toLocaleString()}`,
    `k=${meta.k}`,
    `${meta.training_tokens}`,
  ]
  return (
    <div style={styles.metaBar}>
      {parts.map((p, i) => (
        <span key={i} style={styles.metaItem}>
          {i > 0 && <span style={styles.metaDot}>·</span>}
          {p}
        </span>
      ))}
    </div>
  )
}


function Section2TrainingTiles({ tq }) {
  const tiles = [
    {
      name: 'Dead latents',
      value: `${tq.dead_latent_pct.value.toFixed(2)}%`,
      target: `target <${tq.dead_latent_pct.target_max}%`,
      passing: tq.dead_latent_pct.passing,
      real: tq.dead_latent_pct.real,
      note: 'fraction of latents never firing on training data',
    },
    {
      name: 'FVU',
      value: tq.fvu.value.toFixed(3),
      target: `target <${tq.fvu.target_max}`,
      passing: tq.fvu.passing,
      real: tq.fvu.real,
      note: 'fraction of variance unexplained',
    },
    {
      name: 'L0 sparsity',
      value: `${tq.l0.value} / ${tq.l0.width.toLocaleString()}`,
      target: 'k by construction',
      passing: tq.l0.passing,
      real: tq.l0.real,
      note: 'avg # nonzero latents per token',
    },
    {
      name: 'Loss recovered',
      value: tq.loss_recovered.value.toFixed(2),
      target: `target >${tq.loss_recovered.target_min}`,
      passing: tq.loss_recovered.passing,
      real: tq.loss_recovered.real,
      note: 'CE under SAE substitution',
    },
  ]
  return (
    <SectionWrap title="Training quality" subtitle="Layer-22 winner config (exp=8, auxk=512, lr=3e-4)">
      <div style={styles.tilesRow}>
        {tiles.map((t) => (
          <div key={t.name} style={styles.tile}>
            <div style={styles.tileHeader}>
              <span style={styles.tileName}>{t.name}</span>
              {t.real ? (
                <span style={styles.realBadge}>REAL</span>
              ) : (
                <span style={styles.synthBadge}>SYNTH</span>
              )}
            </div>
            <div style={styles.tileValue}>{t.value}</div>
            <div style={styles.tileTarget}>
              <span style={{ color: t.passing ? COLORS.pass : COLORS.fail, marginRight: '4px' }}>
                {t.passing ? '✓' : '✗'}
              </span>
              {t.target}
            </div>
            <div style={styles.tileNote}>{t.note}</div>
          </div>
        ))}
      </div>
    </SectionWrap>
  )
}


function Section3SixProperty({ sp }) {
  return (
    <SectionWrap
      title="Six-property QC (Bricken et al. 2023)"
      subtitle={`Tested on top ${sp.n_features_tested} features by auto-interp score`}
      synthetic
    >
      <div style={styles.barList}>
        {sp.properties.map((p) => (
          <div key={p.name} style={styles.barRow} title={p.description}>
            <span style={styles.barLabel}>{p.name}</span>
            <div style={styles.barTrack}>
              <div
                style={{
                  ...styles.barFill,
                  width: `${p.pass_rate * 100}%`,
                  background: COLORS.bar,
                }}
              />
            </div>
            <span style={styles.barValue}>{(p.pass_rate * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
      <div style={styles.subnote}>
        <b>{sp.n_headline_features}</b> features pass {sp.headline_threshold} — these are the
        "headline" features the dashboard surfaces by default.
      </div>
    </SectionWrap>
  )
}


function Section4BioQC({ bq }) {
  const maxN = Math.max(...bq.by_database.map((d) => d.n_matching))
  return (
    <SectionWrap
      title="Bio QC — annotation database matches"
      subtitle={`Features matching known biological annotations at F1 > ${bq.f1_threshold}`}
      synthetic
    >
      <div style={styles.bioQCRow}>
        <div style={styles.bioQCBigNum}>
          <div style={styles.bigNum}>{bq.n_total_matching}</div>
          <div style={styles.bigNumLabel}>matching features</div>
        </div>
        <div style={{ ...styles.barList, flex: 1 }}>
          {bq.by_database.map((d) => (
            <div key={d.database} style={styles.barRow}>
              <span style={styles.barLabel}>{d.database}</span>
              <div style={styles.barTrack}>
                <div
                  style={{
                    ...styles.barFill,
                    width: `${(d.n_matching / maxN) * 100}%`,
                    background: COLORS.sae,
                  }}
                />
              </div>
              <span style={styles.barValue}>{d.n_matching}</span>
            </div>
          ))}
        </div>
      </div>
    </SectionWrap>
  )
}


function Section5DownstreamProbes({ dp }) {
  return (
    <SectionWrap
      title="Multivariate downstream probes"
      subtitle="AUROC on real biological tasks; lower y-bound = 0.80 to highlight differences"
      synthetic
    >
      <div style={styles.probesRow}>
        <ProbeChart probe={dp.brca1} color={COLORS.brca1} />
        <ProbeChart probe={dp.exon} color={COLORS.exon} />
      </div>
      <div style={styles.subnote}>
        SAE preserves task-relevant signal within 3–4 AUROC points of the dense Evo 2 baseline.
        Note: y-axis truncated to 0.80–1.00; absolute differences are small, relative differences
        are real signal in this regime.
      </div>
    </SectionWrap>
  )
}


function ProbeChart({ probe, color }) {
  const yMin = 0.80
  const yMax = 1.00
  return (
    <div style={styles.probeCard}>
      <div style={styles.probeTitle}>{probe.task}</div>
      <div style={styles.probeDataset}>{probe.dataset}</div>
      <div style={styles.probeBars}>
        {probe.results.map((r) => {
          const h = ((r.auroc - yMin) / (yMax - yMin)) * 100
          return (
            <div key={r.input} style={styles.probeBarWrap} title={`${r.input}: AUROC ${r.auroc.toFixed(3)}`}>
              <div style={styles.probeBarTrack}>
                <div
                  style={{
                    ...styles.probeBarFill,
                    height: `${h}%`,
                    background: r.is_baseline ? COLORS.baseline : color,
                  }}
                />
              </div>
              <div style={styles.probeBarVal}>{r.auroc.toFixed(2)}</div>
              <div style={styles.probeBarLabel}>{r.input}</div>
            </div>
          )
        })}
      </div>
      <div style={styles.probeAxis}>y-axis: AUROC 0.80–1.00</div>
    </div>
  )
}


function Section6SizeExploration({ se }) {
  const maxInterp = Math.max(...se.data.map((d) => d.interpretable_count))
  return (
    <SectionWrap
      title="Size exploration"
      subtitle={se.in_progress ? 'Larger sweep in progress — only 2 points so far' : ''}
      synthetic
    >
      <div style={styles.sizeTable}>
        <div style={styles.sizeHeader}>
          <span>Expansion</span>
          <span>Width</span>
          <span>Dead %</span>
          <span>FVU</span>
          <span style={styles.sizeBarHeader}>Interpretable features (F1 &gt; 0.2)</span>
        </div>
        {se.data.map((d) => (
          <div key={d.expansion} style={styles.sizeRow}>
            <span style={styles.sizeCell}>×{d.expansion}</span>
            <span style={styles.sizeCell}>{d.width.toLocaleString()}</span>
            <span style={styles.sizeCell}>{d.dead_pct.toFixed(2)}%</span>
            <span style={styles.sizeCell}>{d.fvu.toFixed(3)}</span>
            <div style={styles.sizeBarWrap}>
              <div style={styles.sizeBarTrack}>
                <div
                  style={{
                    ...styles.sizeBarFill,
                    width: `${(d.interpretable_count / maxInterp) * 100}%`,
                    background: d.real ? COLORS.bar : COLORS.pending,
                  }}
                />
              </div>
              <span style={styles.sizeBarVal}>
                {d.interpretable_count}{!d.real && <span style={styles.synthInline}> (projected)</span>}
              </span>
            </div>
          </div>
        ))}
      </div>
    </SectionWrap>
  )
}


function Section7CausalValidation({ cv }) {
  return (
    <SectionWrap title="Causal validation via steering" synthetic>
      <div style={styles.causalRow}>
        <div style={styles.causalBigNum}>
          <div style={styles.bigNum}>{cv.n_validated}</div>
          <div style={styles.bigNumLabel}>features causally validated</div>
        </div>
        <div style={styles.causalText}>
          <div style={styles.callout}>
            <b>Headline result:</b> {cv.headline_narrative}
          </div>
          <a
            href="#preview"
            onClick={(e) => {
              e.preventDefault()
              // The steering explorer is a sibling tab in Preview.jsx;
              // a parent-level tab-switch handler would be plumbed here in
              // a future PR. For now this is a no-op link so the affordance
              // exists in the mockup.
            }}
            style={styles.causalLink}
          >
            → Explore steering interactively (Steering tab)
          </a>
        </div>
      </div>
    </SectionWrap>
  )
}


function Section8MethodologyChecklist({ mc }) {
  return (
    <SectionWrap
      title="Methodology vs Goodfire (Brixi et al. 2025) and Decode-gLM (Hutchinson et al. 2025)"
      subtitle="Planned methodology — checkmarks indicate intent, not all completed work"
    >
      <div style={styles.checklistCols}>
        <ChecklistColumn title="Training / architecture" items={mc.training} />
        <ChecklistColumn title="Evaluation / QC" items={mc.evaluation} />
      </div>
    </SectionWrap>
  )
}


function ChecklistColumn({ title, items }) {
  return (
    <div style={styles.checklistCol}>
      <div style={styles.checklistTitle}>{title}</div>
      {items.map((it, i) => (
        <div key={i} style={styles.checkRow}>
          <span style={{ color: it.done ? COLORS.pass : COLORS.pending, fontFamily: 'monospace' }}>
            {it.done ? '[✓]' : '[ ]'}
          </span>
          <span style={{ ...styles.checkLabel, color: it.done ? 'var(--text)' : 'var(--text-muted, #888)' }}>
            {it.label}
          </span>
          {it.note && <span style={styles.checkNote}>— {it.note}</span>}
        </div>
      ))}
    </div>
  )
}


const QC_PROPERTY_NAMES = ['Specificity', 'Sensitivity', 'Reconstruction', 'Binned', 'Decomposability', 'Causal']

function Section9TopFeatures({ features }) {
  // Clicking a row opens FeatureDetailPage as a modal overlay — same pattern
  // the main dashboard uses from FeatureCard's "Full analysis" button. Examples
  // will be empty (we don't ship per-feature top-activator windows in this
  // mockup); the detail page handles that gracefully with a "no examples" state.
  const [openFeature, setOpenFeature] = useState(null)

  return (
    <SectionWrap
      title="Top features in the catalog"
      subtitle="Sorted by combined quality score (auto-interp × annotation F1)"
      synthetic
    >
      {openFeature && (
        <FeatureDetailPage
          feature={{
            feature_id: openFeature.feature_id,
            description: openFeature.label,
            label: openFeature.label,
            activation_freq: openFeature.activation_freq,
            max_activation: openFeature.max_activation,
          }}
          examples={[]}
          onClose={() => setOpenFeature(null)}
        />
      )}
      <div style={styles.tableScroll}>
        <table style={styles.featureTable}>
          <thead>
            <tr style={styles.featureTableHeader}>
              <th style={{ ...styles.featureCell, ...styles.colId }}>ID</th>
              <th style={{ ...styles.featureCell, ...styles.colLabel }}>Label</th>
              <th style={{ ...styles.featureCell, ...styles.colContext }}>Top context</th>
              <th style={{ ...styles.featureCell, ...styles.colDomain }}>Dom.</th>
              <th style={{ ...styles.featureCell, ...styles.colF1 }}>Best F1</th>
              <th style={{ ...styles.featureCell, ...styles.colAutoInterp }}>Auto-int</th>
              <th style={{ ...styles.featureCell, ...styles.colMaxAct }}>Max act</th>
              <th style={{ ...styles.featureCell, ...styles.colFreq }}>Freq</th>
              <th style={{ ...styles.featureCell, ...styles.colQCScore }} title="Specificity: feature fires on identifiable pattern">Spec</th>
              <th style={{ ...styles.featureCell, ...styles.colQCScore }} title="Sensitivity: fires whenever pattern appears">Sens</th>
              <th style={{ ...styles.featureCell, ...styles.colQCScore }} title="Reconstruction faithfulness: ablation hurts loss">Recon</th>
              <th style={{ ...styles.featureCell, ...styles.colQCScore }} title="Binned activation: clear on/off">Binned</th>
              <th style={{ ...styles.featureCell, ...styles.colQCScore }} title="Decomposability: concentrated in top activators">Decomp</th>
              <th style={{ ...styles.featureCell, ...styles.colQCScore }} title="Causal effect: clamping changes output">Causal</th>
            </tr>
          </thead>
          <tbody>
            {(features || []).length === 0 ? (
              <tr><td colSpan={9} style={{ padding: '20px', textAlign: 'center', color: '#c66' }}>
                DEBUG: no features in array (loaded {features ? features.length : 'undefined'})
              </td></tr>
            ) : (
              features.map((f) => (
                <FeatureRow key={f.feature_id} feature={f} onClick={() => setOpenFeature(f)} />
              ))
            )}
          </tbody>
        </table>
      </div>
      <div style={styles.subnote}>
        Showing top 15 of 142 interpretable features (F1 &gt; 0.2).{' '}
        <a href="#preview" style={styles.catalogLink}>View full catalog →</a>
      </div>
    </SectionWrap>
  )
}


const DOMAIN_BADGE = {
  prok: { label: 'prok', bg: '#fff3cd', color: '#856404' },
  euk: { label: 'euk', bg: '#e7f3ff', color: '#1f4d80' },
  both: { label: 'both', bg: '#e8f5e9', color: '#2e7d32' },
}

function FeatureRow({ feature, onClick }) {
  const [hover, setHover] = useState(false)
  const dBadge = DOMAIN_BADGE[feature.domain] || DOMAIN_BADGE.both
  return (
    <tr
      style={{ ...styles.featureRow, background: hover ? 'var(--bg-card-expanded, #f5fafa)' : 'transparent' }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={onClick}
    >
      <td style={{ ...styles.featureCell, ...styles.colId, fontFamily: 'monospace' }}>
        #{feature.feature_id}
      </td>
      <td style={{ ...styles.featureCell, ...styles.colLabel }}>{feature.label}</td>
      <td style={{ ...styles.featureCell, ...styles.colContext }}>{feature.top_context}</td>
      <td style={{ ...styles.featureCell, ...styles.colDomain }}>
        <span style={{ ...styles.domainBadge, background: dBadge.bg, color: dBadge.color }}>
          {dBadge.label}
        </span>
      </td>
      <td style={{ ...styles.featureCell, ...styles.colF1 }}>
        <div style={styles.f1Cell}>
          <span style={styles.f1Value}>{feature.best_f1.toFixed(2)}</span>
          <span style={styles.f1Db}>{feature.best_f1_database}</span>
        </div>
      </td>
      <td style={{ ...styles.featureCell, ...styles.colAutoInterp }}>
        {feature.auto_interp_score.toFixed(2)}
      </td>
      <td style={{ ...styles.featureCell, ...styles.colMaxAct }}>
        {feature.max_activation.toFixed(1)}
      </td>
      <td style={{ ...styles.featureCell, ...styles.colFreq }}>
        {feature.activation_freq.toFixed(3)}
      </td>
      <QCScoreCell value={feature.qc_scores?.specificity} />
      <QCScoreCell value={feature.qc_scores?.sensitivity} />
      <QCScoreCell value={feature.qc_scores?.reconstruction} />
      <QCScoreCell value={feature.qc_scores?.binned} />
      <QCScoreCell value={feature.qc_scores?.decomposability} />
      <QCScoreCell value={feature.qc_scores?.causal} />
    </tr>
  )
}


// Each QC property cell: shows the numeric score color-coded (green if >=0.5,
// red if <0.5, gray if missing/not-tested). Passes the same threshold the
// Section 3 "≥5 of 6 properties" headline criterion uses.
const QC_THRESHOLD = 0.5

function QCScoreCell({ value }) {
  if (value == null) {
    return (
      <td style={{ ...styles.featureCell, ...styles.colQCScore, color: '#bbb' }}>
        —
      </td>
    )
  }
  const passing = value >= QC_THRESHOLD
  return (
    <td
      style={{
        ...styles.featureCell,
        ...styles.colQCScore,
        color: passing ? COLORS.pass : COLORS.fail,
        fontFamily: 'monospace',
        fontWeight: 600,
      }}
    >
      {value.toFixed(2)}
    </td>
  )
}


function QCDots({ dots }) {
  const titleStr = dots
    .map((d, i) => `${QC_PROPERTY_NAMES[i]}: ${d === true ? 'pass' : d === false ? 'fail' : 'not tested'}`)
    .join(' / ')
  return (
    <div style={styles.dotsRow} title={titleStr}>
      {dots.map((d, i) => {
        let bg, border
        if (d === true) {
          bg = COLORS.pass
          border = COLORS.pass
        } else if (d === false) {
          bg = 'transparent'
          border = COLORS.fail
        } else {
          bg = '#d8d8d8'
          border = '#d8d8d8'
        }
        return (
          <span
            key={i}
            style={{
              ...styles.dot,
              background: bg,
              border: `1.5px solid ${border}`,
            }}
          />
        )
      })}
    </div>
  )
}


function SectionWrap({ title, subtitle, synthetic, children }) {
  return (
    <div style={styles.section}>
      <div style={styles.sectionHeader}>
        <span style={styles.sectionTitle}>{title}</span>
        {synthetic && <span style={styles.synthBadge}>SYNTHETIC</span>}
      </div>
      {subtitle && <div style={styles.sectionSubtitle}>{subtitle}</div>}
      <div style={styles.sectionBody}>{children}</div>
    </div>
  )
}


const styles = {
  container: { fontFamily: 'system-ui, sans-serif', color: 'var(--text, #222)' },
  banner: {
    background: '#fff3cd', border: '1px solid #ffeeba', color: '#856404',
    padding: '8px 14px', borderRadius: '4px', fontSize: '11px', marginBottom: '14px',
    lineHeight: '1.5',
  },
  metaBar: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '10px 16px', marginBottom: '14px',
    fontSize: '12px', color: 'var(--text-secondary, #555)',
    fontFamily: 'monospace',
  },
  metaItem: { whiteSpace: 'nowrap' },
  metaDot: { margin: '0 8px', color: 'var(--text-muted, #aaa)' },
  section: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '12px 16px', marginBottom: '14px',
  },
  sectionHeader: {
    display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px',
  },
  sectionTitle: { fontSize: '13px', fontWeight: 600, color: 'var(--text-heading, #222)' },
  sectionSubtitle: { fontSize: '11px', color: 'var(--text-secondary, #666)', marginBottom: '10px' },
  sectionBody: { marginTop: '6px' },
  realBadge: {
    fontSize: '9px', fontWeight: 700, color: COLORS.pass,
    background: '#eef9ea', border: `1px solid ${COLORS.pass}`,
    borderRadius: '3px', padding: '1px 5px',
  },
  synthBadge: {
    fontSize: '9px', fontWeight: 700, color: '#856404',
    background: '#fff3cd', border: '1px solid #ffeeba',
    borderRadius: '3px', padding: '1px 5px',
  },
  synthInline: { fontSize: '10px', fontStyle: 'italic', color: 'var(--text-muted, #888)' },

  tilesRow: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' },
  tile: {
    background: 'var(--bg-card-expanded, #fafafa)',
    border: '1px solid var(--border-light, #eee)', borderRadius: '4px',
    padding: '10px 12px',
  },
  tileHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  tileName: { fontSize: '11px', color: 'var(--text-secondary, #555)', fontWeight: 500 },
  tileValue: { fontSize: '22px', fontWeight: 700, color: 'var(--text-heading, #222)', margin: '4px 0' },
  tileTarget: { fontSize: '11px', color: 'var(--text-secondary, #555)' },
  tileNote: { fontSize: '10px', color: 'var(--text-muted, #888)', marginTop: '4px', fontStyle: 'italic' },

  barList: { display: 'flex', flexDirection: 'column', gap: '6px' },
  barRow: { display: 'grid', gridTemplateColumns: '180px 1fr 50px', alignItems: 'center', gap: '8px' },
  barLabel: { fontSize: '11px', color: 'var(--text-secondary, #555)' },
  barTrack: { height: '14px', background: '#f0f0f0', borderRadius: '3px', overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: '3px' },
  barValue: { fontFamily: 'monospace', fontSize: '11px', textAlign: 'right', fontWeight: 600 },
  subnote: {
    marginTop: '10px', fontSize: '11px', color: 'var(--text-secondary, #555)', lineHeight: '1.4',
  },

  bioQCRow: { display: 'flex', alignItems: 'center', gap: '20px' },
  bioQCBigNum: {
    textAlign: 'center', padding: '10px 20px',
    background: 'var(--bg-card-expanded, #fafafa)',
    border: '1px solid var(--border-light, #eee)', borderRadius: '4px',
  },
  bigNum: { fontSize: '32px', fontWeight: 700, color: COLORS.sae, lineHeight: 1 },
  bigNumLabel: { fontSize: '10px', color: 'var(--text-secondary, #555)', marginTop: '4px' },

  probesRow: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' },
  probeCard: {
    background: 'var(--bg-card-expanded, #fafafa)',
    border: '1px solid var(--border-light, #eee)', borderRadius: '4px',
    padding: '10px 12px',
  },
  probeTitle: { fontSize: '12px', fontWeight: 600, color: 'var(--text-heading, #222)' },
  probeDataset: { fontSize: '10px', color: 'var(--text-muted, #888)', marginBottom: '8px' },
  probeBars: { display: 'flex', gap: '10px', alignItems: 'flex-end', height: '140px', marginBottom: '4px' },
  probeBarWrap: { flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' },
  probeBarTrack: { width: '100%', flex: 1, display: 'flex', alignItems: 'flex-end', background: '#f5f5f5', borderRadius: '3px 3px 0 0' },
  probeBarFill: { width: '100%', borderRadius: '3px 3px 0 0', transition: 'height 0.2s' },
  probeBarVal: { fontFamily: 'monospace', fontSize: '11px', fontWeight: 700, marginTop: '4px' },
  probeBarLabel: { fontSize: '10px', color: 'var(--text-secondary, #555)', textAlign: 'center', marginTop: '2px', lineHeight: '1.2' },
  probeAxis: { fontSize: '9px', color: 'var(--text-muted, #aaa)', fontStyle: 'italic', textAlign: 'right' },

  sizeTable: { display: 'flex', flexDirection: 'column', gap: '4px' },
  sizeHeader: {
    display: 'grid',
    gridTemplateColumns: '70px 80px 70px 70px 1fr',
    gap: '8px', fontSize: '10px', textTransform: 'uppercase',
    color: 'var(--text-tertiary, #888)', fontWeight: 600, padding: '4px 0',
    borderBottom: '1px solid var(--border-light, #eee)',
  },
  sizeBarHeader: { paddingLeft: '8px' },
  sizeRow: {
    display: 'grid',
    gridTemplateColumns: '70px 80px 70px 70px 1fr',
    gap: '8px', alignItems: 'center', padding: '6px 0',
    fontSize: '12px',
  },
  sizeCell: { fontFamily: 'monospace' },
  sizeBarWrap: { display: 'flex', alignItems: 'center', gap: '8px' },
  sizeBarTrack: { flex: 1, height: '14px', background: '#f0f0f0', borderRadius: '3px', overflow: 'hidden' },
  sizeBarFill: { height: '100%', borderRadius: '3px' },
  sizeBarVal: { fontFamily: 'monospace', fontSize: '11px', fontWeight: 600, minWidth: '120px' },

  causalRow: { display: 'flex', alignItems: 'center', gap: '20px' },
  causalBigNum: {
    textAlign: 'center', padding: '10px 20px',
    background: 'var(--bg-card-expanded, #fafafa)',
    border: '1px solid var(--border-light, #eee)', borderRadius: '4px',
  },
  causalText: { flex: 1 },
  callout: {
    padding: '8px 12px', background: '#eef6ff', border: '1px solid #bcd9ff',
    borderRadius: '4px', fontSize: '12px', color: '#1a3a6a', lineHeight: '1.5',
    marginBottom: '8px',
  },
  causalLink: {
    fontSize: '11px', color: 'var(--accent, #76b900)', textDecoration: 'none',
    fontWeight: 600,
  },

  checklistCols: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' },
  checklistCol: {},
  checklistTitle: {
    fontSize: '10px', textTransform: 'uppercase', fontWeight: 600,
    color: 'var(--text-tertiary, #888)', marginBottom: '6px',
  },
  checkRow: {
    display: 'flex', gap: '6px', alignItems: 'baseline',
    padding: '3px 0', fontSize: '11px',
  },
  checkLabel: {},
  checkNote: { color: 'var(--text-muted, #aaa)', fontStyle: 'italic', fontSize: '10px' },

  loading: { padding: '40px', textAlign: 'center', color: 'var(--text-muted, #aaa)', fontStyle: 'italic' },
  error: { padding: '20px', background: '#fee', color: '#c34', borderRadius: '4px', fontSize: '12px', fontFamily: 'monospace' },

  tableScroll: {
    overflowX: 'auto',
    border: '1px solid var(--border-light, #eee)',
    borderRadius: '4px',
  },
  featureTable: {
    width: '100%',
    minWidth: '900px',
    borderCollapse: 'collapse',
    fontSize: '11px',
  },
  featureTableHeader: {
    fontSize: '10px',
    textTransform: 'uppercase',
    color: 'var(--text-tertiary, #888)',
    fontWeight: 600,
    textAlign: 'left',
    borderBottom: '1px solid var(--border-light, #eee)',
  },
  featureRow: {
    cursor: 'pointer',
    borderBottom: '1px solid var(--border-light, #f0f0f0)',
    transition: 'background 0.1s',
  },
  featureCell: {
    padding: '6px 8px',
    verticalAlign: 'middle',
  },
  colId: { width: '60px' },
  colLabel: { color: 'var(--text-heading, #222)', fontWeight: 500, width: '140px' },
  colContext: { color: 'var(--text-secondary, #555)', fontSize: '11px', minWidth: '180px' },
  colDomain: { width: '50px', textAlign: 'center' },
  colF1: { width: '80px', textAlign: 'right' },
  colAutoInterp: { width: '60px', textAlign: 'right', fontFamily: 'monospace' },
  colMaxAct: { width: '60px', textAlign: 'right', fontFamily: 'monospace' },
  colFreq: { width: '60px', textAlign: 'right', fontFamily: 'monospace' },
  colQC: { width: '80px' },
  colQCScore: { width: '52px', textAlign: 'right' },
  colArrow: { width: '24px', textAlign: 'right', color: 'var(--text-muted, #ccc)' },
  domainBadge: {
    fontSize: '10px',
    fontWeight: 600,
    padding: '2px 6px',
    borderRadius: '3px',
  },
  headlineBadge: {
    fontSize: '14px',
    fontWeight: 700,
    color: COLORS.pass,
  },
  headlineDash: {
    color: 'var(--text-muted, #bbb)',
    fontSize: '14px',
  },
  arrow: { fontSize: '14px' },
  f1Cell: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', lineHeight: '1.1' },
  f1Value: { fontFamily: 'monospace', fontWeight: 600 },
  f1Db: { fontSize: '9px', color: 'var(--text-muted, #aaa)', textTransform: 'uppercase' },
  dotsRow: { display: 'flex', gap: '3px' },
  dot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    display: 'inline-block',
  },
  catalogLink: {
    color: 'var(--accent, #76b900)',
    fontWeight: 600,
    textDecoration: 'none',
    marginLeft: '4px',
  },
}
