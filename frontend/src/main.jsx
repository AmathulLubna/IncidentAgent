import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const MEMORY = {
  none: { label: 'No memory', color: 'neutral', icon: '○', description: 'Cold-start diagnosis' },
  episodic: { label: 'Episodic', color: 'blue', icon: '◉', description: 'Similar past incident' },
  semantic: { label: 'Semantic', color: 'amber', icon: '◆', description: 'Distilled recurring pattern' },
  procedural: { label: 'Procedural', color: 'green', icon: '✓', description: 'Confirmed runbook' },
}

const SAMPLE = {
  service: 'auth-service',
  symptom: 'Users are receiving intermittent 401 responses after token refresh. The issue started shortly after the latest auth deployment. Logs show token validation failures and refresh requests timing out.',
}

async function api(path, options = {}) {
  let res
  try {
    res = await fetch(`${API}${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    })
  } catch (e) {
    throw new Error('Backend unavailable — is FastAPI running on ' + API + '?')
  }
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = body.detail || body.error || `Request failed (${res.status})`
    if (res.status >= 500) throw new Error(`Server error (${res.status}): ${detail}`)
    throw new Error(detail)
  }
  return body
}

function toActive(i) {
  return {
    incident_id: i.id, service: i.service, symptom: i.symptom, timestamp: i.timestamp, status: i.status,
    outcome: i.outcome, diagnosis_text: i.diagnosis_text, memory_hit_type: i.memory_hit_type,
    confidence_pct: i.confidence_pct, root_cause_family_guess: i.predicted_root_cause_family,
    suggested_fix_steps: i.suggested_fix_steps || [], memory_hits: i.memory_hits || [],
    evidence: i.evidence || [], memory_trace: i.memory_trace || [], memory_status: i.memory_status || 'ok',
    memory_errors: i.memory_errors || [], model_used: i.model_used || 'stored result',
    incident_category: i.incident_category, incident_category_label: i.incident_category_label,
    incident_signals: i.incident_signals || [], recommendations: i.recommendations || [],
    confirmed_root_cause: i.root_cause_family, confirmed_fix_steps: i.fix_steps || [],
  }
}

function classifyError(msg) {
  const m = (msg || '').toLowerCase()
  if (m.includes('backend unavailable')) return 'Backend unavailable'
  if (m.includes('llm') || m.includes('groq') || m.includes('diagnosis')) return 'Diagnosis service issue'
  if (m.includes('hindsight') || m.includes('memory')) return 'Memory layer issue'
  return null
}

function Icon({ name, size = 18 }) {
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
    alert: <><path d="M12 3 2.7 20h18.6L12 3Z"/><path d="M12 9v5M12 17.5h.01"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    brain: <><path d="M9 4.5a3 3 0 0 0-3 3v.5a3 3 0 0 0-2 2.8A3.2 3.2 0 0 0 7.2 14 3 3 0 0 0 9 19.5"/><path d="M15 4.5a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 2.8 3.2 3.2 0 0 1-3.2 3.2 3 3 0 0 1-1.8 5.5"/><path d="M9 4.5c1.5 0 3 1.1 3 3v12M15 4.5c-1.5 0-3 1.1-3 3M9 10h3M15 10h-3M9 15h3M15 15h-3"/></>,
    search: <><circle cx="10.8" cy="10.8" r="6.5"/><path d="m16 16 5 5"/></>,
    activity: <path d="M3 12h4l2.3-6 4.4 12 2.3-6H21"/>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    x: <><path d="M6 6l12 12M18 6 6 18"/></>,
    refresh: <><path d="M20 11a8 8 0 0 0-14-5L4 8"/><path d="M4 4v4h4M4 13a8 8 0 0 0 14 5l2-2"/><path d="M20 20v-4h-4"/></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/></>,
    bolt: <path d="m13 2-9 12h7l-1 8 9-12h-7l1-8Z"/>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.8 1.8 0 0 0 .3 2l.1.1-1.8 1.8-.1-.1a1.8 1.8 0 0 0-2-.3 1.8 1.8 0 0 0-1 1.6V20h-2.6v-.2a1.8 1.8 0 0 0-1.1-1.6 1.8 1.8 0 0 0-2 .3l-.1.1-1.8-1.8.1-.1a1.8 1.8 0 0 0 .3-2 1.8 1.8 0 0 0-1.6-1H6v-2.6h.2a1.8 1.8 0 0 0 1.6-1.1 1.8 1.8 0 0 0-.3-2l-.1-.1 1.8-1.8.1.1a1.8 1.8 0 0 0 2 .3 1.8 1.8 0 0 0 1-1.6V4h2.6v.2a1.8 1.8 0 0 0 1.1 1.6 1.8 1.8 0 0 0 2-.3l.1-.1 1.8 1.8-.1.1a1.8 1.8 0 0 0-.3 2 1.8 1.8 0 0 0 1.6 1h.2v2.6h-.2a1.8 1.8 0 0 0-1.6 1.1Z"/></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function MemoryBadge({ type, compact = false }) {
  const m = MEMORY[type] || MEMORY.none
  return <span className={`memory-badge ${m.color} ${compact ? 'compact' : ''}`}><span>{m.icon}</span>{m.label}{!compact && <small>{m.description}</small>}</span>
}

function App() {
  const [page, setPage] = useState('overview')
  const [stats, setStats] = useState(null)
  const [incidents, setIncidents] = useState([])
  const [health, setHealth] = useState('checking')
  const [components, setComponents] = useState(null)
  const [active, setActive] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [query, setQuery] = useState('')
  const [memoryType, setMemoryType] = useState('all')
  const [memoryAll, setMemoryAll] = useState(null)
  const [memoryResults, setMemoryResults] = useState([])
  const [searching, setSearching] = useState(false)
  const [mobileNav, setMobileNav] = useState(false)
  const [notice, setNotice] = useState('')

  const refresh = useCallback(async () => {
    try {
      const [s, i, h] = await Promise.all([api('/stats'), api('/incidents'), api('/health')])
      setStats(s); setIncidents(i)
      setComponents(h.components || null)
      setHealth(h.status === 'ok' ? 'online' : 'degraded'); setError('')
    } catch (e) { setHealth('offline'); setError(e.message) }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const diagnose = async (payload) => {
    setLoading(true); setError('')
    try { const result = await api('/incidents', { method: 'POST', body: JSON.stringify(payload) }); setActive({ ...result, service: payload.service, symptom: payload.symptom }); setPage('incident'); await refresh() }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const resolve = async (payload) => {
    if (!active) return
    setLoading(true); setError('')
    try { const out = await api(`/incidents/${active.incident_id}/resolve`, { method: 'POST', body: JSON.stringify(payload) }); const stored = (out.layers || []).filter(l => l.hindsight !== 'skipped').map(l => l.layer); setNotice(stored.length ? `Stored in ${stored.length} memory layers: ${stored.join(' · ')}` : ''); await refresh(); setActive(null); setPage('history') }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const nav = [
    ['overview', 'Overview', 'grid'],
    ['incident', 'Live incident', 'alert'],
    ['history', 'Incident history', 'clock'],
    ['memory', 'Memory explorer', 'brain'],
  ]

  const currentTitle = { overview: 'Operations overview', incident: active ? `Incident #${active.incident_id}` : 'New incident', history: 'Incident history', memory: 'Memory explorer' }[page]
  const memoryHitRate = stats?.memory_hit_rate_last_5 ?? 0
  const total = stats?.total_incidents ?? 0
  const unresolved = incidents.filter(i => i.status === 'diagnosed').length

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? 'open' : ''}`}>
      <div className="brand"><div className="brand-mark"><Icon name="bolt" size={19}/></div><div><strong>MemorySRE</strong><span>incident intelligence</span></div></div>
      <div className="workspace"><span className="status-dot"/> Production <span className="workspace-caret">⌄</span></div>
      <nav>
        <div className="nav-label">CONTROL ROOM</div>
        {nav.map(([id, label, icon]) => <button key={id} className={`nav-item ${page === id ? 'active' : ''}`} onClick={() => { setPage(id); setMobileNav(false) }}><Icon name={icon}/><span>{label}</span>{id === 'incident' && unresolved > 0 && <em>{unresolved}</em>}</button>)}
      </nav>
      <div className="sidebar-spacer"/>
      <div className="memory-stack">
        <div className="nav-label">MEMORY LAYERS</div>
        {['episodic','semantic','procedural'].map(type => <button key={type} onClick={() => { setMemoryType(type); setPage('memory') }} className="memory-link"><span className={`memory-dot ${type}`}/><span>{MEMORY[type].label}</span><strong>{stats?.[`${type}_count`] ?? '—'}</strong></button>)}
      </div>
      <div className="sidebar-footer"><div className={`connection ${health}`} title={components ? Object.entries(components).map(([k,v])=>`${k}: ${v}`).join(' · ') : ''}><span className="status-dot"/> API {health}</div><span className="version">v1.0 · Hindsight + Groq</span></div>
    </aside>

    <main className="main">
      <header className="topbar">
        <button className="mobile-menu" onClick={() => setMobileNav(!mobileNav)}>☰</button>
        <div><div className="eyebrow">PRODUCTION / SRE</div><h1>{currentTitle}</h1></div>
        <div className="top-actions"><div className="api-pill"><span className={`status-dot ${health}`}/>{health === 'online' ? 'System operational' : health}</div><button className="icon-btn" title="Refresh" onClick={refresh}><Icon name="refresh"/></button></div>
      </header>

      {notice && <div className="notice-banner"><Icon name="check"/><span>{notice}</span><button onClick={() => setNotice('')}><Icon name="x" size={15}/></button></div>}

      {error && <div className="error-banner"><Icon name="alert"/><span>{classifyError(error) ? `${classifyError(error)}: ${error}` : error}</span><button onClick={() => setError('')}><Icon name="x" size={15}/></button></div>}

      {page === 'overview' && <Overview stats={stats} incidents={incidents} onNew={() => { setActive(null); setPage('incident') }} onOpen={(i) => { setActive(toActive(i)); setPage('incident') }}/>} 
      {page === 'incident' && <IncidentPage active={active} loading={loading} onDiagnose={diagnose} onResolve={resolve} onBack={() => setPage('overview')}/>} 
      {page === 'history' && <History incidents={incidents} stats={stats} onOpen={(i) => { setActive(toActive(i)); setPage('incident') }}/>} 
      {page === 'memory' && <MemoryExplorer query={query} setQuery={setQuery} type={memoryType} setType={(t) => { setMemoryType(t); setMemoryAll(null); setMemoryResults([]) }} results={memoryResults} searching={searching} onSearch={async () => { if (!query.trim()) return; setSearching(true); try { if (memoryType === 'all') { setMemoryAll(await api(`/memory/search-all?q=${encodeURIComponent(query)}&limit=3`)); setMemoryResults([]) } else { const r = await api(`/memory/${memoryType}/search?q=${encodeURIComponent(query)}&limit=8`); setMemoryResults(r.results || []); setMemoryAll(null) } } catch(e) { setError(e.message) } finally { setSearching(false) } }} all={memoryAll}/>} 
    </main>
  </div>
}

function Overview({ stats, incidents, onNew, onOpen }) {
  const latest = [...incidents].reverse().slice(0, 5)
  const family = Object.entries(stats?.root_cause_family_counts || {}).sort((a,b) => b[1]-a[1]).slice(0,5)
  const series = stats?.memory_hit_rate_series || []
  return <div className="content">
    <section className="hero-grid">
      <div className="hero-card"><div className="hero-glow"/><div className="hero-content"><span className="live-label"><span className="pulse"/> LIVE RESPONSE</span><h2>Turn production incidents<br/>into institutional memory.</h2><p>Diagnose faster by recalling what your team has already seen, what worked, and what failed.</p><button className="primary-btn" onClick={onNew}><Icon name="plus" size={17}/> Report an incident</button></div><div className="hero-visual"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="core"><Icon name="brain" size={38}/></div><span className="node n1">episodic</span><span className="node n2">semantic</span><span className="node n3">procedural</span></div></div>
      <div className="stat-column">
        <Metric title="Memory hit rate" value={`${Math.round(stats?.memory_hit_rate_last_5 || 0)}%`} note="rolling · last 5 incidents" trend={series.length > 1 ? `${series[series.length-1] >= series[Math.max(0,series.length-2)] ? '↑' : '↓'} vs previous` : 'building baseline'}/>
        <Metric title="Incidents analyzed" value={stats?.total_incidents ?? '—'} note="all-time local history"/>
        <Metric title="Open investigations" value={incidents.filter(i => i.status === 'diagnosed').length} note="awaiting confirmation" accent="amber"/>
      </div>
    </section>

    <section className="section-head"><div><div className="eyebrow">SIGNAL</div><h3>Memory performance</h3></div><span className="muted">Rolling 5-incident hit rate</span></section>
    <div className="performance-card"><div className="chart-wrap"><MiniChart data={series}/></div><div className="chart-side"><span>Current</span><strong>{Math.round(stats?.memory_hit_rate_last_5 || 0)}%</strong><p>Each confirmed incident feeds the memory loop. Repeated patterns become searchable runbooks.</p><div className="loop"><span>Report</span><i>→</i><span>Recall</span><i>→</i><span>Resolve</span><i>→</i><span>Retain</span></div></div></div>

    <div className="two-col"><section className="panel"><div className="panel-head"><div><div className="eyebrow">RECENT</div><h3>Incident feed</h3></div><button className="text-btn" onClick={() => document.querySelector('.nav-item:nth-child(3)')?.click()}>View all <Icon name="chevron" size={14}/></button></div>{latest.length ? <div className="incident-list">{latest.map(i => <IncidentRow key={i.id} incident={i} onClick={() => onOpen(i)}/>)}</div> : <Empty title="No incidents yet" body="Submit the first production signal to start building memory." action="Report incident" onClick={onNew}/>}</section>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">LEARNED</div><h3>Root-cause distribution</h3></div></div>{family.length ? <div className="family-list">{family.map(([name,count], idx) => <div className="family-row" key={name}><div><span className="family-rank">0{idx+1}</span><strong>{name.replaceAll('_',' ')}</strong></div><div className="family-bar"><span style={{width:`${Math.max(10,(count/Math.max(...family.map(x=>x[1])))*100)}%`}}/></div><b>{count}</b></div>)}</div> : <Empty title="Patterns will appear here" body="Resolve a few incidents and the agent will begin grouping root causes."/>}</section></div>
  </div>
}

function Metric({title,value,note,trend,accent}) { return <div className={`metric-card ${accent || ''}`}><span>{title}</span><strong>{value}</strong><small>{note}</small>{trend && <em>{trend}</em>}</div> }

function MiniChart({data}) { if (!data.length) return <div className="chart-empty">No diagnosis history yet</div>; const w=720,h=170,p=12,max=100,min=0; const pts=data.map((v,i)=>`${p+(i*(w-p*2)/Math.max(1,data.length-1))},${h-p-((v-min)/(max-min))*(h-p*2)}`).join(' '); return <svg className="mini-chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"><defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopOpacity=".22"/><stop offset="1" stopOpacity="0"/></linearGradient></defs><line x1="12" x2="708" y1="38" y2="38"/><line x1="12" x2="708" y1="85" y2="85"/><line x1="12" x2="708" y1="132" y2="132"/><polyline points={`${p},${h-p} ${pts} ${w-p},${h-p}`} fill="url(#fill)" stroke="none"/><polyline points={pts} fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>{data.slice(-1).map((v,i)=><circle key={i} cx={p+((data.length-1)*(w-p*2)/Math.max(1,data.length-1))} cy={h-p-((v-min)/(max-min))*(h-p*2)} r="4" fill="currentColor"/>)}</svg> }

function IncidentRow({incident,onClick}) { return <button className="incident-row" onClick={onClick}><div className={`severity ${incident.status === 'resolved' ? 'resolved' : 'open'}`}>{incident.status === 'resolved' ? '✓' : '!'}</div><div className="incident-main"><div><strong>#{incident.id} · {incident.service}</strong><MemoryBadge type={incident.memory_hit_type} compact/></div><p>{incident.symptom}</p></div><div className="incident-meta"><span>{incident.status}</span><small>{incident.confidence_pct ? `${Math.round(incident.confidence_pct)}% conf.` : '—'}</small></div><Icon name="chevron" size={16}/></button> }

function formatTs(ts) {
  if (!ts) return ''
  try { return new Date(ts).toLocaleString() } catch (e) { return ts }
}

function HistoricalHit({ hit }) {
  const isIncident = hit.historical_incident_id != null
  const title = isIncident ? `Historical incident #${hit.historical_incident_id}`
    : hit.memory_type === 'semantic' ? 'Historical pattern'
    : 'Successful runbook'
  return <article className="history-hit">
    <div className="hit-head">
      <MemoryBadge type={hit.memory_type} compact/>
      <strong>{title}</strong>
      {hit.service && <span className="hit-service">{hit.service}</span>}
      {hit.outcome && <span className={`outcome-tag ${hit.outcome}`}>{hit.outcome === 'success' ? 'RESOLVED · SUCCESS' : 'RESOLVED · FAIL'}</span>}
    </div>
    {hit.root_cause_family && <div className="hit-meta"><span>Root cause</span><code>{String(hit.root_cause_family).slice(0, 160)}</code></div>}
    <p>{hit.snippet}</p>
    <div className="hit-foot">
      {hit.timestamp && <span>occurred {formatTs(hit.timestamp)}</span>}
      {hit.relevance_score != null && <span>relevance {Number(hit.relevance_score).toFixed(3)}</span>}
      {hit.source && <span className="mem-id">{hit.source === 'local' ? 'layer store' : 'Hindsight'}</span>}
      {hit.memory_id && <span className="mem-id">memory {String(hit.memory_id).slice(0, 14)}</span>}
    </div>
    {hit.topical_reason && <div className="hit-why">Why relevant: {hit.topical_reason}</div>}
  </article>
}

function MemoryTracePanel({ trace, hits, errors }) {
  if (trace?.length) {
    return <div className="trace-list">{trace.map((s, i) => <div className={`trace-tier ${s.status}`} key={i}>
      <span className="tier-name">{s.tier}</span>
      {s.status === 'failed'
        ? <span className="tier-error">Memory search failed: {s.error}</span>
        : <span className="tier-count">{s.accepted} historical {s.accepted === 1 ? 'memory' : 'memories'} used · {s.returned} returned{s.rejected ? ` · ${s.rejected} rejected as current incident` : ''}{s.low_relevance ? ` · ${s.low_relevance} dropped as weak match` : ''}{s.off_topic ? ` · ${s.off_topic} dropped as a different kind of problem` : ''}{s.local_added ? ` · ${s.local_added} added from layer store` : ''}</span>}
      <code>{s.query}</code>
    </div>)}</div>
  }
  // Older records have no stored trace — show what is known, honestly.
  const groups = (hits || []).reduce((acc, h) => { acc[h.memory_type] = (acc[h.memory_type] || 0) + 1; return acc }, {})
  const tiers = Object.keys(groups)
  return <div className="trace-list">
    {tiers.length
      ? tiers.map(t => <div className="trace-tier" key={t}><span className="tier-name">{t}</span><span className="tier-count">{groups[t]} historical {groups[t] === 1 ? 'memory' : 'memories'} used</span></div>)
      : <p className="muted">No memory searches recorded for this incident.</p>}
    {(errors || []).map((e, i) => <div className="trace-tier failed" key={`e${i}`}><span className="tier-name">error</span><span className="tier-error">{e}</span></div>)}
  </div>
}


function CopyCmd({ text }) {
  const [done, setDone] = useState(false)
  return <button className="copy-btn" type="button" onClick={() => { navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1200) }}>{done ? 'copied' : 'copy'}</button>
}

function RecList({ title, items, ordered }) {
  if (!items?.length) return null
  const Tag = ordered ? 'ol' : 'ul'
  return <div className="rec-block"><h5>{title}</h5><Tag>{items.map((x, i) => <li key={i}>{x}</li>)}</Tag></div>
}

function Recommendations({ recs }) {
  if (!recs.length) return null
  return <section className="panel section-card recs-card">
    <div className="section-label-row"><span className="section-label recs">DEVELOPER RECOMMENDATIONS</span><span className="section-note">problem → solution · specific to this incident type · nothing is executed automatically</span></div>
    {recs.map((r, idx) => <article className={`rec ${r.severity}`} key={idx}>
      <header className="rec-head">
        <div><span className={`sev-tag ${r.severity}`}>{r.severity.toUpperCase()}</span><span className="rec-role">{r.role === 'primary' ? 'Primary' : 'Also possible'} · {r.category_label} · {Math.round(r.confidence_pct)}% match</span></div>
        <h4>{r.title}</h4>
      </header>
      <div className="rec-problem"><span>PROBLEM</span><p>{r.problem}</p></div>
      <RecList title="Likely causes" items={r.likely_causes}/>
      <RecList title="Do now (mitigate)" items={r.immediate_actions} ordered/>
      {r.investigate?.length > 0 && <div className="rec-block"><h5>Investigate</h5>{r.investigate.map((c, i) => <div className="cmd" key={i}><small>{c.label}</small><div><code>{c.command}</code><CopyCmd text={c.command}/></div></div>)}</div>}
      <RecList title="Permanent fix" items={r.permanent_fix} ordered/>
      {r.proven_in_history?.length > 0 && <div className="rec-block proven"><h5>Confirmed in your own history</h5>{r.proven_in_history.map((p, i) => <div key={i}><small>Incident #{p.incident_id}{p.service ? ` · ${p.service}` : ''}{p.outcome ? ` · ${p.outcome}` : ''}</small><ol>{p.steps.map((x, j) => <li key={j}>{x}</li>)}</ol></div>)}</div>}
      <RecList title="Verify it worked" items={r.verify}/>
      <RecList title="Prevent recurrence" items={r.prevent}/>
      {r.rollback && <div className="rec-note rollback"><b>Rollback</b> {r.rollback}</div>}
      {r.caution && <div className="rec-note caution"><b>Caution</b> {r.caution}</div>}
    </article>)}
  </section>
}

function IncidentPage({active,loading,onDiagnose,onResolve,onBack}) {
  const [service,setService]=useState(active?.service || '')
  const [symptom,setSymptom]=useState(active?.symptom || '')
  const [root,setRoot]=useState(active?.root_cause_family_guess || '')
  const [steps,setSteps]=useState((active?.suggested_fix_steps || []).join('\n'))
  const [outcome,setOutcome]=useState('success')
  useEffect(()=>{ if(active){setRoot(active.root_cause_family_guess||'');setSteps((active.suggested_fix_steps||[]).join('\n'))} },[active])
  const submit=e=>{e.preventDefault();if(service.trim()&&symptom.trim())onDiagnose({service,symptom})}
  if (!active) return <div className="content narrow"><div className="incident-intro"><span className="live-label"><span className="pulse"/> NEW INCIDENT</span><h2>What is breaking?</h2><p>Give the agent the service and raw production signal. It will recall prior incidents before forming a diagnosis.</p></div><form className="incident-form" onSubmit={submit}><label>Service name<input value={service} onChange={e=>setService(e.target.value)} placeholder="e.g. auth-service" autoFocus/></label><label>Symptom, logs, or error trace<textarea value={symptom} onChange={e=>setSymptom(e.target.value)} placeholder="Describe what operators are seeing..." rows="8"/></label><div className="form-foot"><button type="button" className="ghost-btn" onClick={()=>{setService(SAMPLE.service);setSymptom(SAMPLE.symptom)}}>Use demo incident</button><button className="primary-btn" disabled={loading||!service.trim()||!symptom.trim()}>{loading ? <><span className="spinner"/> Incident recorded → searching memory → generating diagnosis…</> : <><Icon name="send" size={16}/> Diagnose incident</>}</button></div></form></div>

  const hits = active.memory_hits || []
  const fixes = active.suggested_fix_steps || []
  const resolved = active.status === 'resolved'

  return <div className="content">
    <div className="incident-top">
      <div>
        <button className="back-btn" onClick={onBack}>← Back to feed</button>
        <div className="eyebrow">{resolved ? 'RESOLVED INCIDENT' : 'LIVE DIAGNOSIS'}</div>
        <h2>Incident #{active.incident_id}</h2>
        <p className="incident-sub">{active.service || service || 'production service'} · {resolved ? 'confirmed resolution recorded' : 'open — awaiting engineer confirmation'}</p>
      </div>
      <div className="top-badges"><MemoryBadge type={active.memory_hit_type}/>{active.memory_status === 'degraded' && <span className="status-tag degraded">memory degraded</span>}</div>
    </div>

    <div className="diagnosis-grid">
      <section className="diagnosis-main">

        {/* 1 — CURRENT INCIDENT (never historical evidence) */}
        <section className="panel section-card current-card">
          <div className="section-label-row"><span className="section-label current">CURRENT INCIDENT</span><span className="section-note">happening now · not historical evidence</span></div>
          <h3>{active.service || service || 'production service'}</h3>
          <pre className="current-signal">{active.symptom || symptom || 'Incident submitted via API.'}</pre>
        </section>

        {/* 2 — DIAGNOSIS */}
        <section className="panel section-card">
          <div className="section-label-row">
            <span className="section-label diagnose">DIAGNOSIS</span>
            {active.confidence_pct != null && <div className="confidence"><span>Confidence</span><strong>{Math.round(active.confidence_pct)}%</strong></div>}
          </div>
          <div className="diagnosis-copy">{active.diagnosis_text}</div>
          {active.incident_category_label && <div className="root-cause"><span>Detected category</span><strong>{active.incident_category_label}</strong></div>}
          {active.incident_signals?.length > 0 && <div className="signal-chips">{active.incident_signals.map((x,i)=><code key={i}>{x}</code>)}</div>}
          {active.root_cause_family_guess && <div className="root-cause"><span>Predicted root cause</span><strong>{active.root_cause_family_guess.replaceAll('_',' ')}</strong></div>}
          {active.evidence?.length > 0 && <div className="evidence"><span>Why this recommendation?</span><p className="evidence-note">Because these past memories matched this incident — strongest match first, weakest already filtered out:</p><ul>{active.evidence.map((e,i)=><li key={i}>{e}</li>)}</ul></div>}
        </section>

        {/* 3 — HISTORICAL EVIDENCE (separate from the current incident) */}
        <section className="panel section-card historical-card">
          <div className="section-label-row">
            <span className="section-label historical">HISTORICAL EVIDENCE</span>
            <span className="section-note">{hits.length ? `${hits.length} retrieved from Hindsight` : 'none retrieved'}</span>
          </div>
          {hits.length
            ? <div className="history-hits">{hits.map((h,i)=><HistoricalHit key={h.memory_id || i} hit={h}/>)}</div>
            : active.memory_status === 'degraded'
              ? <p className="muted">Memory search failed — see the memory trace below. No historical evidence was used, and the current incident is never counted as its own precedent.</p>
              : <p className="muted">No previously-resolved incident matched this one. Treat the diagnosis as a fresh hypothesis and triage manually.</p>}
        </section>

        {/* 4a — DEVELOPER RECOMMENDATIONS (problem -> solution, per category) */}
        <Recommendations recs={active.recommendations || []}/>

        {/* 4 — RECOMMENDED ACTIONS */}
        <section className="panel section-card actions-card">
          <div className="section-label-row">
            <span className="section-label actions">AGENT'S ORDERED STEPS</span>
            <span className="section-note">recommendations only — nothing is executed automatically</span>
          </div>
          {fixes.length
            ? <ol className="fix-list">{fixes.map((s,i)=><li key={i}><span>{String(i+1).padStart(2,'0')}</span><p>{s}</p></li>)}</ol>
            : <p className="muted">No steps were returned. Treat the diagnosis as a hypothesis and triage manually.</p>}
        </section>

        {/* 5 — MEMORY USED (real pipeline, incl. zero-result tiers) */}
        <section className="panel section-card trace-card">
          <div className="section-label-row"><span className="section-label trace">MEMORY USED</span><span className="section-note">what actually happened during recall</span></div>
          <MemoryTracePanel trace={active.memory_trace} hits={hits} errors={active.memory_errors}/>
        </section>
      </section>

      <aside className="right-stack">
        <div className="panel signal-card">
          <div className="eyebrow">AGENT RUN</div>
          <div className="model-line"><span>Reasoning model</span><strong>{active.model_used}</strong></div>
          <div className="model-line"><span>Memory status</span><strong className={active.memory_status === 'degraded' ? 'warn' : ''}>{active.memory_status === 'degraded' ? 'degraded' : 'ok'}</strong></div>
          <div className="model-line"><span>Historical tier used</span><strong>{active.memory_hit_type}</strong></div>
          {active.memory_errors?.length > 0 && <ul className="error-list">{active.memory_errors.map((e,i)=><li key={i}>{e}</li>)}</ul>}
        </div>

        <div className="panel resolve-card">
          <div className="eyebrow">{resolved ? 'CONFIRMED RESOLUTION' : 'ENGINEER CONFIRMATION'}</div>
          {resolved ? <div className="confirmed-block">
            <p className="section-note">Human-confirmed — this is what was retained as learned experience.</p>
            <div className="root-cause"><span>Confirmed root cause</span><strong>{(active.confirmed_root_cause || active.root_cause_family || '').replaceAll('_',' ') || '—'}</strong></div>
            <div className="confirmed-fix"><span>Confirmed fix</span>{(active.confirmed_fix_steps || []).length ? <ol>{active.confirmed_fix_steps.map((s,i)=><li key={i}>{s}</li>)}</ol> : <p className="muted">—</p>}</div>
            <div className="outcome"><span className={`outcome-tag ${active.outcome || ''}`}>OUTCOME: {(active.outcome || '').toUpperCase() || '—'}</span></div>
          </div> : <>
            <p>Your confirmation becomes historical memory and can promote a reusable runbook. The agent's prediction stays stored separately as the predicted root cause.</p>
            <label>Confirmed root-cause family<input value={root} onChange={e=>setRoot(e.target.value)} placeholder="e.g. db_connection_pool_exhaustion"/></label>
            <label>Confirmed fix steps<textarea value={steps} onChange={e=>setSteps(e.target.value)} rows="6"/></label>
            <div className="outcome"><button className={outcome==='success'?'selected success':''} onClick={()=>setOutcome('success')} type="button"><Icon name="check" size={15}/> Resolved</button><button className={outcome==='fail'?'selected fail':''} onClick={()=>setOutcome('fail')} type="button"><Icon name="x" size={15}/> Failed</button></div>
            <button className="primary-btn full" disabled={loading||!root.trim()} onClick={()=>onResolve({root_cause_family:root,fix_steps:steps.split('\n').map(s=>s.trim()).filter(Boolean),outcome})}>{loading?<><span className="spinner"/> Retaining…</>:<><Icon name="database" size={16}/> Confirm &amp; retain</>}</button>
          </>}
        </div>
      </aside>
    </div>
  </div>
}

function History({incidents,stats,onOpen}) { return <div className="content"><section className="panel"><div className="history-toolbar"><div><div className="eyebrow">AUDIT TRAIL</div><h3>{incidents.length} incidents</h3></div><div className="history-summary"><span><b>{incidents.filter(i=>i.status==='resolved').length}</b> resolved</span><span><b>{Math.round(stats?.memory_hit_rate_last_5||0)}%</b> memory hit rate</span></div></div>{incidents.length ? <div className="table-wrap"><table><thead><tr><th>Incident</th><th>Service</th><th>Memory</th><th>Root cause</th><th>Status</th><th>Confidence</th><th/></tr></thead><tbody>{[...incidents].reverse().map(i=><tr key={i.id} onClick={()=>onOpen(i)}><td><strong>#{i.id}</strong></td><td>{i.service}</td><td><MemoryBadge type={i.memory_hit_type} compact/></td><td>{i.root_cause_family ? i.root_cause_family.replaceAll('_',' ') : i.predicted_root_cause_family ? `${i.predicted_root_cause_family.replaceAll('_',' ')} (predicted)` : '—'}</td><td><span className={`status-tag ${i.status}`}>{i.status}</span></td><td>{i.confidence_pct ? `${Math.round(i.confidence_pct)}%` : '—'}</td><td><Icon name="chevron" size={15}/></td></tr>)}</tbody></table></div>:<Empty title="Nothing here yet" body="Incident records will appear after the first diagnosis."/>}</section></div> }

const LAYER_COPY = {
  episodic: { function: 'WHAT HAPPENED', purpose: 'The case file of one resolved incident: symptoms, signals, what was tried, how it ended.' },
  semantic: { function: 'WHY IT HAPPENS', purpose: 'Causal knowledge: how to recognise this failure, the mechanism behind it, what it gets confused with.' },
  procedural: { function: 'HOW TO FIX IT', purpose: 'The reusable runbook: trigger, ordered steps, verification, rollback, track record.' },
}

function FieldValue({ v }) {
  if (Array.isArray(v)) return <ol>{v.map((x, i) => <li key={i}>{x}</li>)}</ol>
  return <p>{String(v)}</p>
}

function LayerResult({ r, index }) {
  const fields = Object.entries(r.fields || {}).filter(([, v]) => v && (!Array.isArray(v) || v.length))
  return <article className={`memory-result layer-${r.layer}`}>
    <div className="result-index">{String(index + 1).padStart(2, '0')}</div>
    <div className="result-body">
      <div className="result-head">
        <MemoryBadge type={r.layer} compact/>
        <strong>{r.title || r.gist}</strong>
        {r.category_label && <span className="cat-chip">{r.category_label}</span>}
        {r.outcome && <span className={`outcome-tag ${r.outcome}`}>{String(r.outcome).toUpperCase()}</span>}
      </div>
      {fields.length ? <dl className="layer-fields">{fields.map(([k, v]) => <div key={k}><dt>{k}</dt><dd><FieldValue v={v}/></dd></div>)}</dl> : <p className="raw-text">{r.text}</p>}
      <span className="result-foot">relevance · {typeof r.score === 'number' ? r.score.toFixed(3) : '—'} · {r.source === 'local' ? 'layer store' : 'Hindsight'}{r.legacy ? ' · older memory (unstructured)' : ''}{r.incident_id != null ? ` · incident #${r.incident_id}` : ''}</span>
    </div>
  </article>
}

function MemoryExplorer({ query, setQuery, type, setType, results, searching, onSearch, all }) {
  const tabs = [['all', 'All layers', 'One keyword, three views'], ['episodic', 'Episodic', 'What happened'], ['semantic', 'Semantic', 'Why it happens'], ['procedural', 'Procedural', 'How to fix it']]
  return <div className="content">
    <div className="memory-hero"><div><span className="live-label"><span className="pulse"/> MEMORY LAYERS</span><h2>Three layers. Three different answers.</h2><p>Every resolved incident is stored three ways — what happened, why it happens, and how it was fixed. Search a keyword and each layer answers its own question.</p></div><div className="memory-orb"><Icon name="brain" size={42}/></div></div>
    <div className="memory-tabs four">{tabs.map(([t, label, sub]) => <button className={type === t ? 'selected' : ''} onClick={() => { setType(t); setQuery('') }} key={t}>{t !== 'all' && <span className={`memory-dot ${t}`}/>}{label}<small>{sub}</small></button>)}</div>
    {type !== 'all' && <div className={`layer-purpose ${type}`}><b>{LAYER_COPY[type].function}</b><span>{LAYER_COPY[type].purpose}</span></div>}
    <form className="search-bar" onSubmit={e => { e.preventDefault(); onSearch() }}><Icon name="search"/><input value={query} onChange={e => setQuery(e.target.value)} placeholder={type === 'all' ? 'Search all layers… e.g. token refresh' : `Search ${type} memory…`}/><button disabled={searching || !query.trim()}>{searching ? <span className="spinner"/> : <><Icon name="search" size={15}/> Search</>}</button></form>
    {type === 'all' && all
      ? <section className="layer-columns">{['episodic', 'semantic', 'procedural'].map(t => <div className={`layer-col ${t}`} key={t}>
          <div className={`layer-purpose ${t}`}><b>{MEMORY[t].label} · {LAYER_COPY[t].function}</b><span>{LAYER_COPY[t].purpose}</span></div>
          {(all[t]?.results || []).length ? all[t].results.map((r, i) => <LayerResult r={r} index={i} key={i}/>) : <div className="layer-empty">{all[t]?.error ? `Search failed: ${all[t].error}` : t === 'procedural' ? 'No runbook stored for this keyword (runbooks exist only for confirmed successful fixes).' : 'Nothing stored in this layer for this keyword.'}</div>}
          {all[t]?.warning && <div className="layer-warn">Hindsight: {all[t].warning}</div>}
        </div>)}</section>
      : <section className="results">{results.length ? results.map((r, i) => <LayerResult r={r} index={i} key={i}/>) : <div className="empty-memory"><div><Icon name="search" size={25}/></div><h3>{type === 'all' ? 'Search all three layers' : `Search the ${type} layer`}</h3><p>Try “token refresh”, “connection pool”, or “no space left”.</p></div>}</section>}
  </div>
}

function Empty({title,body,action,onClick}) { return <div className="empty"><div className="empty-icon"><Icon name="activity"/></div><h3>{title}</h3><p>{body}</p>{action&&<button className="ghost-btn" onClick={onClick}>{action}</button>}</div> }

createRoot(document.getElementById('root')).render(<App />)
