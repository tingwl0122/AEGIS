#!/usr/bin/env python3
"""
Graph-Structured Error Attribution with Schema-Augmented Inference
Tailored to the AEGIS dataset format.

Pipeline:
  1. Agent Dependency Graph (ADG) from conversation_history
  2. Error schema extraction from labeled training data (label-informed)
  3. Schema retrieval + graph-augmented prompt construction

Usage:
    python graph_schema_pipeline.py build-schemas --train_data train.jsonl --output schemas.json
    python graph_schema_pipeline.py prepare-inference --input test.jsonl --schemas schemas.json --prompt_template prompt.txt --output augmented.jsonl
    python graph_schema_pipeline.py analyze --schemas schemas.json
"""

import json, re, argparse, logging, hashlib
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALL_ERROR_CODES = [
    "FM-1.1","FM-1.2","FM-1.3","FM-1.4","FM-1.5",
    "FM-2.1","FM-2.2","FM-2.3","FM-2.4","FM-2.5","FM-2.6",
    "FM-3.1","FM-3.2","FM-3.3",
]
ERROR_CATEGORY = {
    "FM-1.1":"specification","FM-1.2":"specification","FM-1.3":"specification",
    "FM-1.4":"specification","FM-1.5":"specification",
    "FM-2.1":"coordination","FM-2.2":"coordination","FM-2.3":"coordination",
    "FM-2.4":"coordination","FM-2.5":"coordination","FM-2.6":"coordination",
    "FM-3.1":"verification","FM-3.2":"verification","FM-3.3":"verification",
}
ERROR_DESCRIPTIONS = {
    "FM-1.1":"Task specification deviation - Agent deviates from specified task requirements",
    "FM-1.2":"Role specification deviation - Agent acts outside its designated role",
    "FM-1.3":"Add redundant steps - Agent adds unnecessary or duplicate steps",
    "FM-1.4":"Remove conversation history - Agent ignores important context from previous turns",
    "FM-1.5":"Remove termination conditions - Agent fails to define proper stopping criteria",
    "FM-2.1":"Repeat handled tasks - Agent redundantly handles already completed tasks",
    "FM-2.2":"Make request ambiguous - Agent provides unclear instructions to other agents",
    "FM-2.3":"Deviate from main goal - Agent pursues objectives unrelated to the main task",
    "FM-2.4":"Hide important information - Agent withholds crucial information needed by others",
    "FM-2.5":"Ignore other agents - Agent fails to consider input from other agents",
    "FM-2.6":"Inconsistent reasoning - Agent's logic contradicts its own previous statements",
    "FM-3.1":"Premature termination - Agent stops before all requirements are met",
    "FM-3.2":"Remove verification steps - Agent skips necessary validation or testing steps",
    "FM-3.3":"Incorrect verification - Agent performs flawed or wrong verification",
}
DISTINGUISH_HINT = {
    "FM-1.1":"Look for: output that solves WRONG task or uses wrong method/language/format.",
    "FM-1.2":"Look for: agent performing actions outside its designated role.",
    "FM-1.3":"Look for: repeated/redundant steps, circular logic loops adding no value.",
    "FM-1.4":"Look for: agent ignoring/contradicting info from previous conversation turns.",
    "FM-1.5":"Look for: missing stopping criteria, no convergence condition, infinite loops.",
    "FM-2.1":"Look for: agent re-doing work already completed by itself or another agent.",
    "FM-2.2":"Look for: vague/ambiguous instructions leaving critical details unspecified.",
    "FM-2.3":"Look for: agent going off-topic, pursuing tangential objectives.",
    "FM-2.4":"Look for: agent deliberately omitting key information that others need.",
    "FM-2.5":"Look for: agent not acknowledging or incorporating feedback from peers.",
    "FM-2.6":"Look for: logical contradictions between agent's earlier and later statements.",
    "FM-3.1":"Look for: task declared complete when clearly unfinished, missing steps.",
    "FM-3.2":"Look for: no testing/checking/validation performed on output.",
    "FM-3.3":"Look for: verification that incorrectly approves wrong output or wrong criteria.",
}

# ============================================================
# Part 1: Agent Dependency Graph
# ============================================================

@dataclass
class AgentDependencyGraph:
    agents: List[str] = field(default_factory=list)
    agent_phases: Dict[str, Set[str]] = field(default_factory=dict)
    agent_steps: Dict[str, List[int]] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    topology: str = ""
    framework: str = ""

    def get_predecessors(self, agent: str) -> List[str]:
        return list(set(s for s, d, _ in self.edges if d == agent and s != agent))
    def get_successors(self, agent: str) -> List[str]:
        return list(set(d for s, d, _ in self.edges if s == agent and d != agent))

    def serialize(self) -> str:
        lines = [f"TOPOLOGY: {self.topology}"]
        if self.framework: lines.append(f"FRAMEWORK: {self.framework}")
        lines.append(f"AGENTS: {', '.join(self.agents)}")
        unique = list(set(self.edges))
        if unique:
            lines.append("\nINFORMATION FLOW:")
            for s, d, t in unique:
                lines.append(f"  {s} --[{t}]--> {d}")
        lines.append("\nAGENT ANALYSIS GUIDE:")
        for agent in self.agents:
            phases = self.agent_phases.get(agent, set())
            preds = self.get_predecessors(agent)
            succs = self.get_successors(agent)
            parts = [f"  {agent}: phase={'|'.join(sorted(phases))}"]
            if preds: parts.append(f"receives_from=[{','.join(preds)}]")
            if succs: parts.append(f"sends_to=[{','.join(succs)}]")
            if not preds: parts.append("(ENTRY POINT - errors here are ROOT CAUSES)")
            if not succs: parts.append("(TERMINAL)")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def get_propagation_hints(self) -> str:
        hints = []
        for agent in self.agents:
            preds = self.get_predecessors(agent)
            succs = self.get_successors(agent)
            if not preds and succs:
                hints.append(f"- {agent} is an ENTRY POINT. If faulty, errors propagate to {', '.join(succs)}.")
            elif preds and succs:
                hints.append(f"- {agent} is INTERMEDIATE (from {', '.join(preds)}, to {', '.join(succs)}). Check if error is independent or propagated from upstream.")
            elif preds and not succs:
                hints.append(f"- {agent} is TERMINAL (from {', '.join(preds)}). If this agent fails, check if upstream already corrupted its input.")
        return "\n".join(hints)


def build_graph(history: List[Dict], metadata: Dict = None) -> AgentDependencyGraph:
    """Build ADG from AEGIS conversation_history entries: {step, agent_name, content, phase}."""
    g = AgentDependencyGraph()
    g.framework = (metadata or {}).get("framework", "")
    if not history: return g

    first_seen = {}
    for e in history:
        a = e.get("agent_name", "Unknown")
        s = e.get("step", 0)
        p = e.get("phase", "")
        if a not in first_seen:
            first_seen[a] = s
            g.agent_phases[a] = set()
            g.agent_steps[a] = []
        g.agent_phases[a].add(p)
        g.agent_steps[a].append(s)
    g.agents = sorted(first_seen.keys(), key=lambda a: first_seen[a])

    edge_set = set()
    # Sequential edges
    prev = None
    for e in history:
        a = e.get("agent_name", "Unknown")
        if prev and prev != a:
            edge_set.add((prev, a, "sequential"))
        prev = a
    # Phase-based edges
    by_phase = defaultdict(set)
    for e in history:
        by_phase[e.get("phase", "")].add(e.get("agent_name", ""))
    # Initialization agents feed into reasoning/discussion agents
    init_phases = {"initialization"}
    work_phases = {"reasoning", "discussion", "execution", "action"}
    review_phases = {"evaluation", "review", "scoring"}
    init_agents = set()
    for p in init_phases:
        init_agents |= by_phase.get(p, set())
    work_agents = set()
    for p in work_phases:
        work_agents |= by_phase.get(p, set())
    review_agents = set()
    for p in review_phases:
        review_agents |= by_phase.get(p, set())
    for ia in init_agents:
        for wa in work_agents:
            if ia != wa: edge_set.add((ia, wa, "initialization"))
    for wa in work_agents:
        for ra in review_agents:
            if wa != ra:
                edge_set.add((wa, ra, "submit_for_review"))
                edge_set.add((ra, wa, "feedback"))
    # Discussion phase: agents in discussion interact with each other
    disc_agents = by_phase.get("discussion", set())
    disc_list = sorted(disc_agents)
    for i in range(len(disc_list)):
        for j in range(i+1, len(disc_list)):
            edge_set.add((disc_list[i], disc_list[j], "discussion"))
            edge_set.add((disc_list[j], disc_list[i], "discussion"))
    g.edges = list(edge_set)

    # Topology inference
    n = len(g.agents)
    directed = set((s,d) for s,d,_ in g.edges)
    if n <= 1: g.topology = "single"
    elif n == 2: g.topology = "chain" if len(directed) <= 2 else "bidirectional"
    elif init_agents and (work_agents or review_agents):
        g.topology = "hierarchical"
    elif disc_agents and len(disc_agents) >= 2:
        g.topology = "debate"
    else:
        out_deg = Counter(s for s,d in directed)
        if any(v >= n-1 for v in out_deg.values()): g.topology = "star"
        else: g.topology = "graph"
    return g


# ============================================================
# Part 2: Error Schema Extraction
# ============================================================

@dataclass
class ErrorSchema:
    schema_id: str
    error_code: str
    error_description: str
    agent_role: str
    agent_phase: str
    framework_pattern: str
    topology_pattern: str
    injection_strategy: str
    signature: str
    structural_context: str
    distinguishing_features: str
    example_snippet: str
    frequency: int = 0
    benchmark_distribution: str = ""


def infer_role(agent_name: str, phases: Set[str],
               history: List[Dict] = None,
               graph: 'AgentDependencyGraph' = None) -> str:
    """
    Dynamically infer agent role from multiple signals — NO hardcoded role list.
    
    Signals (priority order):
      1. Behavioral: what does the agent actually DO in its content?
      2. Structural: where does it sit in the graph?
      3. Phase: if the framework provides a phase label
      4. Name: extract meaningful tokens from agent name (never a fixed map)
    
    Returns a descriptive role string discovered on the fly.
    """
    role_parts = []
    
    # ---- Signal 1: Behavioral analysis (strongest signal) ----
    behavioral = ""
    if history:
        agent_turns = [h for h in history if h.get("agent_name") == agent_name]
        all_content = " ".join(str(t.get("content","")) for t in agent_turns).lower()
        
        if all_content:
            # Count behavioral signals — let the data speak
            signals = {}
            signals["code_production"] = len(re.findall(r'```|def\s+\w|class\s+\w|import\s+\w|return\s+', all_content))
            signals["evaluation"]      = len(re.findall(r'score\b|rating|correct|incorrect|approv|reject|review|grade', all_content))
            signals["role_assignment"]  = len(re.findall(r'role|expert|specialist|assign|you (?:are|will|should)', all_content))
            signals["planning"]        = len(re.findall(r'step\s*\d|first.*then|plan|decompos|subtask|delegat', all_content))
            signals["web_research"]    = len(re.findall(r'search|browse|fetch|url|http|web\b|click|navigate|website', all_content))
            signals["verification"]    = len(re.findall(r'test\b|verify|check|assert|valid|pass\b|fail\b|bug', all_content))
            signals["debate"]          = len(re.findall(r'agree|disagree|opinion|perspective|argument|counter|debate', all_content))
            signals["tool_use"]        = len(re.findall(r'tool\b|execute|command|terminal|output:\s', all_content))
            signals["summarization"]   = len(re.findall(r'summar|conclud|final answer|therefore|in conclusion', all_content))
            
            # Pick top signal (must have at least 2 matches to be meaningful)
            top = max(signals.items(), key=lambda x: x[1])
            if top[1] >= 2:
                behavioral = top[0]
    
    # ---- Signal 2: Structural position ----
    structural = ""
    if graph:
        preds = graph.get_predecessors(agent_name)
        succs = graph.get_successors(agent_name)
        steps = graph.agent_steps.get(agent_name, [])
        
        if not preds and succs:
            structural = "entry_point"
        elif preds and not succs:
            structural = "terminal"
        elif len(set(succs)) >= 3:
            structural = "hub"
        
        # One-shot at start = likely initializer/assigner
        if steps and len(steps) == 1 and steps[0] <= 2 and not behavioral:
            structural = "one_shot_init"
    
    # ---- Signal 3: Phase label (if the framework provides it) ----
    phase_role = ""
    if phases:
        clean_phases = phases - {"", "unknown", "general", "default"}
        if clean_phases:
            # Use the phase directly as a role descriptor
            phase_role = "|".join(sorted(clean_phases))
    
    # ---- Signal 4: Agent name tokens (always available, never a fixed map) ----
    # Split camelCase, snake_case, spaces into tokens
    name_tokens = re.findall(r'[A-Z][a-z]+|[a-z]+|\d+', agent_name)
    name_clean = "_".join(t.lower() for t in name_tokens) if name_tokens else agent_name.lower()
    
    # ---- Combine: behavioral > structural > phase > name ----
    if behavioral:
        role_parts.append(behavioral)
    if structural:
        role_parts.append(structural)
    if not behavioral and phase_role:
        role_parts.append(phase_role)
    
    if role_parts:
        return "+".join(role_parts)
    
    # Fallback: the agent's own name is its role (e.g. "tickets_pricing_expert")
    return name_clean


def safe_content(entry: Dict, max_len: int = 0) -> str:
    """Safely get content from a history entry, handling None."""
    c = entry.get("content")
    if c is None:
        return "(no content)"
    c = str(c)
    if max_len > 0 and len(c) > max_len:
        c = c[:max_len - 30] + "\n[...truncated...]"
    return c


def extract_evidence(history: List[Dict], agent: str, graph: AgentDependencyGraph, max_chars=600):
    """Extract what the faulty agent did + its structural context."""
    agent_turns = [h for h in history if h.get("agent_name") == agent]
    preds = set(graph.get_predecessors(agent))

    # Agent's own actions — take ALL turns, let max_chars handle truncation
    snippet_parts = []
    chars_used = 0
    for t in agent_turns:
        c = safe_content(t, max_len=250)
        line = f"[Step {t.get('step')}] {c}"
        if chars_used + len(line) > max_chars:
            snippet_parts.append(f"[...{len(agent_turns) - len(snippet_parts)} more turns omitted...]")
            break
        snippet_parts.append(line)
        chars_used += len(line) + 1
    snippet = "\n".join(snippet_parts)

    # Upstream context
    ctx_parts = []
    if preds and agent_turns:
        first_step = agent_turns[0].get("step", 0)
        pred_turns = [h for h in history if h.get("agent_name") in preds and h.get("step",0) < first_step][-2:]
        if pred_turns:
            ctx_parts.append("UPSTREAM INPUT:")
            for t in pred_turns:
                ctx_parts.append(f"  {t.get('agent_name')}: {safe_content(t, max_len=120)}")
    struct = "\n".join(ctx_parts)[:300]
    return snippet, struct


def build_schemas(samples: List[Dict], min_freq: int = 1) -> List[ErrorSchema]:
    logger.info(f"Building schemas from {len(samples)} samples")
    groups = defaultdict(lambda: {"examples":[],"frameworks":Counter(),"benchmarks":Counter(),
                                   "topologies":Counter(),"injections":Counter()})
    for sample in samples:
        inp = sample.get("input",{})
        out = sample.get("output",{})
        meta = sample.get("metadata",{})
        history = inp.get("conversation_history",[])
        if not history or not out.get("faulty_agents"): continue

        graph = build_graph(history, meta)
        for fa in out["faulty_agents"]:
            aname = fa.get("agent_name","")
            ecode = fa.get("error_type","")
            inj = fa.get("injection_strategy","unknown")
            if not aname or ecode not in ALL_ERROR_CODES: continue

            phases = graph.agent_phases.get(aname, set())
            role = infer_role(aname, phases, history=history, graph=graph)
            phase = sorted(phases)[0] if phases else "unknown"
            snippet, struct = extract_evidence(history, aname, graph)

            key = (ecode, role, phase)
            g = groups[key]
            g["examples"].append({"snippet": snippet, "struct": struct,
                                   "preds": graph.get_predecessors(aname),
                                   "succs": graph.get_successors(aname)})
            g["frameworks"][meta.get("framework","?")] += 1
            g["benchmarks"][meta.get("benchmark","?")] += 1
            g["topologies"][graph.topology] += 1
            g["injections"][inj] += 1

    schemas = []
    for (ecode, role, phase), g in groups.items():
        exs = g["examples"]
        freq = len(exs)
        if freq < min_freq: continue

        edesc = ERROR_DESCRIPTIONS.get(ecode, ecode)
        ecat = ERROR_CATEGORY.get(ecode, "?")
        top_fw = g["frameworks"].most_common(1)[0][0]
        top_topo = g["topologies"].most_common(1)[0][0]
        top_inj = g["injections"].most_common(1)[0][0]
        bench_dist = ", ".join(f"{b}:{c}" for b,c in g["benchmarks"].most_common(3))
        fw_dist = ", ".join(f"{f}:{c}" for f,c in g["frameworks"].most_common(3))

        has_preds = sum(1 for e in exs if e["preds"]) / max(freq,1)
        has_succs = sum(1 for e in exs if e["succs"]) / max(freq,1)

        sig = f"A {role} agent (phase: {phase}) commits {ecode}: {edesc}. Category: {ecat}."
        if top_inj == "prompt_injection":
            sig += " Typically caused by corrupted input/instructions."
        else:
            sig += " Typically manifests as corrupted agent output."
        if has_succs > 0.5:
            sig += " Agent has downstream dependents — errors may propagate."

        # Distinguishing features
        same_role = sorted(set(ec for (ec,r,_) in groups if r == role and ec != ecode))
        dist = f"This is {ecode}, NOT {', '.join(same_role[:4])}. " if same_role else f"This is {ecode}. "
        dist += DISTINGUISH_HINT.get(ecode, "")

        snippets = sorted([e["snippet"] for e in exs if e["snippet"]], key=len)
        example = snippets[0][:400] if snippets else ""
        structs = [e["struct"] for e in exs if e["struct"]]
        struct = structs[0][:300] if structs else ""

        sid = hashlib.md5(f"{ecode}_{role}_{phase}".encode()).hexdigest()[:12]
        schemas.append(ErrorSchema(
            schema_id=sid, error_code=ecode, error_description=edesc,
            agent_role=role, agent_phase=phase,
            framework_pattern=fw_dist, topology_pattern=top_topo,
            injection_strategy=top_inj, signature=sig,
            structural_context=struct, distinguishing_features=dist,
            example_snippet=example, frequency=freq,
            benchmark_distribution=bench_dist,
        ))

    schemas.sort(key=lambda s: s.frequency, reverse=True)
    logger.info(f"Built {len(schemas)} schemas from {len(groups)} groups")
    return schemas


# ============================================================
# Part 3: Schema Retrieval
# ============================================================

class SchemaRetriever:
    def __init__(self, schemas: List[ErrorSchema]):
        self.schemas = schemas
        self.schema_words = {}
        for s in schemas:
            text = f"{s.signature} {s.example_snippet}".lower()
            self.schema_words[s.schema_id] = Counter(re.findall(r'\w+', text))

    def retrieve(self, graph: AgentDependencyGraph, history: List[Dict],
                 metadata: Dict = None, top_k: int = 5) -> List[ErrorSchema]:
        fw = (metadata or {}).get("framework","")
        roles = set(infer_role(a, graph.agent_phases.get(a, set()),
                              history=history, graph=graph) for a in graph.agents)
        query = Counter(re.findall(r'\w+', " ".join(safe_content(h, 200) for h in history).lower()))

        # Build role token sets for fuzzy matching
        # e.g. roles = {"code_production+entry_point", "evaluation"}
        # role_tokens = {"code", "production", "entry", "point", "evaluation"}
        role_tokens = set()
        for r in roles:
            role_tokens.update(re.findall(r'[a-z]+', r.lower()))

        scored = []
        for s in self.schemas:
            score = 0.0
            if fw and fw in s.framework_pattern: score += 3.0
            if graph.topology == s.topology_pattern: score += 2.0
            
            # Fuzzy role matching: check token overlap between schema role and trajectory roles
            schema_role_tokens = set(re.findall(r'[a-z]+', s.agent_role.lower()))
            role_overlap = len(role_tokens & schema_role_tokens)
            if role_overlap > 0:
                score += min(role_overlap * 1.2, 3.0)  # up to 3.0 for strong match
            
            score += min(s.frequency / 100.0, 1.0)
            sw = self.schema_words.get(s.schema_id, Counter())
            common = set(query) & set(sw)
            if common:
                dot = sum(query[w]*sw[w] for w in common)
                mq = sum(v**2 for v in query.values())**.5
                ms = sum(v**2 for v in sw.values())**.5
                if mq > 0 and ms > 0: score += 2.0 * dot/(mq*ms)
            scored.append((score, s))
        scored.sort(key=lambda x: -x[0])

        selected = []; codes = set(); roles_seen = set()
        for score, s in scored:
            if len(selected) >= top_k: break
            pen = 1.0
            if s.error_code in codes: pen *= 0.5
            if s.agent_role in roles_seen: pen *= 0.7
            if score * pen > 0 or len(selected) < 3:
                selected.append(s)
                codes.add(s.error_code); roles_seen.add(s.agent_role)
        return selected

    def format_for_prompt(self, schemas: List[ErrorSchema], max_chars=3000) -> str:
        if not schemas: return ""
        lines = ["## REFERENCE ERROR PATTERNS (similar past cases — not all may apply)\n"]
        chars = len(lines[0])
        for i, s in enumerate(schemas):
            block = [
                f"### Pattern {i+1}: {s.error_code} in {s.agent_role} ({s.agent_phase} phase)",
                f"  {s.signature}",
                f"  How to identify: {s.distinguishing_features}",
                "",
            ]
            bt = "\n".join(block)
            if chars + len(bt) > max_chars: break
            lines.extend(block); chars += len(bt)
        return "\n".join(lines)


# ============================================================
# Part 4: Augmented Prompt Builder
# ============================================================

def build_augmented_prompt(sample: Dict, retriever: SchemaRetriever, base_prompt: str) -> Dict:
    inp = sample.get("input",{})
    meta = sample.get("metadata",{})
    history = inp.get("conversation_history",[])
    query = inp.get("query","")

    graph = build_graph(history, meta)
    schemas = retriever.retrieve(graph, history, meta, top_k=5)

    # Build conversation text
    conv = []
    for e in history:
        conv.append(f"[Step {e.get('step','')}] [{e.get('agent_name','')}] (phase: {e.get('phase','')})")
        conv.append(safe_content(e))
        conv.append("")
    conv_text = "\n".join(conv)

    # Assemble augmented context
    parts = [
        "## AGENT DEPENDENCY GRAPH",
        graph.serialize(), "",
        "## ERROR PROPAGATION GUIDE",
        graph.get_propagation_hints(), "",
        retriever.format_for_prompt(schemas),
        "## TASK QUERY",
        query[:500] if query else "(no query)", "",
        "## CONVERSATION LOG",
        conv_text,
    ]
    augmented = "\n".join(parts)

    try:
        prompt = base_prompt.replace("{conversation_text}", augmented)
    except:
        prompt = base_prompt + "\n\n" + augmented

    return {
        "id": sample.get("id",""),
        "augmented_prompt": prompt,
        "graph_info": {"topology": graph.topology, "framework": graph.framework,
                       "agents": graph.agents, "num_edges": len(graph.edges)},
        "retrieved_schema_ids": [s.schema_id for s in schemas],
        "retrieved_schema_codes": [s.error_code for s in schemas],
        "metadata": meta,
        "output": sample.get("output",{}),
    }


# ============================================================
# CLI
# ============================================================

def cmd_build_schemas(args):
    samples = []
    with open(args.train_data, 'r') as f:
        for line in f:
            if line.strip():
                try: samples.append(json.loads(line))
                except: continue
    logger.info(f"Loaded {len(samples)} samples")
    schemas = build_schemas(samples, args.min_freq)
    with open(args.output, 'w') as f:
        json.dump({"num_schemas": len(schemas), "schemas": [asdict(s) for s in schemas]}, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(schemas)} schemas to {args.output}")

    print(f"\n{'='*70}")
    print(f"SCHEMA SUMMARY: {len(schemas)} schemas from {len(samples)} samples")
    print(f"{'='*70}")
    by_error = defaultdict(list)
    for s in schemas: by_error[s.error_code].append(s)
    for code in ALL_ERROR_CODES:
        ss = by_error.get(code,[])
        total = sum(s.frequency for s in ss)
        roles = sorted(set(s.agent_role for s in ss))
        print(f"  {code}: {len(ss):2d} schemas, {total:5d} instances | roles: {roles}")

def cmd_prepare_inference(args):
    with open(args.schemas) as f: sd = json.load(f)
    schemas = [ErrorSchema(**s) for s in sd["schemas"]]
    retriever = SchemaRetriever(schemas)
    with open(args.prompt_template) as f: base_prompt = f.read()
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                try: samples.append(json.loads(line))
                except: continue
    logger.info(f"Processing {len(samples)} samples with {len(schemas)} schemas")
    results = []
    for i, s in enumerate(samples):
        results.append(build_augmented_prompt(s, retriever, base_prompt))
        if (i+1) % 100 == 0: logger.info(f"  {i+1}/{len(samples)}")
    with open(args.output, 'w') as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    logger.info(f"Saved {len(results)} augmented prompts to {args.output}")
    topos = Counter(r["graph_info"]["topology"] for r in results)
    print(f"\nStats: {len(results)} samples, topologies={dict(topos)}")

def cmd_analyze(args):
    with open(args.schemas) as f: sd = json.load(f)
    schemas = [ErrorSchema(**s) for s in sd["schemas"]]
    roles = sorted(set(s.agent_role for s in schemas))
    print(f"{'Code':<8}" + "".join(f"{r:<12}" for r in roles))
    print("-"*80)
    for code in ALL_ERROR_CODES:
        row = f"{code:<8}"
        for role in roles:
            m = [s for s in schemas if s.error_code==code and s.agent_role==role]
            row += f"{sum(s.frequency for s in m) if m else '---':<12}"
        print(row)

def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd")
    p1 = sp.add_parser("build-schemas")
    p1.add_argument("--train_data", required=True); p1.add_argument("--output", default="schemas.json"); p1.add_argument("--min_freq", type=int, default=1)
    p2 = sp.add_parser("prepare-inference")
    p2.add_argument("--input", required=True); p2.add_argument("--schemas", required=True); p2.add_argument("--prompt_template", required=True); p2.add_argument("--output", default="augmented.jsonl")
    p3 = sp.add_parser("analyze"); p3.add_argument("--schemas", required=True)
    args = p.parse_args()
    if args.cmd == "build-schemas": cmd_build_schemas(args)
    elif args.cmd == "prepare-inference": cmd_prepare_inference(args)
    elif args.cmd == "analyze": cmd_analyze(args)
    else: p.print_help()

if __name__ == "__main__": main()