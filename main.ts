// free-model-aggregator: one server, two client wires, ten free providers.
//
// The trial gives this server a brief (MAKEFUN_PROMPT_FILE) and an env
// (MAKEFUN_ENV_URL). The brief carries three JSON documents: a provider
// directory (ten simulated free-tier providers, each with its wire, auth
// scheme, key variable, key and base path), a quality table (every free
// model's rating and capabilities) and a workload manifest (the twenty
// requests the client will send, in order, with each one's requirements).
//
// The server probes every provider's catalog route, builds one catalog,
// registers it with the env, and then answers the client's requests:
// POST /v1/chat/completions (OpenAI wire) and POST /v1/messages
// (Anthropic wire), JSON or server-sent events, plus GET /v1/models in
// the shape of the auth header it got. Every answer comes from a free
// model on one of the ten providers, through the env; there is no model
// of its own. `node main.ts --list` prints the catalog and exits.
//
// Routing: for each request the server ranks every free model that meets
// the request's requirements by the quality table's rating, then discounts
// a provider for every call it has already made and every refusal it has
// given, so traffic spreads over the providers instead of hammering the
// single best one until its per-trial budget is spent. A refusal (402,
// 429, 5xx, timeout) is never the answer: the request is retried on the
// next-best candidate, and once every candidate has refused, the best of
// them is tried once more, because a provider's failure schedule is
// per-call and a retry may land. Exactly one successful provider call
// answers each request.
//
// Wires: the client wire and the provider wire are independent. The
// request is translated into the provider's wire without dropping or
// rewriting a field the provider echoes back (system prompt, user text,
// images, tools, stop sequences, max_tokens, JSON mode), and the answer
// is translated back into the client's wire with the provider's text
// whole, its usage counts as that wire reports them, and its stop reason
// mapped to the client's family. JSON-mode requests on the OpenAI wire
// only go to a wire that can carry response_format (OpenAI or Gemini);
// the Anthropic wire cannot, so it is never picked for those.
//
// Determinism: every choice is a function of the brief, the requests
// seen so far and the providers' answers. No wall clock and no unseeded
// randomness anywhere in a response or in the catalog; the fixed
// timestamps below are the same on every run, live or replayed.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** The Anthropic version the client sends and the providers expect. */
const ANTHROPIC_VERSION = '2023-06-01';
/** How long one provider call may take before it counts as a refusal. */
const PROVIDER_TIMEOUT_MS = 3000;
/** How long one catalog probe may take, per attempt. */
const PROBE_TIMEOUT_MS = 5000;
/** Catalog probe attempts before a provider is left out of the catalog. */
const PROBE_ATTEMPTS = 3;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most upstream attempts for one client request. */
const MAX_ATTEMPTS = 8;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: Provider[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): Provider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, key_env: asString(v.key_env), key: asString(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(v.rating_milli) ?? 0,
    context_length: asNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') {
    empty.error = 'MAKEFUN_PROMPT_FILE is not set';
    return empty;
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    empty.error = `cannot read ${path}: ${(err as Error).message}`;
    return empty;
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    empty.error = 'the brief holds no provider directory document';
    return empty;
  }
  empty.providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)) {
    empty.ratings = ratingsDoc.ratings.flatMap((v) => {
      const r = ratingOf(v);
      return r === null ? [] : [r];
    });
    if (Array.isArray(ratingsDoc.best)) {
      for (const b of ratingsDoc.best) {
        if (isRecord(b) && asString(b.id) !== '') empty.best.set(asString(b.id), asNumber(b.rating_milli) ?? 0);
      }
    }
  }
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  if (workloadDoc !== null && Array.isArray(workloadDoc.manifest)) {
    empty.manifest = workloadDoc.manifest.flatMap((v) => {
      const w = workloadOf(v);
      return w === null ? [] : [w];
    });
  }
  return empty;
}

// ---------------------------------------------------------------------------
// The catalog probe
// ---------------------------------------------------------------------------

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: Provider;
  ok: boolean;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A price field of a listing: true when every price is zero, null when the listing says nothing about price. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  return {
    id,
    display_name:
      asString(v.display_name) !== ''
        ? asString(v.display_name)
        : asString(v.displayName) !== ''
          ? asString(v.displayName)
          : id,
    free: asBool(v.free) ?? zeroPriced(v),
    context_length: asNumber(v.context_length) ?? asNumber(v.context_window) ?? asNumber(v.inputTokenLimit),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

async function probeOnce(envUrl: string, provider: Provider): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

async function probeProvider(envUrl: string, provider: Provider): Promise<Probe> {
  let last: Probe = { provider, ok: false, models: [], error: 'no probe attempt ran' };
  for (let attempt = 0; attempt < PROBE_ATTEMPTS; attempt++) {
    last = await probeOnce(envUrl, provider);
    if (last.ok) return last;
  }
  return last;
}

function ratingRowOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table's row where it has one, the listing's own word otherwise. */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What a model can do: the quality table's row where it has one, the listing's fields otherwise. */
function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

const byModelId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The catalog the probes and the quality table make: every provider that answered, every model it listed. */
function buildCatalog(brief: Brief, probes: readonly Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// The client wires: request extraction
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One image part as its data URL or plain URL, plus its media type when it is a data URL. */
interface ImageRef {
  url: string;
  media_type: string | null;
  data: string | null;
}

function imageRefOfOpenAI(block: Json): ImageRef | null {
  if (!isRecord(block.image_url)) return null;
  const url = asString(block.image_url.url);
  if (url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { url, media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] };
  return { url, media_type: null, data: null };
}

function imageRefOfAnthropic(block: Json): ImageRef | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = asString(source.media_type) !== '' ? asString(source.media_type) : 'image/png';
    return { url: `data:${media};base64,${source.data}`, media_type: media, data: source.data };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { url: source.url, media_type: null, data: null };
  return null;
}

/** A tool definition in one shape, whatever wire declared it. */
interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

function toolDefsOfOpenAI(body: Json): ToolDef[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || asString(raw.function.name) === '') return [];
    return [{ name: asString(raw.function.name), description: asString(raw.function.description), parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

function toolDefsOfAnthropic(body: Json): ToolDef[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || asString(raw.name) === '') return [];
    return [{ name: asString(raw.name), description: asString(raw.description), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

/** The fields of a client request that the provider must echo back, in one shape. */
interface RequestEcho {
  systemText: string;
  /** User messages in order: text parts and image parts, as the client sent them. */
  userMessages: { text: string; images: ImageRef[] }[];
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
}

function echoOfOpenAI(body: Json): RequestEcho {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const userMessages: { text: string; images: ImageRef[] }[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = textOf(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      const text = textOf(message.content);
      const images: ImageRef[] = [];
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (isRecord(block) && block.type === 'image_url') {
            const ref = imageRefOfOpenAI(block);
            if (ref !== null) images.push(ref);
          }
        }
      }
      userMessages.push({ text, images });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stop = body.stop;
  return {
    systemText: systemParts.join('\n'),
    userMessages,
    tools: toolDefsOfOpenAI(body),
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    stops: typeof stop === 'string' ? [stop] : Array.isArray(stop) ? stop.filter((s): s is string => typeof s === 'string') : [],
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
  };
}

function echoOfAnthropic(body: Json): RequestEcho {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const userMessages: { text: string; images: ImageRef[] }[] = [];
  for (const message of messages) {
    if (message.role !== 'user') continue;
    const text = textOf(message.content);
    const images: ImageRef[] = [];
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') {
          const ref = imageRefOfAnthropic(block);
          if (ref !== null) images.push(ref);
        }
      }
    }
    userMessages.push({ text, images });
  }
  return {
    systemText: textOf(body.system),
    userMessages,
    tools: toolDefsOfAnthropic(body),
    maxTokens: asNumber(body.max_tokens),
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [],
    jsonMode: false,
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
  };
}

// ---------------------------------------------------------------------------
// The client wires: request translation into a provider's wire
// ---------------------------------------------------------------------------

/** An Anthropic content block for one image reference. */
function anthropicImageBlock(ref: ImageRef): Json {
  if (ref.data !== null) return { type: 'image', source: { type: 'base64', media_type: ref.media_type ?? 'image/png', data: ref.data } };
  return { type: 'image', source: { type: 'url', url: ref.url } };
}

/** An OpenAI content part for one image reference. */
function openAIImagePart(ref: ImageRef): Json {
  return { type: 'image_url', image_url: { url: ref.url } };
}

/** A Gemini content part for one image reference. */
function geminiImagePart(ref: ImageRef): Json {
  if (ref.data !== null) return { inlineData: { mimeType: ref.media_type ?? 'image/png', data: ref.data } };
  return { fileData: { fileUri: ref.url } };
}

/** The provider's generation config for a Gemini call: only the fields the request set. */
function geminiConfig(echo: RequestEcho): Json {
  const config: Json = {};
  if (echo.maxTokens !== null) config.maxOutputTokens = echo.maxTokens;
  if (echo.stops.length > 0) config.stopSequences = echo.stops;
  if (echo.temperature !== null) config.temperature = echo.temperature;
  if (echo.topP !== null) config.topP = echo.topP;
  if (echo.jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/** An OpenAI client request as an Anthropic Messages request naming `model`. */
function openAIToAnthropic(body: Json, echo: RequestEcho, model: string, stream: boolean): Json {
  const out: Json = { model, max_tokens: echo.maxTokens ?? 1024 };
  if (echo.systemText !== '') out.system = echo.systemText;
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const raw of source) {
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) input = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          content.push({ type: 'tool_use', id: asString(call.id) !== '' ? asString(call.id) : 'toolu_unknown', name: asString(call.function.name), input });
        }
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: asString(raw.tool_call_id), content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }] });
      continue;
    }
    // A user message: text and image parts, in the order the client sent them.
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url') {
          const ref = imageRefOfOpenAI(block);
          if (ref !== null) parts.push(anthropicImageBlock(ref));
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  out.messages = messages;
  if (echo.tools.length > 0) {
    out.tools = echo.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: asString(choice.function.name) };
  }
  if (echo.temperature !== null) out.temperature = echo.temperature;
  if (echo.topP !== null) out.top_p = echo.topP;
  if (echo.stops.length > 0) out.stop_sequences = echo.stops;
  if (stream) out.stream = true;
  return out;
}

/** An OpenAI client request as a Gemini generateContent request. */
function openAIToGemini(body: Json, echo: RequestEcho): Json {
  const out: Json = {};
  if (echo.systemText !== '') out.systemInstruction = { parts: [{ text: echo.systemText }] };
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const raw of source) {
    if (raw.role === 'system' || raw.role === 'tool') continue;
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url') {
          const ref = imageRefOfOpenAI(block);
          if (ref !== null) parts.push(geminiImagePart(ref));
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  out.contents = contents;
  if (echo.tools.length > 0) {
    out.tools = [{ functionDeclarations: echo.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config = geminiConfig(echo);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic client request as an OpenAI Chat Completions request naming `model`. */
function anthropicToOpenAI(body: Json, echo: RequestEcho, model: string, stream: boolean): Json {
  const messages: Json[] = [];
  if (echo.systemText !== '') messages.push({ role: 'system', content: echo.systemText });
  const source = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const raw of source) {
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          toolCalls.push({
            id: asString(block.id) !== '' ? asString(block.id) : 'call_unknown',
            type: 'function',
            function: { name: asString(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
          });
        }
      }
      const text = textOf(raw.content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: asString(block.tool_use_id), content: toolResultText(block.content) });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else if (block.type === 'image') {
        const ref = imageRefOfAnthropic(block);
        if (ref !== null) parts.push(openAIImagePart(ref));
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (echo.maxTokens !== null) out.max_tokens = echo.maxTokens;
  if (echo.temperature !== null) out.temperature = echo.temperature;
  if (echo.topP !== null) out.top_p = echo.topP;
  if (echo.stops.length > 0) out.stop = echo.stops;
  if (echo.tools.length > 0) {
    out.tools = echo.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic client request as a Gemini generateContent request. */
function anthropicToGemini(body: Json, echo: RequestEcho): Json {
  const out: Json = {};
  if (echo.systemText !== '') out.systemInstruction = { parts: [{ text: echo.systemText }] };
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const raw of source) {
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image') {
          const ref = imageRefOfAnthropic(block);
          if (ref !== null) parts.push(geminiImagePart(ref));
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  out.contents = contents;
  if (echo.tools.length > 0) {
    out.tools = [{ functionDeclarations: echo.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config = geminiConfig(echo);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// The client wires: provider answers parsed into one shape
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop in its own spelling: stop, end_turn, STOP, length, MAX_TOKENS, ... */
  stop: string;
  id: string;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of an OpenAI message's content, string or blocks. */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

function parseOpenAIAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(body)) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: openAIText(message.content),
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
      id: asString(body.id) !== '' ? asString(body.id) : 'chatcmpl-aggregate',
    };
  }
  let answerText = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  let id = 'chatcmpl-aggregate';
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (asString(chunk.id) !== '') id = asString(chunk.id);
      if (isRecord(chunk.usage)) {
        input = numberOrZero(chunk.usage.prompt_tokens);
        output = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') answerText += delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    }
  }
  return { text: answerText, inputTokens: input, outputTokens: output, stop, id };
}

function parseAnthropicAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(body) || !Array.isArray(body.content)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: textOf(body.content),
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
      id: asString(body.id) !== '' ? asString(body.id) : 'msg_aggregate',
    };
  }
  let answerText = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  let id = 'msg_aggregate';
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    let data = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data = line.slice('data:'.length).replace(/^ /, '');
    }
    if (data === '') continue;
    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(payload)) continue;
    if (event === 'message_start' && isRecord(payload.message)) {
      if (asString(payload.message.id) !== '') id = asString(payload.message.id);
      if (isRecord(payload.message.usage)) input = numberOrZero(payload.message.usage.input_tokens);
    } else if (event === 'content_block_delta' && isRecord(payload.delta) && payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') {
      answerText += payload.delta.text;
    } else if (event === 'message_delta') {
      if (isRecord(payload.delta) && typeof payload.delta.stop_reason === 'string') stop = payload.delta.stop_reason;
      if (isRecord(payload.usage)) output = numberOrZero(payload.usage.output_tokens);
    }
  }
  return { text: answerText, inputTokens: input, outputTokens: output, stop, id };
}

function geminiTextOf(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

function parseGeminiAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(body)) return null;
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text: geminiTextOf(content.parts),
      inputTokens: numberOrZero(usage.promptTokenCount),
      outputTokens: numberOrZero(usage.candidatesTokenCount),
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
      id: asString(body.modelVersion) !== '' ? asString(body.modelVersion) : 'gemini-aggregate',
    };
  }
  let answerText = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  let id = 'gemini-aggregate';
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (asString(chunk.modelVersion) !== '') id = asString(chunk.modelVersion);
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      if (candidate !== null) {
        if (isRecord(candidate.content)) answerText += geminiTextOf(candidate.content.parts);
        if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
      }
      if (isRecord(chunk.usageMetadata)) {
        input = numberOrZero(chunk.usageMetadata.promptTokenCount);
        output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
      }
    }
  }
  return { text: answerText, inputTokens: input, outputTokens: output, stop, id };
}

/** A provider stop reason's family: 0 stop, 1 length, 2 tool calls, 3 refusal. */
function stopFamily(stop: string): number {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 1;
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 2;
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 3;
  return 0;
}

/** The provider's stop reason in OpenAI's spelling. */
function openAIStop(stop: string): string {
  const family = stopFamily(stop);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** The provider's stop reason in Anthropic's spelling. */
function anthropicStop(stop: string): string {
  const family = stopFamily(stop);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** Split text into the 13-character pieces the providers stream in. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A provider answer as an OpenAI Chat Completions object. */
function renderOpenAIJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: answer.id,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStop(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A provider answer as an OpenAI event stream: role chunk, content deltas, finish chunk, usage chunk, [DONE]. */
function renderOpenAIStream(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: answer.id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: answer.id,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A provider answer as an Anthropic Messages object. */
function renderAnthropicJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: answer.id.startsWith('msg_') ? answer.id : `msg_${answer.id}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A provider answer as the Anthropic event sequence: message_start, content blocks, message_delta, message_stop. */
function renderAnthropicStream(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: answer.id.startsWith('msg_') ? answer.id : `msg_${answer.id}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Routing: rank candidates, spread load, fail over
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far in this trial, by provider id. */
const callsMade = new Map<string, number>();
/** Provider refusals seen so far in this trial, by provider id. */
const refusals = new Map<string, number>();
/** Models a provider refused as unknown or paid, never tried again this trial. */
const badModels = new Set<string>();

function loadOf(providerId: string): number {
  return callsMade.get(providerId) ?? 0;
}

function refusalsOf(providerId: string): number {
  return refusals.get(providerId) ?? 0;
}

interface Candidate {
  provider: Provider;
  model: ProbedModel;
  rating: number;
  score: number;
}

/**
 * The free models that can serve one request, best first. Each provider
 * contributes its best model for the request; a provider is discounted
 * for every call it has already made, scaled by its conservative
 * per-trial capacity (the smallest budget the generator draws for it),
 * and for every refusal it has given, so traffic spreads over providers
 * instead of spending one provider's whole per-trial budget on the first
 * requests. A JSON-mode request on the OpenAI wire never goes to an
 * Anthropic-wire provider: that wire cannot carry response_format, so
 * the provider would echo it absent.
 */
const PROVIDER_CAPACITY: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** A provider's conservative per-trial capacity: its smallest drawn budget, 6 for one the table does not name. */
function capacityOf(providerId: string): number {
  return PROVIDER_CAPACITY[providerId] ?? 6;
}

function rankCandidates(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    let best: { model: ProbedModel; rating: number } | null = null;
    for (const model of probe.models) {
      if (badModels.has(`${probe.provider.id}\0${model.id}`)) continue;
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      if (best === null || rating > best.rating || (rating === best.rating && model.id < best.model.id)) {
        best = { model, rating };
      }
    }
    if (best === null) continue;
    const score = best.rating - (60 * loadOf(probe.provider.id)) / capacityOf(probe.provider.id) - 20 * refusalsOf(probe.provider.id);
    out.push({ provider: probe.provider, model: best.model, rating: best.rating, score });
  }
  out.sort((a, b) => (b.score !== a.score ? b.score - a.score : a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0));
  return out;
}

/** One model the entry serves, as GET /v1/models lists it. */
interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byModelId)) {
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
}

async function callUpstream(provider: Provider, wire: Wire, requestBody: Json, model: string, stream: boolean): Promise<UpstreamResult> {
  const base = joinBase(ENV_URL, provider.base);
  let url: string;
  if (wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (wire === 'anthropic') url = `${base}/v1/messages`;
  else url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}${stream ? '?alt=sse' : ''}`;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  if (stream) headers.accept = 'text/event-stream';
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
    return { ok: res.ok, status: res.status, text: await res.text() };
  } catch {
    return { ok: false, status: 0, text: '' };
  }
}

/** The upstream request body for one candidate, in the provider's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, echo: RequestEcho, provider: Provider, model: string, stream: boolean): Json {
  if (clientApi === 'openai') {
    if (provider.wire === 'openai') {
      const out: Json = { ...body, model, stream };
      if (stream) out.stream_options = { include_usage: true };
      return out;
    }
    if (provider.wire === 'anthropic') return openAIToAnthropic(body, echo, model, stream);
    return openAIToGemini(body, echo);
  }
  if (provider.wire === 'openai') return anthropicToOpenAI(body, echo, model, stream);
  if (provider.wire === 'anthropic') {
    const out: Json = { ...body, model, stream };
    return out;
  }
  return anthropicToGemini(body, echo);
}

function parseProviderAnswer(wire: Wire, text: string, stream: boolean): ProviderAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(text, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

interface ServedAnswer {
  answer: ProviderAnswer;
  model: string;
}

/**
 * Serve one client request: try candidates in ranked order, count every
 * call and every refusal, and never answer a refusal. A 2xx ends the
 * search at once (a second success would break the one-provider rule);
 * once every candidate has refused, the best of them is tried once more,
 * because a provider's failure schedule is per-call and a retry may land.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, echo: RequestEcho, requires: Requirement, stream: boolean): Promise<ServedAnswer | null> {
  const tried = new Set<string>();
  let unparsable: ServedAnswer | null = null;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const candidates = rankCandidates(requires, clientApi, echo.jsonMode);
    if (candidates.length === 0) return null;
    let candidate = candidates.find((c) => !tried.has(c.provider.id)) ?? null;
    if (candidate === null) {
      // Every candidate has refused once: retry the best of them.
      candidate = candidates[0];
      if (candidate === undefined) return null;
    }
    tried.add(candidate.provider.id);
    const requestBody = upstreamBody(clientApi, body, echo, candidate.provider, candidate.model.id, stream);
    const result = await callUpstream(candidate.provider, candidate.provider.wire, requestBody, candidate.model.id, stream);
    callsMade.set(candidate.provider.id, loadOf(candidate.provider.id) + 1);
    if (result.ok) {
      const answer = parseProviderAnswer(candidate.provider.wire, result.text, stream);
      if (answer === null) {
        // A 2xx that cannot be read: do not call another provider (its
        // success is already logged), just report the request unserved.
        log(`the ${candidate.provider.id} answer for ${candidate.model.id} could not be read`);
        unparsable = null;
        return null;
      }
      return { answer, model: candidate.model.id };
    }
    refusals.set(candidate.provider.id, refusalsOf(candidate.provider.id) + 1);
    if (result.status === 400) badModels.add(`${candidate.provider.id}\0${candidate.model.id}`);
    log(`${candidate.provider.id}/${candidate.model.id} refused with HTTP ${result.status}`);
  }
  return unparsable;
}

// ---------------------------------------------------------------------------
// HTTP: the two client APIs and the model list
// ---------------------------------------------------------------------------

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.manifest;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  const echo = clientApi === 'openai' ? echoOfOpenAI(body) : echoOfAnthropic(body);
  const vision = echo.userMessages.some((m) => m.images.length > 0);
  return { tools: echo.tools.length > 0, vision, json: echo.jsonMode };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const echo = echoOfOpenAI(body);
  const stream = body.stream === true;
  const served = await serve('openai', body, echo, requirementsFor(body, 'openai'), stream);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  if (stream) sendText(res, 200, 'text/event-stream; charset=utf-8', renderOpenAIStream(served.answer, served.model));
  else sendJson(res, 200, renderOpenAIJson(served.answer, served.model));
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const echo = echoOfAnthropic(body);
  const stream = body.stream === true;
  const served = await serve('anthropic', body, echo, requirementsFor(body, 'anthropic'), stream);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  if (stream) sendText(res, 200, 'text/event-stream; charset=utf-8', renderAnthropicStream(served.answer, served.model));
  else sendJson(res, 200, renderAnthropicJson(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------------------
// Register, poll, start
// ---------------------------------------------------------------------------

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(brief, probes);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError = 'no attempt ran';
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) {
        lastError = `POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`;
      } else {
        const reply: unknown = await res.json();
        if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
        return;
      }
    } catch (err) {
      lastError = (err as Error).message;
    }
    await sleep(200);
  }
  throw new Error(`register failed: ${lastError}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
