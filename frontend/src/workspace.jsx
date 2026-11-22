import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Button, IconButton, Switch, TextField, Theme } from "@radix-ui/themes";
import {
  Activity,
  AudioLines,
  BookOpen,
  ChartNoAxesCombined,
  ChartSpline,
  ChevronDown,
  CircleDot,
  CircleGauge,
  FileText,
  FolderOutput,
  Moon,
  PanelRight,
  RefreshCw,
  ScanLine,
  Settings2,
  Sun,
  Trash2,
  Waves,
  Waypoints,
} from "lucide-react";
import "@radix-ui/themes/tokens.css";
import "@radix-ui/themes/components.css";
import "./workspace.css";

const viewIcons = [
  Activity,
  ChartNoAxesCombined,
  Waypoints,
  ChartSpline,
  CircleGauge,
  AudioLines,
  Waves,
];
const groupAt = { 0: "Signal", 1: "Standards", 5: "Reversals" };
const formatIcons = [AudioLines, FileText, Waypoints];
const fileActions = [
  ["Export all", FolderOutput],
  ["Word report", BookOpen],
  ["Clear all", Trash2],
];

function NumberField({ label, value, min, max, step = 1, unit, change }) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  function commit(e) {
    const n = e.target.valueAsNumber;
    if (Number.isFinite(n) && e.target.validity.valid) change(n);
    else setDraft(String(value));
  }
  return (
    <label className="field">
      <span>{label}</span>
      <TextField.Root
        type="number"
        size="1"
        min={min}
        max={max}
        step={step}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
        }}
        onBlur={commit}
        aria-label={label}
      >
        {unit && <TextField.Slot side="right">{unit}</TextField.Slot>}
      </TextField.Root>
    </label>
  );
}

function SelectField({ label, values, value, change, compact = false }) {
  return (
    <label className={"field " + (compact ? "source" : "")}>
      <span>{label}</span>
      <span className="select-wrap">
        <select
          value={value}
          disabled={!values.length}
          title={values[value]}
          aria-label={label}
          onChange={(e) => change(Number(e.target.value))}
        >
          {!values.length && <option value={-1}>Select a recording</option>}
          {values.map((v, i) => (
            <option key={i} value={i}>
              {v}
            </option>
          ))}
        </select>
        <ChevronDown size={13} />
      </span>
    </label>
  );
}

function Workspace({ bus, state: s }) {
  const slot = useRef(null);
  const set = (name, value) => bus.setting(name, String(value));
  const act = (name) => bus.command(name);
  useLayoutEffect(() => {
    if (!slot.current) return;
    const element = slot.current;
    function measure() {
      const r = element.getBoundingClientRect();
      bus.chartArea(
        Math.round(r.x),
        Math.round(r.y),
        Math.round(r.width),
        Math.round(r.height),
        innerWidth,
        innerHeight,
      );
    }
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    window.addEventListener("resize", measure);
    measure();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [bus]);

  return (
    <Theme
      appearance={s.dark ? "dark" : "light"}
      accentColor="teal"
      grayColor="sage"
      radius="large"
    >
      <div className="workspace">
        <aside className="rail">
          <div className="brand">
            <span className="brand-mark">
              <AudioLines size={25} />
            </span>
            <strong>
              BERT<span>Signal Lab</span>
            </strong>
          </div>
          <nav aria-label="Analysis views">
            {s.views.map((title, i) => {
              const Icon = viewIcons[i];
              return (
                <React.Fragment key={title}>
                  {groupAt[i] && <h2>{groupAt[i]}</h2>}
                  <button
                    className={"view " + (s.view === i ? "active" : "")}
                    aria-current={s.view === i ? "page" : undefined}
                    onClick={() => bus.navigate(i)}
                  >
                    <Icon size={17} />
                    <span>{title}</span>
                  </button>
                </React.Fragment>
              );
            })}
          </nav>
          <section className="settings">
            <h2>
              <Settings2 size={14} /> Analysis settings
            </h2>
            <SelectField
              label="Restraint"
              values={s.restraints}
              value={s.restraint}
              change={(v) => set("restraint", v)}
            />
            <SelectField
              label="Condition"
              values={s.conditions}
              value={s.condition}
              change={(v) => set("condition", v)}
            />
            <div className="pair">
              <NumberField
                label="Height"
                unit="in"
                min={0}
                max={80}
                value={s.height}
                change={(v) => set("height", v)}
              />
              <NumberField
                label="Trend window"
                unit="s"
                min={0.001}
                max={60}
                step={0.001}
                value={s.period}
                change={(v) => set("period", v)}
              />
            </div>
            <label className="switch-field">
              Prone posture
              <Switch
                size="1"
                checked={s.prone}
                onCheckedChange={(v) => set("prone", v)}
              />
            </label>
          </section>
        </aside>
        <main>
          <header>
            <div className="chart-title">
              <h1>{s.view === 0 ? "Signal timeline" : s.views[s.view]}</h1>
              <span className={"status " + s.state} role="status">
                <CircleDot size={13} />
                {s.status}
              </span>
            </div>
            <div className="actions">
              <div className="file-actions" role="group" aria-label="Workspace files">
                {fileActions.map(([name, Icon]) => (
                  <Button
                    key={name}
                    size="1"
                    variant="ghost"
                    color="gray"
                    disabled={!s.files[name]}
                    onClick={() => act(name)}
                  >
                    <Icon size={14} />
                    {name}
                  </Button>
                ))}
              </div>
              <Button
                variant={s.library ? "soft" : "ghost"}
                color="gray"
                aria-label="Toggle recording library"
                aria-pressed={s.library}
                onClick={() => act("library")}
              >
                <PanelRight size={16} />
                Library
              </Button>
              <IconButton
                variant="soft"
                color="gray"
                aria-label="Toggle appearance"
                title="Toggle appearance"
                onClick={() => act("theme")}
              >
                {s.dark ? <Sun size={17} /> : <Moon size={17} />}
              </IconButton>
            </div>
          </header>
          <div className="command-panel">
            <div className="file-controls" role="group" aria-label="Import recording">
              <span className="group-label">Import</span>
              {s.formats.map((name, i) => {
                const Icon = formatIcons[i];
                return (
                  <Button
                    className="format"
                    key={name}
                    size="1"
                    variant="soft"
                    disabled={!s.canImport}
                    onClick={() => bus.openFormat(i)}
                  >
                    <Icon size={14} />
                    {name}
                  </Button>
                );
              })}
            </div>
            <div className="controls">
              <SelectField
                label="Recording"
                compact
                values={s.sources}
                value={s.source}
                change={(v) => set("source", v)}
              />
              <NumberField
                label="Low-pass"
                unit="Hz"
                min={1}
                max={249}
                value={s.cutoff}
                change={(v) => set("cutoff", v)}
              />
              <Button
                size="2"
                variant="soft"
                disabled={!s.update}
                onClick={() => act("update")}
              >
                <RefreshCw size={14} />
                Update
              </Button>
              <Button
                size="2"
                variant={s.analyzing ? "solid" : "outline"}
                disabled={!s.canAnalyze}
                aria-pressed={s.analyzing}
                onClick={() => act("analyze")}
              >
                <ScanLine size={14} />
                Analyze
              </Button>
            </div>
          </div>
          <div
            className="chart-slot"
            ref={slot}
            aria-label="Interactive analysis chart and recording library"
          />
        </main>
      </div>
    </Theme>
  );
}

const root = createRoot(document.getElementById("root"));
new QWebChannel(qt.webChannelTransport, (channel) => {
  const bus = channel.objects.workspace;
  const render = (s) => root.render(<Workspace bus={bus} state={s} />);
  bus.changed.connect(render);
  bus.snapshot(render);
});
