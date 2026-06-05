function parseValue(raw: string): unknown {
  const value = raw.trim();
  if (!value) {
    return "";
  }
  if (value === "true") {
    return true;
  }
  if (value === "false") {
    return false;
  }
  const number = Number(value);
  if (!Number.isNaN(number) && value !== "") {
    return number;
  }
  return value;
}

export function parseParams(text: string): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  for (const [lineNumber, rawLine] of text.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const equals = line.indexOf("=");
    if (equals <= 0) {
      throw new Error(`Invalid inference parameter on line ${lineNumber + 1}. Expected key=value.`);
    }
    const key = line.slice(0, equals).trim();
    const value = line.slice(equals + 1);
    if (!key) {
      throw new Error(`Invalid inference parameter on line ${lineNumber + 1}. Key is empty.`);
    }
    params[key] = parseValue(value);
  }
  return params;
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const units = ["B", "KiB", "MiB", "GiB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
