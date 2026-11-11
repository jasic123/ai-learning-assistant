// 把模型常见的数学标记转成不依赖公式字体的可读文本。
export function readableText(value) {
  if (typeof value !== "string") return value;
  let text = value
    .replace(/\\?int\s*_\s*\{([^{}]*)\}\s*\^\s*\{([^{}]*)\}/gi,
      (_, lower, upper) => `∫${toSubscript(lower)}${toSuperscript(upper)}`)
    .replace(/\bquad\b/g, " ")
    .replace(/\^\s*\{([^{}]*)\}/g, (_, exponent) => toSuperscript(exponent))
    .replace(/\^\s*([A-Za-z0-9+-]+)/g, (_, exponent) => toSuperscript(exponent))
    .replace(/_\s*\{([^{}]*)\}/g, (_, subscript) => toSubscript(subscript))
    .replace(/e([⁻⁺ᵗˣⁿ][ᵗˣⁿ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]*)/g, (_, exponent) => {
      const sign = exponent.startsWith("⁻") ? "-" : exponent.startsWith("⁺") ? "+" : "";
      const body = exponent.replace(/^[⁻⁺]/, "");
      const plain = body.replace(/[ᵗˣⁿ]/g, (char) => ({ "ᵗ": "t", "ˣ": "x", "ⁿ": "n" }[char]));
      return `e^(${sign}${plain})`;
    })
    .replace(/\\boxed\s*\{([^{}]*)\}/g, "$1")
    .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1) / ($2)")
    .replace(/\\sqrt\s*\{([^{}]*)\}/g, "√($1)")
    .replace(/\\(?:approx|sim)/g, "约等于")
    .replace(/\\neq/g, "≠")
    .replace(/\\leq?/g, "≤")
    .replace(/\\geq?/g, "≥")
    .replace(/\\times/g, "×")
    .replace(/\\cdot/g, "·")
    .replace(/\\infty/g, "∞")
    .replace(/\\to/g, "→")
    .replace(/\\(?:,|;|!)/g, "")
    .replace(/\\(?:left|right|mathrm|text)\b/g, "")
    .replace(/\\[()[\]]/g, "")
    .replace(/\\{([^{}]*)\}/g, "$1")
    .replace(/\\([a-zA-Z]+)/g, "$1")
    .replace(/\$\$?([^$]+)\$\$?/g, "$1")
    .replace(/\\\[|\\\]/g, "")
    .replace(/[\uF028\uF029\uF02B\uF02D\uF03D\uF0A2\uF0F2\uF067]/g, (char) => ({
      "\uF028": "(", "\uF029": ")", "\uF02B": "+", "\uF02D": "-",
      "\uF03D": "=", "\uF0A2": "′", "\uF0F2": "∫", "\uF067": "γ",
    }[char] || char));
  return text;
}

const superscripts = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
  "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻",
  "(": "⁽", ")": "⁾", "=": "⁼", "x": "ˣ", "t": "ᵗ", "n": "ⁿ" };
const subscripts = { "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
  "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋",
  "(": "₍", ")": "₎", "x": "ₓ", "t": "ₜ", "n": "ₙ" };

function toSuperscript(value) {
  return [...value.replace(/\^/g, "")].map((char) => superscripts[char] || char).join("");
}

function toSubscript(value) {
  return [...value].map((char) => subscripts[char] || char).join("");
}
