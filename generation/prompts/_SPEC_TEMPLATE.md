# <Asset name>

Short one-liner describing the asset and why it exists (e.g. "A golden
medallion with a star emblem, used as a collectible prop.")

## Type & knobs

| Key | Value |
|---|---|
| type | `prop` / `weapon` / `armor` / `background` |
| canvas | `768` (prop/weapon/armor) or `1024` (background) |
| strength | ControlNet strength used for the guide |
| guidance | Flux guidance scale |
| steps | sampling steps |
| denoise | `1.0` |

## Subject

The one-line thing being drawn, used to fill `<subject>` in the LoRA formula.

## Positive

The assembled positive prompt (LoRA formula + this asset's detail). Keep it
short and concrete; the shared style block carries the 2D/task framing.

```
GRPZA, <subject>, <type/context>, white background, game asset
```

## Negative

The negative prompt (shared block + type block + subject-specific closes).
List what this asset must NOT generate (hands/characters, multiple items,
shadows, secondary objects).

## Iterations

Reverse-chronological log. Each entry: prompt change → result → verdict.

### `<date>` — summary
- **Changed:** what was edited
- **Prompt:** the exact positive used
- **Result:** what the imagelooked like (edges, background, fidelity to guide)
- **Verdict:** 👍 reliable / 👎 degrade / 🔁 iterate → next change