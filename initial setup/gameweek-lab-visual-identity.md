# The Gameweek Lab — Visual Identity Guide

Reference doc for the Project knowledge base. Use these specs whenever generating visuals (Artifacts) for posts, so every piece stays consistent.

## Color Palette

| Role | Hex | Use |
|---|---|---|
| Background | `#0B0E14` | Main post background (near-black, slightly blue) |
| Panel / card background | `#11151F` | Stat boxes, secondary containers |
| Grid line | `#1E2430` | Subtle background grid texture, borders |
| Accent (primary) | `#7CFF6B` | Lime green — highlights, key data points, "picks/analysis" category badge |
| Accent (secondary) | `#6C4FE0` | Muted purple — secondary category badge (e.g. recaps), secondary chart bars |
| Text (primary) | `#F5F5F5` | Headlines, key labels |
| Text (secondary) | `#8A93A6` | Supporting text, captions, footnotes |

Rule of thumb: dark background stays constant across every post — that consistency alone is what makes the feed recognizable. The accent color is used sparingly, only to mark the one data point or category that matters most in that post.

## Typography

- **Headlines / big numbers:** Archivo Black (or Barlow Condensed Bold as an alternate) — bold, condensed, "scoreboard" feel
- **Body text / chart labels:** Inter — clean and highly legible, standard for data-dashboard contexts
- **Numbers in stats/tables:** use tabular (fixed-width) numerals so figures align cleanly

## Layout Pattern

Every post follows the same structural skeleton so the brand reads instantly, even before the caption:

1. **Brand row:** "THE GAMEWEEK LAB" wordmark (top-left) + gameweek tag e.g. "GW3 · PRE" (top-right)
2. **Category badge:** solid-fill pill directly under the brand row, stating the post's purpose in a couple words (e.g. "CAPTAINCY PICKS", "POST-GAMEWEEK RECAP"). This is the second thing the eye sees, before the headline — it exists specifically so the post's purpose is clear immediately, not something the reader has to work out.
   - Lime green fill (`#7CFF6B`) = picks/analysis-type content
   - Purple fill (`#6C4FE0`) = recap/review-type content
3. **Headline:** the single takeaway, in plain language (e.g. "Haaland is the play again.")
4. **Supporting line:** 1-2 sentences of context/reasoning
5. **Data visualization:** simple bar comparison or stat grid — no gradients, no 3D, minimal chart junk
6. **Footer:** small model/source note (left) + post number in carousel if applicable (right)

## Notes

- Subtle background grid texture (26px lines, low opacity) reinforces the "data lab" feel without being distracting
- Keep decoration minimal — the accent color and the badge are the only "loud" elements; everything else stays quiet and disciplined
- A working HTML template exists (`gameweek-lab-post-template.html`) demonstrating both a "picks" and a "recap" layout — use it as the starting structure for new posts
