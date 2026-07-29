import type { KeywordReactionDef, MotionBehaviorConfig } from './types';

/**
 * KeywordReactionController — scans sentence text for configured keywords and
 * returns the matching reaction, honouring a global cooldown so reactions stay
 * occasional. Returns null when nothing matches or we're still in cooldown.
 */
export class KeywordReactionController {
  private lastFireSec = -Infinity;

  constructor(private cfg: MotionBehaviorConfig) {}

  scan(
    text: string,
    nowSec: number,
  ): { reaction: KeywordReactionDef; skipped?: false } | { skipped: true; reason: string } | null {
    if (!text) return null;

    const lower = text.toLowerCase();
    const match = (this.cfg.keywordReactions || []).find((r) =>
      r.keywords.some((k) => lower.includes(k.toLowerCase())),
    );
    if (!match) return null;

    const cooldownSec = this.cfg.keywordReactionCooldownMs / 1000;
    if (nowSec - this.lastFireSec < cooldownSec) {
      return { skipped: true, reason: `cooldown (${match.reaction})` };
    }

    this.lastFireSec = nowSec;
    return { reaction: match };
  }
}
