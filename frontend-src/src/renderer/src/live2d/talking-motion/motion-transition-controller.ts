/**
 * MotionTransitionController — lightweight priority + cooldown gate for
 * *triggering existing motions/expressions* (keyword reactions), so reactions
 * don't stomp each other or fire back-to-back. The continuous sway/beat layer
 * itself is always smooth (parameter-additive), so this only governs the
 * heavier, event-driven reactions.
 *
 * Priority: higher wins. A higher-or-equal priority request may interrupt a
 * still-active lower one; a lower one is skipped while a higher one is active.
 */
export class MotionTransitionController {
  private activeUntilSec = -Infinity;
  private activePriority = 0;

  /**
   * Ask to start a reaction of `priority` lasting ~`holdSec`. Returns true if
   * allowed (and records it), false if a stronger reaction is still playing.
   */
  request(priority: number, holdSec: number, nowSec: number): boolean {
    const stillActive = nowSec < this.activeUntilSec;
    if (stillActive && priority < this.activePriority) {
      return false; // a stronger reaction owns the body right now
    }
    this.activeUntilSec = nowSec + holdSec;
    this.activePriority = priority;
    return true;
  }

  /** Whether a reaction is currently considered active (for logging). */
  isActive(nowSec: number): boolean {
    return nowSec < this.activeUntilSec;
  }
}
