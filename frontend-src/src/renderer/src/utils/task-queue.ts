/* eslint-disable no-promise-executor-return */
/* eslint-disable arrow-parens */
/* eslint-disable no-restricted-syntax */

interface QueuedTask {
  task: () => Promise<void>;
  /** Estimated audio duration in seconds (0 for silent / non-audio tasks). */
  weight: number;
}

export class TaskQueue {
  private queue: QueuedTask[] = [];

  private running = false;

  private taskInterval: number;

  private pendingComplete = false;

  private activeTasks = new Set<Promise<void>>();

  // --- Pre-roll (look-ahead) buffering ------------------------------------
  // Synthesis time per sentence is uneven: an occasional slow sentence mid-reply
  // used to drain the playback queue and leave dead air — the avatar appeared to
  // "stop talking" until the slow sentence finally arrived. To absorb those
  // spikes we hold playback until a few seconds of audio are buffered ahead,
  // then keep playing; during fast stretches the lead grows on its own and a
  // later slow sentence is covered by what's already queued.
  //
  // `buffering` is true while we are accumulating and NOT yet playing. We start
  // playing as soon as EITHER ~prerollSeconds of audio is queued, OR the
  // max-wait timer fires (so short replies / slow synthesis still start
  // promptly), OR the backend signals synthesis is complete (flush the tail).
  // After an underrun mid-reply we re-enter buffering to rebuild the lead.
  private buffering = true;

  private streamComplete = false;

  private prerollSeconds: number;

  private prerollMaxWaitMs: number;

  private bufferTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    taskIntervalMs = 3000,
    prerollSeconds = 5,
    prerollMaxWaitMs = 1500,
  ) {
    this.taskInterval = taskIntervalMs;
    this.prerollSeconds = prerollSeconds;
    this.prerollMaxWaitMs = prerollMaxWaitMs;
  }

  /** Seconds of audio currently queued but not yet played. */
  private bufferedSeconds(): number {
    let total = 0;
    for (const item of this.queue) total += item.weight;
    return total;
  }

  private clearBufferTimer() {
    if (this.bufferTimer !== null) {
      clearTimeout(this.bufferTimer);
      this.bufferTimer = null;
    }
  }

  /**
   * While buffering, arm a fallback timer that starts playback after
   * prerollMaxWaitMs even if we never reach prerollSeconds (short reply or slow
   * synthesis). Idempotent: only one timer runs at a time.
   */
  private ensureBufferTimer() {
    if (this.bufferTimer !== null) return;
    this.bufferTimer = setTimeout(() => {
      this.bufferTimer = null;
      this.buffering = false;
      this.runNextTask();
    }, this.prerollMaxWaitMs);
  }

  /**
   * Tell the queue whether the backend has finished synthesizing this turn.
   * Once complete there is nothing left to wait for, so stop buffering and play
   * whatever is queued straight through.
   */
  setStreamComplete(complete: boolean) {
    this.streamComplete = complete;
    if (complete) {
      this.buffering = false;
      this.clearBufferTimer();
      this.runNextTask();
    }
  }

  addTask(task: () => Promise<void>, estDurationSec = 0) {
    this.queue.push({ task, weight: estDurationSec });
    if (this.buffering && !this.streamComplete) this.ensureBufferTimer();
    this.runNextTask();
  }

  clearQueue() {
    this.queue = [];
    this.activeTasks.clear();
    this.running = false;
    // Reset buffering state for the next speaking turn.
    this.buffering = true;
    this.streamComplete = false;
    this.clearBufferTimer();
  }

  private async runNextTask() {
    if (this.running) return;

    // Pre-roll gate: while still buffering and the stream is mid-flight, wait
    // until ~prerollSeconds is queued. The max-wait timer is the fallback start.
    if (this.buffering && !this.streamComplete) {
      if (this.queue.length > 0 && this.bufferedSeconds() >= this.prerollSeconds) {
        this.buffering = false;
        this.clearBufferTimer();
      } else {
        if (this.queue.length > 0) this.ensureBufferTimer();
        return;
      }
    }

    if (this.queue.length === 0) {
      if (this.activeTasks.size === 0 && this.pendingComplete) {
        this.pendingComplete = false;
        await new Promise(resolve => setTimeout(resolve, this.taskInterval));
      }
      // Mid-reply underrun: more audio is still coming but we ran dry. Re-enter
      // buffering so we rebuild a small look-ahead lead before resuming instead
      // of playing each trickling clip the instant it arrives.
      if (!this.streamComplete) this.buffering = true;
      return;
    }

    this.running = true;
    const item = this.queue.shift();
    if (item) {
      const taskPromise = item.task();
      this.activeTasks.add(taskPromise);

      try {
        await taskPromise;
        await new Promise(resolve => setTimeout(resolve, this.taskInterval));
      } catch (error) {
        console.error('Task Queue Error', error);
      } finally {
        this.activeTasks.delete(taskPromise);
        this.running = false;
        this.runNextTask();
      }
    }
  }

  public hasTask(): boolean {
    return this.queue.length > 0 || this.activeTasks.size > 0 || this.running;
  }

  public waitForCompletion(): Promise<void> {
    this.pendingComplete = true;
    return new Promise((resolve) => {
      const check = () => {
        if (!this.hasTask()) {
          resolve();
        } else {
          setTimeout(check, 100);
        }
      };
      check();
    });
  }
}

// taskIntervalMs=20 (tiny gap between clips), prerollSeconds=3 (build ~3s of
// look-ahead before speaking so a slow sentence mid-reply doesn't cause dead
// air), prerollMaxWaitMs=400 (never delay the first word by more than ~0.4s).
// 2026-07-21: prerollMaxWaitMs 1500->400 + prerollSeconds 5->3 to cut the
// first-word latency: the old 1.5s look-ahead wait stacked on top of the ~1.5s
// TTS synth, pushing time-to-first-audio to ~3s. The pipelined per-sentence
// synth (~1.5s/sentence) stays ahead of playback (~2-4s/sentence), so a short
// look-ahead is enough; underrun re-buffering (runNextTask) still covers a slow
// sentence mid-reply. Raise these back up if dead-air gaps reappear mid-reply.
export const audioTaskQueue = new TaskQueue(20, 3, 400);
