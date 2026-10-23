/**
 * Plays the server's `backchannel` clip — a one-word acknowledgment sent the
 * moment a turn ends (src/kehai/utils/backchannel.py). Its own AudioContext so
 * it never touches the streaming reply player's clock; in render-host mode the
 * render-source tap picks it up like any other audio leaving for the speakers.
 */
let ctx: AudioContext | null = null;

export async function playBackchannel(b64: string): Promise<void> {
  try {
    ctx ??= new AudioContext();
    void ctx.resume();
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const buffer = await ctx.decodeAudioData(bytes.buffer);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    src.start();
    console.info(`[backchannel] playing ${buffer.duration.toFixed(2)}s`);
  } catch (e) {
    console.warn("[backchannel] could not play:", e);
  }
}
