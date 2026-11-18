// kehai-glgrab: capture the render Chrome's window ON THE GPU and either encode it
// with NVENC straight from the GL texture (H.264 Annex B, one AUD per frame, to
// stdout) or write raw BGRA frames (debugging).
//
// Why not ffmpeg's x11grab (2026-11-05): x11grab asks the X server for the pixels
// (XShmGetImage). On the GPU X server that readback waits until the GPU has
// drained the work every CUDA client has queued — 100-700 ms while Whisper's
// encoder/decoder, the TTS graphs or GestureLSM run — and the X server is
// single-threaded, so Chrome's presents wait behind it and no frame is drawn.
// Measured on a second X server: a test page under a queued CUDA load held
// 60 fps with no capture, 15 fps with x11grab capturing (even with rawvideo, no
// NVENC), 60 fps with this capture.
//
// The window's composite pixmap is bound as a GL texture
// (GLX_EXT_texture_from_pixmap) and copied into our own texture on the GPU; NVENC
// reads that texture (NV_ENC_DEVICE_TYPE_OPENGL). No pixel crosses to the CPU, and
// any wait for the GPU happens in THIS process, never in the X server.
//
// usage: kehai-glgrab WIDTH HEIGHT FPS [MBPS]     (DISPLAY selects the X server)
//   with MBPS: H.264 like ffmpeg -c:v h264_nvenc -preset p1 -tune ull -bf 0 -g FPS
//              -rc cbr -b:v MBPS M -maxrate MBPS M -bufsize max(1,MBPS/15) M -aud 1,
//              SPS/PPS repeated on every IDR, VUI as ffmpeg writes for RGB input.
//   without:   raw BGRA, top row first, one frame per tick.
// Exits non-zero when the window is gone for 5 s (render_stream restarts it).
#define _GNU_SOURCE
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "include/nvEncodeAPI.h"

// libXcomposite has no headers on this box; these are its stable entry points.
extern Bool XCompositeQueryExtension(Display*, int*, int*);
extern void XCompositeRedirectSubwindows(Display*, Window, int);
extern Pixmap XCompositeNameWindowPixmap(Display*, Window);
#define CompositeRedirectAutomatic 0

#define RING 4

static PFNGLXBINDTEXIMAGEEXTPROC pBindTex;
static PFNGLXRELEASETEXIMAGEEXTPROC pReleaseTex;
static PFNGLGENBUFFERSPROC pGenBuffers;
static PFNGLBINDBUFFERPROC pBindBuffer;
static PFNGLBUFFERDATAPROC pBufferData;
static PFNGLMAPBUFFERRANGEPROC pMapBufferRange;
static PFNGLUNMAPBUFFERPROC pUnmapBuffer;
static PFNGLFENCESYNCPROC pFenceSync;
static PFNGLCLIENTWAITSYNCPROC pClientWaitSync;
static PFNGLDELETESYNCPROC pDeleteSync;
static PFNGLGENFRAMEBUFFERSPROC pGenFramebuffers;
static PFNGLBINDFRAMEBUFFERPROC pBindFramebuffer;
static PFNGLFRAMEBUFFERTEXTURE2DPROC pFramebufferTexture2D;

static volatile int g_xerr;
static int on_xerror(Display* d, XErrorEvent* e) {
  (void)d;
  g_xerr = e->error_code ? e->error_code : 1;
  return 0;
}

static volatile sig_atomic_t g_stop, g_snap;
static void on_signal(int s) { (void)s; g_stop = 1; }
static void on_snap(int s) { (void)s; g_snap = 1; }

// SIGUSR1: write the frame being encoded, quarter size, as a binary PPM
// ($GLGRAB_SNAPSHOT, default /tmp/kehai-glgrab-<pid>.ppm) — "is the stream black?"
static void snapshot(int W, int H) {
  uint8_t* px = malloc((size_t)W * H * 4);
  if (!px) return;
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
  char path[256];
  const char* env = getenv("GLGRAB_SNAPSHOT");
  if (env)
    snprintf(path, sizeof path, "%s", env);
  else
    snprintf(path, sizeof path, "/tmp/kehai-glgrab-%d.ppm", getpid());
  FILE* f = fopen(path, "wb");
  if (f) {
    fprintf(f, "P6\n%d %d\n255\n", W / 4, H / 4);
    for (int y = 0; y < H / 4; y++)
      for (int x = 0; x < W / 4; x++) fwrite(px + ((size_t)(y * 4) * W + x * 4) * 4, 1, 3, f);
    fclose(f);
    fprintf(stderr, "[glgrab] snapshot -> %s\n", path);
  }
  free(px);
}

static void* gp(const char* n) {
  void* f = (void*)glXGetProcAddressARB((const GLubyte*)n);
  if (!f) {
    fprintf(stderr, "[glgrab] missing GL entry point %s\n", n);
    exit(3);
  }
  return f;
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static int write_all(int fd, const uint8_t* p, size_t n) {
  while (n) {
    ssize_t k = write(fd, p, n);
    if (k < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    p += k;
    n -= (size_t)k;
  }
  return 0;
}

// ---------------------------------------------------------------- the window

// The render window: the largest viewable child of the root.
static Window find_window(Display* d, Window root, int* w, int* h, Visual** vis, int* depth) {
  Window r, p, *kids = NULL, best = 0;
  unsigned n = 0;
  long best_area = 0;
  if (!XQueryTree(d, root, &r, &p, &kids, &n)) return 0;
  for (unsigned i = 0; i < n; i++) {
    XWindowAttributes a;
    g_xerr = 0;
    if (!XGetWindowAttributes(d, kids[i], &a) || g_xerr) continue;
    if (a.map_state != IsViewable || a.class != InputOutput) continue;
    long area = (long)a.width * a.height;
    if (area > best_area) {
      best_area = area;
      best = kids[i];
      *w = a.width;
      *h = a.height;
      *vis = a.visual;
      *depth = a.depth;
    }
  }
  if (kids) XFree(kids);
  return best_area >= 64 * 64 ? best : 0;
}

typedef struct {
  Window win;
  Pixmap pix;
  GLXPixmap glxpix;
  int w, h, y_inverted;
} Source;

static void source_drop(Display* d, Source* s) {
  if (s->glxpix) glXDestroyPixmap(d, s->glxpix);
  if (s->pix) XFreePixmap(d, s->pix);
  s->glxpix = 0;
  s->pix = 0;
  s->win = 0;
}

static int source_open(Display* d, Window root, Source* s) {
  int w = 0, h = 0, depth = 0;
  Visual* vis = NULL;
  Window win = find_window(d, root, &w, &h, &vis, &depth);
  if (!win) return 0;
  int scr = DefaultScreen(d), nc = 0;
  int attrs[] = {GLX_DRAWABLE_TYPE, GLX_PIXMAP_BIT,
                 depth == 32 ? GLX_BIND_TO_TEXTURE_RGBA_EXT : GLX_BIND_TO_TEXTURE_RGB_EXT, True,
                 GLX_BIND_TO_TEXTURE_TARGETS_EXT, GLX_TEXTURE_2D_BIT_EXT, None};
  GLXFBConfig* cfgs = glXChooseFBConfig(d, scr, attrs, &nc);
  GLXFBConfig cfg = NULL;
  for (int i = 0; cfgs && i < nc && !cfg; i++) {
    XVisualInfo* vi = glXGetVisualFromFBConfig(d, cfgs[i]);
    if (vi && vi->visualid == XVisualIDFromVisual(vis)) cfg = cfgs[i];
    if (vi) XFree(vi);
  }
  for (int i = 0; cfgs && i < nc && !cfg; i++) {  // no exact visual: same depth
    int bs = 0;
    glXGetFBConfigAttrib(d, cfgs[i], GLX_BUFFER_SIZE, &bs);
    if (bs == depth || (depth == 24 && bs == 32)) cfg = cfgs[i];
  }
  if (!cfg) {
    if (cfgs) XFree(cfgs);
    fprintf(stderr, "[glgrab] no texture-from-pixmap config for depth %d\n", depth);
    return 0;
  }
  int yinv = 0;
  glXGetFBConfigAttrib(d, cfg, GLX_Y_INVERTED_EXT, &yinv);
  g_xerr = 0;
  XCompositeRedirectSubwindows(d, root, CompositeRedirectAutomatic);
  XSelectInput(d, win, StructureNotifyMask);
  Pixmap pix = XCompositeNameWindowPixmap(d, win);
  XSync(d, False);
  if (g_xerr || !pix) {
    static double said;
    if (now_s() - said > 1) fprintf(stderr, "[glgrab] naming the window pixmap failed (X error %d)\n", g_xerr), said = now_s();
    XFree(cfgs);
    return 0;
  }
  int pattrs[] = {GLX_TEXTURE_TARGET_EXT, GLX_TEXTURE_2D_EXT, GLX_TEXTURE_FORMAT_EXT,
                  depth == 32 ? GLX_TEXTURE_FORMAT_RGBA_EXT : GLX_TEXTURE_FORMAT_RGB_EXT, None};
  GLXPixmap gpix = glXCreatePixmap(d, cfg, pix, pattrs);
  XSync(d, False);
  XFree(cfgs);
  if (g_xerr || !gpix) {
    // BadAlloc (11) here: another process holds this window's pixmap as a GL
    // texture — one capture per window (a second kehai-glgrab gets this).
    static double said;
    if (now_s() - said > 1) fprintf(stderr, "[glgrab] binding the window pixmap to GL failed (X error %d)\n", g_xerr), said = now_s();
    XFreePixmap(d, pix);
    return 0;
  }
  s->win = win;
  s->pix = pix;
  s->glxpix = gpix;
  s->w = w;
  s->h = h;
  s->y_inverted = yinv;
  fprintf(stderr, "[glgrab] window 0x%lx %dx%d depth %d y_inverted %d\n", win, w, h, depth, yinv);
  return 1;
}

// ---------------------------------------------------------------- NVENC

static NV_ENCODE_API_FUNCTION_LIST nv = {NV_ENCODE_API_FUNCTION_LIST_VER};

typedef struct {
  void* enc;
  NV_ENC_OUTPUT_PTR bitstream;
  NV_ENC_REGISTERED_PTR reg;
  int w, h;
  long n;
} Encoder;

#define NVCHECK(call)                                                          \
  do {                                                                         \
    NVENCSTATUS st_ = (call);                                                  \
    if (st_ != NV_ENC_SUCCESS) {                                               \
      fprintf(stderr, "[glgrab] %s failed: %d\n", #call, (int)st_);            \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static int encoder_open(Encoder* e, GLuint tex, int W, int H, int fps, int mbps) {
  void* lib = dlopen("libnvidia-encode.so.1", RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "[glgrab] cannot load libnvidia-encode.so.1\n");
    return -1;
  }
  NVENCSTATUS(NVENCAPI * create)(NV_ENCODE_API_FUNCTION_LIST*) = dlsym(lib, "NvEncodeAPICreateInstance");
  if (!create) return -1;
  NVCHECK(create(&nv));

  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op = {NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
  op.deviceType = NV_ENC_DEVICE_TYPE_OPENGL;  // device NULL: the current GL context
  op.apiVersion = NVENCAPI_VERSION;
  NVCHECK(nv.nvEncOpenEncodeSessionEx(&op, &e->enc));

  NV_ENC_PRESET_CONFIG pc = {NV_ENC_PRESET_CONFIG_VER, 0, {NV_ENC_CONFIG_VER}};
  NVCHECK(nv.nvEncGetEncodePresetConfigEx(e->enc, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P1_GUID,
                                          NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &pc));
  NV_ENC_CONFIG cfg = pc.presetCfg;
  cfg.version = NV_ENC_CONFIG_VER;
  cfg.profileGUID = NV_ENC_H264_PROFILE_MAIN_GUID;
  cfg.gopLength = fps;
  cfg.frameIntervalP = 1;  // no B-frames
  cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
  cfg.rcParams.averageBitRate = (uint32_t)mbps * 1000000u;
  cfg.rcParams.maxBitRate = (uint32_t)mbps * 1000000u;
  cfg.rcParams.vbvBufferSize = (uint32_t)(mbps / 15 > 1 ? mbps / 15 : 1) * 1000000u;
  cfg.rcParams.vbvInitialDelay = cfg.rcParams.vbvBufferSize;
  NV_ENC_CONFIG_H264* h = &cfg.encodeCodecConfig.h264Config;
  h->idrPeriod = fps;
  h->repeatSPSPPS = 1;  // a viewer can join at any IDR
  h->outputAUD = 1;     // render_stream splits access units on the AUD
  // What ffmpeg's nvenc writes for RGB input: the encoder converts RGB with BT.601,
  // limited range, and says so; primaries/transfer left unspecified.
  NV_ENC_CONFIG_H264_VUI_PARAMETERS* v = &h->h264VUIParameters;
  v->videoSignalTypePresentFlag = 1;
  v->videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
  v->videoFullRangeFlag = 0;
  v->colourDescriptionPresentFlag = 1;
  v->colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT470BG;
  v->colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_UNSPECIFIED;
  v->transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_UNSPECIFIED;

  NV_ENC_INITIALIZE_PARAMS ip = {NV_ENC_INITIALIZE_PARAMS_VER};
  ip.encodeGUID = NV_ENC_CODEC_H264_GUID;
  ip.presetGUID = NV_ENC_PRESET_P1_GUID;
  ip.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
  ip.encodeWidth = ip.darWidth = ip.maxEncodeWidth = W;
  ip.encodeHeight = ip.darHeight = ip.maxEncodeHeight = H;
  ip.frameRateNum = fps;
  ip.frameRateDen = 1;
  ip.enablePTD = 1;
  ip.encodeConfig = &cfg;
  NVCHECK(nv.nvEncInitializeEncoder(e->enc, &ip));

  NV_ENC_CREATE_BITSTREAM_BUFFER bb = {NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
  NVCHECK(nv.nvEncCreateBitstreamBuffer(e->enc, &bb));
  e->bitstream = bb.bitstreamBuffer;

  static NV_ENC_INPUT_RESOURCE_OPENGL_TEX gltex;
  gltex.texture = tex;
  gltex.target = GL_TEXTURE_2D;
  NV_ENC_REGISTER_RESOURCE rr = {NV_ENC_REGISTER_RESOURCE_VER};
  rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX;
  rr.width = W;
  rr.height = H;
  rr.pitch = W * 4;
  rr.resourceToRegister = &gltex;
  rr.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;  // GL_RGBA8: bytes R,G,B,A
  rr.bufferUsage = NV_ENC_INPUT_IMAGE;
  NVCHECK(nv.nvEncRegisterResource(e->enc, &rr));
  e->reg = rr.registeredResource;
  e->w = W;
  e->h = H;
  fprintf(stderr, "[glgrab] NVENC H.264 %dx%d@%d %d Mbps CBR, P1 ultra-low-latency, GOP %d\n", W, H,
          fps, mbps, fps);
  return 0;
}

// Encode the texture as it is now; write the access unit to stdout.
static int encoder_frame(Encoder* e) {
  NV_ENC_MAP_INPUT_RESOURCE mp = {NV_ENC_MAP_INPUT_RESOURCE_VER};
  mp.registeredResource = e->reg;
  NVCHECK(nv.nvEncMapInputResource(e->enc, &mp));
  NV_ENC_PIC_PARAMS pp = {NV_ENC_PIC_PARAMS_VER};
  pp.inputWidth = e->w;
  pp.inputHeight = e->h;
  pp.inputPitch = e->w * 4;
  pp.inputBuffer = mp.mappedResource;
  pp.bufferFmt = mp.mappedBufferFmt;
  pp.outputBitstream = e->bitstream;
  pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  pp.inputTimeStamp = e->n++;
  NVENCSTATUS st = nv.nvEncEncodePicture(e->enc, &pp);
  if (st != NV_ENC_SUCCESS) {
    nv.nvEncUnmapInputResource(e->enc, mp.mappedResource);
    fprintf(stderr, "[glgrab] nvEncEncodePicture failed: %d\n", (int)st);
    return -1;
  }
  NV_ENC_LOCK_BITSTREAM lb = {NV_ENC_LOCK_BITSTREAM_VER};
  lb.outputBitstream = e->bitstream;
  st = nv.nvEncLockBitstream(e->enc, &lb);  // waits for this frame
  int rc = -1;
  if (st == NV_ENC_SUCCESS) {
    rc = write_all(1, lb.bitstreamBufferPtr, lb.bitstreamSizeInBytes) ? -2 : 0;
    nv.nvEncUnlockBitstream(e->enc, e->bitstream);
  } else {
    fprintf(stderr, "[glgrab] nvEncLockBitstream failed: %d\n", (int)st);
  }
  nv.nvEncUnmapInputResource(e->enc, mp.mappedResource);
  return rc;
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s WIDTH HEIGHT FPS [MBPS]\n", argv[0]);
    return 2;
  }
  const int W = atoi(argv[1]), H = atoi(argv[2]), FPS = atoi(argv[3]);
  const int MBPS = argc > 4 ? atoi(argv[4]) : 0;
  const double period = 1.0 / FPS;
  const size_t frame = (size_t)W * H * 4;
  const double report_s = getenv("GLGRAB_REPORT_S") ? atof(getenv("GLGRAB_REPORT_S")) : 60;
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGUSR1, on_snap);
  fcntl(1, F_SETPIPE_SZ, 1 << 20);

  Display* d = XOpenDisplay(NULL);
  if (!d) {
    fprintf(stderr, "[glgrab] cannot open display\n");
    return 2;
  }
  XSetErrorHandler(on_xerror);
  int ev, er;
  if (!XCompositeQueryExtension(d, &ev, &er)) {
    fprintf(stderr, "[glgrab] no Composite extension\n");
    return 2;
  }
  Window root = DefaultRootWindow(d);

  // A context on a tiny pbuffer: we never draw to the screen.
  int cattrs[] = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT,
                  GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
  int nc = 0;
  GLXFBConfig* cc = glXChooseFBConfig(d, DefaultScreen(d), cattrs, &nc);
  if (!cc || !nc) {
    fprintf(stderr, "[glgrab] no pbuffer config\n");
    return 2;
  }
  int pb[] = {GLX_PBUFFER_WIDTH, 16, GLX_PBUFFER_HEIGHT, 16, None};
  GLXPbuffer pbuf = glXCreatePbuffer(d, cc[0], pb);
  GLXContext ctx = glXCreateNewContext(d, cc[0], GLX_RGBA_TYPE, NULL, True);
  if (!ctx || !glXMakeContextCurrent(d, pbuf, pbuf, ctx)) {
    fprintf(stderr, "[glgrab] cannot make a GL context current\n");
    return 2;
  }
  XFree(cc);
  pBindTex = gp("glXBindTexImageEXT");
  pReleaseTex = gp("glXReleaseTexImageEXT");
  pGenBuffers = gp("glGenBuffers");
  pBindBuffer = gp("glBindBuffer");
  pBufferData = gp("glBufferData");
  pMapBufferRange = gp("glMapBufferRange");
  pUnmapBuffer = gp("glUnmapBuffer");
  pFenceSync = gp("glFenceSync");
  pClientWaitSync = gp("glClientWaitSync");
  pDeleteSync = gp("glDeleteSync");
  pGenFramebuffers = gp("glGenFramebuffers");
  pBindFramebuffer = gp("glBindFramebuffer");
  pFramebufferTexture2D = gp("glFramebufferTexture2D");
  fprintf(stderr, "[glgrab] GL %s on %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

  // Our copy of the frame: W x H, row 0 = the top of the window.
  GLuint dst_tex, dst_fbo, src_tex, pbo[RING];
  glGenTextures(1, &dst_tex);
  glBindTexture(GL_TEXTURE_2D, dst_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  pGenFramebuffers(1, &dst_fbo);
  pBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
  pFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst_tex, 0);
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  glGenTextures(1, &src_tex);

  Encoder enc = {0};
  uint8_t* black = NULL;
  if (MBPS > 0) {
    if (encoder_open(&enc, dst_tex, W, H, FPS, MBPS)) return 5;
  } else {
    pGenBuffers(RING, pbo);
    for (int i = 0; i < RING; i++) {
      pBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[i]);
      pBufferData(GL_PIXEL_PACK_BUFFER, frame, NULL, GL_STREAM_READ);
    }
    pBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    black = calloc(1, frame);
  }

  GLsync fence[RING] = {0};
  long seq[RING] = {0};
  long issued = 0;
  int shown = -1;  // raw mode: the PBO holding the frame last written
  Source src = {0};
  double next = now_s(), lost_since = 0, last_report = now_s(), last_try = 0;
  long n_new = 0, n_repeat = 0, n_skip = 0, n_late = 0;
  double wait_max = 0;

  while (!g_stop) {
    // Window: (re)open, and follow resizes / Chrome restarts.
    while (XPending(d)) {
      XEvent e;
      XNextEvent(d, &e);
      if (src.win && (e.type == ConfigureNotify || e.type == UnmapNotify || e.type == DestroyNotify))
        source_drop(d, &src);
    }
    if (!src.win && now_s() - last_try > 0.25) {
      last_try = now_s();
      if (!source_open(d, root, &src)) {
        if (!lost_since) lost_since = now_s();
        if (now_s() - lost_since > 5) {
          fprintf(stderr, "[glgrab] no render window (or it cannot be bound) for 5 s\n");
          _exit(4);
        }
      } else {
        lost_since = 0;
      }
    }

    // Copy the window into our texture (GPU only).
    int copied = 0;
    if (src.win) {
      g_xerr = 0;
      // NVIDIA will not attach a texture-from-pixmap texture to a framebuffer
      // (GL_FRAMEBUFFER_UNSUPPORTED), so draw it: one textured quad, pixel for pixel.
      pBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
      int cw = src.w < W ? src.w : W, ch = src.h < H ? src.h : H;
      glViewport(0, 0, cw, ch);
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, src_tex);
      pBindTex(d, src.glxpix, GLX_FRONT_LEFT_EXT, NULL);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      // Row 0 of dst (y = -1) must be the TOP of the window.
      float s1 = (float)cw / src.w, th = (float)ch / src.h;
      float t0 = src.y_inverted ? 0.f : 1.f, t1 = src.y_inverted ? th : 1.f - th;
      glBegin(GL_QUADS);
      glTexCoord2f(0, t0);
      glVertex2f(-1, -1);
      glTexCoord2f(s1, t0);
      glVertex2f(1, -1);
      glTexCoord2f(s1, t1);
      glVertex2f(1, 1);
      glTexCoord2f(0, t1);
      glVertex2f(-1, 1);
      glEnd();
      if (getenv("GLGRAB_DEBUG") && n_new + n_repeat < 3)
        fprintf(stderr, "[glgrab] debug: glError 0x%x\n", glGetError());
      pReleaseTex(d, src.glxpix, GLX_FRONT_LEFT_EXT);
      glDisable(GL_TEXTURE_2D);
      if (g_xerr)
        source_drop(d, &src);
      else
        copied = 1;
    }

    if (g_snap) {
      g_snap = 0;
      pBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
      snapshot(W, H);
    }

    if (MBPS > 0) {
      // NVENC reads the texture: wait (here, not in the X server) for the copy.
      GLsync f = pFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      double t0 = now_s();
      pClientWaitSync(f, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
      pDeleteSync(f);
      double waited = now_s() - t0;
      if (waited > wait_max) wait_max = waited;
      if (waited > period) n_late++;
      int rc = encoder_frame(&enc);
      if (rc == -2) break;  // reader gone
      if (rc) _exit(5);
      if (copied) n_new++; else n_repeat++;
    } else {
      // Raw: readback into a free PBO; write the newest finished one (else repeat).
      int slot = -1;
      for (int i = 0; i < RING; i++)
        if (!fence[i] && i != shown) {
          slot = i;
          break;
        }
      if (copied && slot >= 0) {
        pBindFramebuffer(GL_READ_FRAMEBUFFER, dst_fbo);
        pBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[slot]);
        glReadPixels(0, 0, W, H, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        pBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        fence[slot] = pFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        seq[slot] = ++issued;
        glFlush();
      } else if (copied) {
        n_skip++;
      }
      int ready = -1;
      for (int i = 0; i < RING; i++)
        if (fence[i] && pClientWaitSync(fence[i], 0, 0) != GL_TIMEOUT_EXPIRED &&
            (ready < 0 || seq[i] > seq[ready]))
          ready = i;
      if (ready >= 0) {
        for (int i = 0; i < RING; i++)
          if (fence[i] && seq[i] < seq[ready]) {
            pDeleteSync(fence[i]);
            fence[i] = 0;
          }
        pDeleteSync(fence[ready]);
        fence[ready] = 0;
        shown = ready;
        n_new++;
      } else {
        n_repeat++;
      }
      int rc;
      if (shown >= 0) {
        pBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[shown]);
        const uint8_t* p = pMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, frame, GL_MAP_READ_BIT);
        rc = p ? write_all(1, p, frame) : write_all(1, black, frame);
        pUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        pBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
      } else {
        rc = write_all(1, black, frame);
      }
      if (rc) break;
    }

    double t = now_s();
    if (t - last_report > report_s) {
      fprintf(stderr,
              "[glgrab] last %.0f s: %ld frames, %ld repeated, %ld readbacks skipped, %ld late "
              "(GPU wait max %.0f ms)\n",
              report_s, n_new, n_repeat, n_skip, n_late, wait_max * 1000);
      n_new = n_repeat = n_skip = n_late = 0;
      wait_max = 0;
      last_report = t;
    }
    next += period;
    if (t > next + 2 * period) next = t;  // fell behind: do not burst to catch up
    double wait = next - now_s();
    if (wait > 0) {
      struct timespec ts = {(time_t)wait, (long)((wait - (time_t)wait) * 1e9)};
      nanosleep(&ts, NULL);
    }
  }
  // Close the encoder while the GL context is still current, then leave without
  // the driver's exit handlers: a plain exit() with the session open hung in a
  // futex inside the driver (2026-11-05); the kernel frees the rest.
  if (enc.enc) {
    if (enc.reg) nv.nvEncUnregisterResource(enc.enc, enc.reg);
    if (enc.bitstream) nv.nvEncDestroyBitstreamBuffer(enc.enc, enc.bitstream);
    nv.nvEncDestroyEncoder(enc.enc);
  }
  fprintf(stderr, "[glgrab] stopped\n");
  _exit(0);
}
