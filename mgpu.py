"""GPU render backend for Milky (moderngl).

Moves the two CPU-bound passes -- the feedback warp and the bloom -- onto the
GPU as fragment-shader passes, and folds mirror/kaleido into the present shader
(a UV manipulation, far cheaper than the CPU rotozoom copies).

The scene itself is still drawn on the CPU (cheap vector work) to a pygame
Surface, uploaded once per frame and additively composited onto the GPU
accumulation buffer. So all ten scene_* methods keep working unchanged.

Pipeline per frame (accum is a ping-pong pair of RGBA textures):
    1. warp    : sample accum(prev) with rotate+zoom+offset, * decay -> accum(cur)
    2. scene   : upload CPU scene layer -> additive blend onto accum(cur)
    3. bloom   : bright-pass + separable gaussian on a quarter-res target
    4. present : accum(cur) (+ bloom), mirror/kaleido UV fold -> screen
    5. hud     : upload CPU hud layer -> alpha blend over the screen
"""
import moderngl
import numpy as np

_VERT = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() { uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# feedback: sample the previous accumulation frame, warped and decayed
_WARP = """
#version 330
uniform sampler2D tex;
uniform float angle, zoom, decay;
uniform vec2 offset;
in vec2 uv; out vec4 frag;
void main() {
    vec2 c = uv - 0.5;
    float s = sin(angle), co = cos(angle);
    c = mat2(co, -s, s, co) * c / zoom;   // rotate + zoom about center
    vec2 su = c + 0.5 + offset;
    if (su.x < 0.0 || su.x > 1.0 || su.y < 0.0 || su.y > 1.0)
        frag = vec4(0.0);                 // no edge smear outside the frame
    else
        frag = texture(tex, su) * decay;
}
"""

# plain textured quad; flip_y compensates pygame's top-left origin on upload
_BLIT = """
#version 330
uniform sampler2D tex;
uniform bool flip_y;
in vec2 uv; out vec4 frag;
void main() {
    vec2 u = uv; if (flip_y) u.y = 1.0 - u.y;
    frag = texture(tex, u);
}
"""

# bloom bright-pass: keep only the bright pixels
_BRIGHT = """
#version 330
uniform sampler2D tex;
uniform float thresh;
in vec2 uv; out vec4 frag;
void main() {
    vec3 c = texture(tex, uv).rgb;
    float l = max(max(c.r, c.g), c.b);
    frag = vec4(c * smoothstep(thresh, thresh + 0.15, l), 1.0);
}
"""

# separable gaussian blur (run once per axis via `texel`)
_BLUR = """
#version 330
uniform sampler2D tex;
uniform vec2 texel;
in vec2 uv; out vec4 frag;
void main() {
    vec3 c = texture(tex, uv).rgb * 0.227027;
    c += (texture(tex, uv + texel).rgb        + texture(tex, uv - texel).rgb)        * 0.194594;
    c += (texture(tex, uv + 2.0*texel).rgb    + texture(tex, uv - 2.0*texel).rgb)    * 0.121622;
    c += (texture(tex, uv + 3.0*texel).rgb    + texture(tex, uv - 3.0*texel).rgb)    * 0.054054;
    c += (texture(tex, uv + 4.0*texel).rgb    + texture(tex, uv - 4.0*texel).rgb)    * 0.016216;
    frag = vec4(c, 1.0);
}
"""

# present: mirror/kaleido UV fold + additive bloom, straight to the screen
_PRESENT = """
#version 330
uniform sampler2D scene;
uniform sampler2D bloom;
uniform int mirror;        // 0 none, 1 lr, 2 quad, 3 kaleido
uniform float bloom_amt;
in vec2 uv; out vec4 frag;
vec2 fold(vec2 u) {
    if (mirror == 1) {
        u.x = 0.5 - abs(u.x - 0.5);
    } else if (mirror == 2) {
        u.x = 0.5 - abs(u.x - 0.5);
        u.y = 0.5 - abs(u.y - 0.5);
    } else if (mirror == 3) {
        vec2 p = u - 0.5;
        float r = length(p);
        float a = atan(p.y, p.x);
        float seg = 3.14159265 / 3.0;
        a = abs(mod(a, seg) - seg * 0.5);
        u = vec2(cos(a), sin(a)) * r + 0.5;
    }
    return u;
}
void main() {
    vec2 u = fold(uv);
    vec3 c = texture(scene, u).rgb + texture(bloom, u).rgb * bloom_amt;
    frag = vec4(c, 1.0);
}
"""

MIRROR_ID = {"none": 0, "lr": 1, "quad": 2, "kaleido": 3}


class GPURenderer:
    def __init__(self, rw, rh):
        self.rw, self.rh = rw, rh
        self.ctx = moderngl.create_context()
        ctx = self.ctx

        # fullscreen quad (triangle strip)
        vbo = ctx.buffer(np.array(
            [-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())

        def prog(fs):
            p = ctx.program(vertex_shader=_VERT, fragment_shader=fs)
            vao = ctx.vertex_array(p, [(vbo, "2f", "in_pos")])
            return p, vao

        self.warp, self.warp_vao = prog(_WARP)
        self.blit, self.blit_vao = prog(_BLIT)
        self.bright, self.bright_vao = prog(_BRIGHT)
        self.blur, self.blur_vao = prog(_BLUR)
        self.present, self.present_vao = prog(_PRESENT)

        def tex(w, h, comp=4):
            t = ctx.texture((w, h), comp)
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            return t

        # ping-pong accumulation
        self.accum = [tex(rw, rh) for _ in range(2)]
        self.acc_fbo = [ctx.framebuffer([t]) for t in self.accum]
        self.cur = 0
        for f in self.acc_fbo:                 # start black
            f.use(); ctx.clear(0, 0, 0, 1)

        self.scene_tex = tex(rw, rh, 3)        # CPU scene layer (RGB)
        self.hud_tex = tex(1, 1, 4)            # CPU hud layer (RGBA); resized on demand
        self.dino_tex = tex(1, 1, 4)           # dino overlay; resized on demand
        self.post_tex = tex(1, 1, 4)           # scene post layer (party, etc); resized on demand

        # quarter-res bloom scratch
        bw, bh = max(1, rw // 4), max(1, rh // 4)
        self.bw, self.bh = bw, bh
        self.bloom_a, self.bloom_b = tex(bw, bh), tex(bw, bh)
        self.bloom_fa = ctx.framebuffer([self.bloom_a])
        self.bloom_fb = ctx.framebuffer([self.bloom_b])
        self.black = tex(1, 1)                  # bound when bloom is off
        self.black.write(b"\x00\x00\x00\xff")

    def _quad(self, prog, vao):
        vao.render(moderngl.TRIANGLE_STRIP)

    def render(self, *, angle, zoom, offset, decay, scene_bytes,
               mirror_id, bloom_amt, hud_bytes, dino_bytes,
               scene_post_bytes, win_size):
        ctx = self.ctx
        prev = self.accum[self.cur]
        self.cur ^= 1
        dst_fbo = self.acc_fbo[self.cur]
        dst = self.accum[self.cur]

        # 1. warp previous accumulation into the current buffer
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.scissor = None
        ctx.depth_mask = False
        dst_fbo.use()
        ctx.viewport = (0, 0, self.rw, self.rh)
        prev.use(0)
        self.warp["tex"] = 0
        self.warp["angle"] = angle
        self.warp["zoom"] = zoom
        self.warp["decay"] = decay
        self.warp["offset"] = offset
        self._quad(self.warp, self.warp_vao)

        # 2. upload scene layer, additively composite on top
        self.scene_tex.write(scene_bytes)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self.scene_tex.use(0)
        self.blit["tex"] = 0
        self.blit["flip_y"] = True
        self._quad(self.blit, self.blit_vao)
        ctx.disable(moderngl.BLEND)

        # 3. bloom (bright-pass + separable gaussian) at quarter res
        bloom_src = self.black
        if bloom_amt > 0.01:
            self.bloom_fa.use()
            ctx.viewport = (0, 0, self.bw, self.bh)
            dst.use(0)
            self.bright["tex"] = 0
            self.bright["thresh"] = 0.5
            self._quad(self.bright, self.bright_vao)
            # horizontal then vertical
            for tex_in, fbo_out, dx, dy in (
                    (self.bloom_a, self.bloom_fb, 1.0 / self.bw, 0.0),
                    (self.bloom_b, self.bloom_fa, 0.0, 1.0 / self.bh)):
                fbo_out.use()
                tex_in.use(0)
                self.blur["tex"] = 0
                self.blur["texel"] = (dx, dy)
                self._quad(self.blur, self.blur_vao)
            bloom_src = self.bloom_a

        # 4. present to the screen with mirror/kaleido fold + bloom
        ctx.screen.use()
        ctx.viewport = (0, 0, *win_size)
        ctx.scissor = None
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        dst.use(0)
        bloom_src.use(1)
        self.present["scene"] = 0
        self.present["bloom"] = 1
        self.present["mirror"] = mirror_id
        self.present["bloom_amt"] = bloom_amt
        self._quad(self.present, self.present_vao)

        # 5. hud overlay (alpha blended, unaffected by the fold)
        if hud_bytes is not None:
            hw, hh = self.hud_tex.size
            expected = (win_size[0], win_size[1])
            if (hw, hh) != expected:
                self.hud_tex = ctx.texture(expected, 4, data=hud_bytes)
                self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.hud_tex.repeat_x = self.hud_tex.repeat_y = False
            else:
                self.hud_tex.write(hud_bytes)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.hud_tex.use(0)
            self.blit["tex"] = 0
            self.blit["flip_y"] = True
            self._quad(self.blit, self.blit_vao)
            ctx.disable(moderngl.BLEND)

        # 7. scene post overlay (party animals, etc); alpha blended
        if scene_post_bytes is not None:
            pw, ph = self.post_tex.size
            expected = (win_size[0], win_size[1])
            if (pw, ph) != expected:
                self.post_tex = ctx.texture(expected, 4, data=scene_post_bytes)
                self.post_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.post_tex.repeat_x = self.post_tex.repeat_y = False
            else:
                self.post_tex.write(scene_post_bytes)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.post_tex.use(0)
            self.blit["tex"] = 0
            self.blit["flip_y"] = True
            self._quad(self.blit, self.blit_vao)
            ctx.disable(moderngl.BLEND)

        # 6. dino overlay (also alpha blended, after everything)
        if dino_bytes is not None:
            dw, dh = self.dino_tex.size
            if (dw, dh) != (win_size[0], win_size[1]):
                self.dino_tex = ctx.texture(win_size, 4, data=dino_bytes)
                self.dino_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.dino_tex.repeat_x = self.dino_tex.repeat_y = False
            else:
                self.dino_tex.write(dino_bytes)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.dino_tex.use(0)
            self.blit["tex"] = 0
            self.blit["flip_y"] = True
            self._quad(self.blit, self.blit_vao)
            ctx.disable(moderngl.BLEND)
