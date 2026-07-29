"""One-off generator: wire up the other Rem Live2D models the same way the
mizugi (swimsuit) Rem is set up.

For each target model it:
  1. Rewrites <model>.model3.json so the loose motion files are organised into
     named motion groups (Idle, Joy_big, Anger_rage, ...) with fade times — only
     including groups whose source files actually exist for that model.
     `face_*` motions are also registered as Face_* groups so they can be used
     as motion-based expressions.
  2. Appends an entry to model_dict.json with emotionMap / motionGroupMap copied
     from the mizugi template but pruned to the groups this model really has.
     expressionGroupMap is left empty because these models have no .exp3 files
     (their act_ motions already animate the face).

Idempotent: re-running regenerates cleanly. The mizugi model is never touched.
"""

import json
import os

# Paths below are relative to the repo root (model_dict.json, "Live2D Characters").
# This script lives in scripts/, so anchor the working directory to the repo root
# regardless of where it is invoked from.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

BASE = "Live2D Characters"
TEMPLATE = "ac_base_rem_mizugi01"
TARGETS = [
    "ac_base_rem01",
    "ac_base_rem_childhood01",
    "ac_base_rem_nemaki01",
    "ac_base_rem_ogre01",
    "ac_base_rem_ogre_normal01",
    "ac_base_rem_pokerface01",
    "ac_base_rem_valentine01",
    "ac_base_rem_xmas01",
]

IDLE_GROUPS = {"Idle", "Idle_nod", "Idle_sigh"}
FADE_IDLE = 0.8
FADE_ACTION = 0.5

# group name -> ordered list of candidate motion-file suffixes (after the model
# prefix, without ".motion3.json"). Only suffixes that exist are included; a
# group with no existing candidate is dropped entirely.
GROUP_RULES = {
    "Idle": ["act_normal_w", "act_normal02_w", "act_normal03_w"],
    "Idle_nod": ["act_unazuku", "act_unazuku02"],
    "Idle_sigh": ["act_tameiki", "act_tameiki02"],
    "Joy_big": [
        "act_egao", "act_egao_w", "act_egao02", "act_egao02_w",
        "act_egao03", "act_egao03_w",
        "add_select_C_egao", "add_select_L_egao", "add_select_R_egao",
    ],
    "Joy_smile": ["act_hohoemu", "act_hohoemu_w"],
    # gekido (rage) stands in for ikaru where a model has it (pokerface/ogre).
    "Anger_rage": ["act_ikaru", "act_ikaru_w", "act_gekido", "act_gekido_w"],
    "Anger_pout": ["act_punpun", "act_punpun_w"],
    "Anger_serious": [
        "act_shinken", "act_shinken_w", "act_shinken02", "act_shinken02_w",
    ],
    "Anger_sulk": ["act_suneru", "act_suneru_w"],
    "Sadness_worry": [
        "act_nayamu", "act_nayamu_w",
        "add_select_C_nayamu", "add_select_L_nayamu", "add_select_R_nayamu",
    ],
    "Sadness_sad": ["act_nayamu", "act_nayamu_w"],
    "Sadness_tearful": [
        "act_uru", "act_uru_w",
        "add_select_C_uru", "add_select_L_uru", "add_select_R_uru",
    ],
    "Surprise_shock": [
        "act_bikkuri", "act_bikkuri_w",
        "add_select_C_bikkuri", "add_select_L_bikkuri", "add_select_R_bikkuri",
    ],
    "Surprise_amazed": ["act_odoroku", "act_odoroku_w"],
    "Thinking": ["act_kangaeru", "act_kangaeru_w"],
    "Look": ["add_select_C", "add_select_L", "add_select_R"],
    "Option": ["option01_on", "option01_off"],
}

# Face_* groups built from face_* motions (used as motion-based expressions).
FACE_GROUP_RULES = {
    "Face_egao": ["face_egao", "face_egao_w", "face_egao02", "face_egao02_w"],
    "Face_hohoemu": ["face_hohoemu", "face_hohoemu_w"],
    "Face_ikaru": ["face_ikaru", "face_ikaru_w", "face_ikaru02", "face_ikaru03",
                   "face_gekido", "face_gekido_w"],
    "Face_punpun": ["face_punpun", "face_punpun_w"],
    "Face_nayamu": ["face_nayamu", "face_nayamu_w", "face_kanashimu_w"],
    "Face_uru": ["face_uru", "face_uru_w"],
    "Face_bikkuri": ["face_bikkuri", "face_bikkuri_w"],
    "Face_odoroku": ["face_odoroku", "face_odoroku_w"],
    "Face_kangaeru": ["face_kangaeru", "face_kangaeru_w"],
    "Face_normal": ["face_normal_w", "face_normal02_w", "face_normal03_w"],
    "Face_cheek": ["face_cheek_on"],
}

# emotion keyword -> preferred motion group, in priority order. The first group
# that exists for the model wins. Mirrors the mizugi mapping.
EMOTION_TO_GROUP = {
    "neutral": ["Idle"],
    "idle_normal": ["Idle"],
    "idle_nod": ["Idle_nod", "Idle"],
    "idle_sigh": ["Idle_sigh", "Idle"],
    "joy": ["Joy_big", "Joy_smile"],
    "joy_big": ["Joy_big", "Joy_smile"],
    "joy_smile": ["Joy_smile", "Joy_big"],
    "joy_grin": ["Joy_big", "Joy_smile"],
    "anger": ["Anger_rage", "Anger_pout"],
    "anger_rage": ["Anger_rage", "Anger_pout"],
    "anger_pout": ["Anger_pout", "Anger_rage"],
    "anger_serious": ["Anger_serious", "Anger_rage"],
    "anger_sulk": ["Anger_sulk", "Anger_pout"],
    "anger_glare": ["Anger_rage", "Anger_pout"],
    "disgust": ["Anger_pout", "Anger_rage"],
    "fear": ["Surprise_shock"],
    "sadness": ["Sadness_worry", "Sadness_sad"],
    "sadness_worry": ["Sadness_worry", "Sadness_sad"],
    "sadness_sad": ["Sadness_sad", "Sadness_worry"],
    "sadness_tearful": ["Sadness_tearful", "Sadness_worry"],
    "surprise": ["Surprise_shock"],
    "surprise_shock": ["Surprise_shock"],
    "surprise_amazed": ["Surprise_amazed", "Surprise_shock"],
    "thinking": ["Thinking"],
    "shy": ["Joy_smile", "Joy_big"],
    "look": ["Look", "Idle"],
}


def list_suffixes(model_name):
    """Return set of motion suffixes available for a model."""
    mdir = os.path.join(BASE, model_name, "motions")
    prefix = "AC_rem" + model_name[len("ac_base_rem"):] + "_"
    # e.g. ac_base_rem01 -> AC_rem01_ ; ac_base_rem_xmas01 -> AC_rem_xmas01_
    sufs = {}
    for f in os.listdir(mdir):
        if not f.endswith(".motion3.json"):
            continue
        if not f.startswith(prefix):
            # fall back: strip a generic "AC_rem..._" prefix
            base = f[: -len(".motion3.json")]
            suf = base.split("_", 2)[-1]
        else:
            suf = f[len(prefix):-len(".motion3.json")]
        sufs[suf] = "motions/" + f
    return sufs


def build_motions(sufs):
    """Build the Motions dict (groups -> entries) for a model."""
    motions = {}
    for group, candidates in {**GROUP_RULES, **FACE_GROUP_RULES}.items():
        entries = []
        fade = FADE_IDLE if group in IDLE_GROUPS else FADE_ACTION
        for cand in candidates:
            if cand in sufs:
                entries.append(
                    {
                        "File": sufs[cand],
                        "FadeInTime": fade,
                        "FadeOutTime": fade,
                    }
                )
        if entries:
            motions[group] = entries
    return motions


def main():
    log = []
    # load template model_dict entry for kScale etc.
    model_dict = json.load(open("model_dict.json", encoding="utf-8"))
    tmpl = next(m for m in model_dict if m["name"] == TEMPLATE)

    for model_name in TARGETS:
        m3_path = os.path.join(BASE, model_name, model_name + ".model3.json")
        m3 = json.load(open(m3_path, encoding="utf-8"))
        sufs = list_suffixes(model_name)
        motions = build_motions(sufs)
        m3["FileReferences"]["Motions"] = motions
        # these models have no exp3 expressions
        m3["FileReferences"]["Expressions"] = []
        json.dump(
            m3, open(m3_path, "w", encoding="utf-8"),
            ensure_ascii=False, indent=2,
        )

        available_groups = set(motions.keys())

        # motionGroupMap / emotionMap pruned to what exists
        motion_group_map = {"default": "Idle"}
        emotion_map = {}
        idx = 0
        for emo, prefs in EMOTION_TO_GROUP.items():
            chosen = next((g for g in prefs if g in available_groups), None)
            if chosen is None:
                continue
            motion_group_map[emo] = chosen
            emotion_map[emo] = idx
            idx += 1

        entry = {
            "name": model_name,
            "description": "",
            "url": "/Live2D%20Characters/{0}/{0}.model3.json".format(
                model_name
            ),
            "kScale": tmpl["kScale"],
            "initialXshift": tmpl["initialXshift"],
            "initialYshift": tmpl["initialYshift"],
            "kXOffset": tmpl["kXOffset"],
            "idleMotionGroupName": "Idle",
            "emotionMap": emotion_map,
            "motionGroupMap": motion_group_map,
            "expressionGroupMap": {},
            "tapMotions": {},
            "defaultEmotion": "neutral",
        }

        # replace if already present, else append
        model_dict = [m for m in model_dict if m["name"] != model_name]
        model_dict.append(entry)

        log.append(
            "%-26s groups=%d emotions=%d"
            % (model_name, len(motions), len(emotion_map))
        )

    json.dump(
        model_dict, open("model_dict.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=4,
    )

    open("_build_log.txt", "w", encoding="utf-8").write(
        "\n".join(log)
        + "\n\nTotal models in dict: %d" % len(model_dict)
    )
    print("DONE")


if __name__ == "__main__":
    main()
