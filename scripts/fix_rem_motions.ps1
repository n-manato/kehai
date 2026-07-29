$path = 'Live2D Characters/ac_base_rem_mizugi01/ac_base_rem_mizugi01.model3.json'
$lines = Get-Content -LiteralPath $path

$start = -1
$end = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
  if ($start -lt 0 -and $lines[$i] -match '^\s*"Motions"\s*:\s*\{') {
    $start = $i
    continue
  }
  if ($start -ge 0 -and $lines[$i] -match '^\s*"UserData"\s*:') {
    $end = $i
    break
  }
}

if ($start -lt 0 -or $end -lt 0) {
  throw "Motions/UserData markers not found"
}

$replacement = @(
  '    "Motions": {',
  '      "Idle": [',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_normal_w.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_normal_w.motion3.json"',
  '        }',
  '      ],',
  '      "Talk": [',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_talk_normal.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_talk_small.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_talk_large.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_talk_off.motion3.json"',
  '        }',
  '      ],',
  '      "Joy": [',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_egao.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_egao02.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_hohoemu.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_egao.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_egao02.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_hohoemu.motion3.json"',
  '        }',
  '      ],',
  '      "Anger": [',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_ikaru.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_punpun.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_shinken.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_suneru.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_ikaru.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_ikaru02.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_ikaru03.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_punpun.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_shinken_w.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_suneru.motion3.json"',
  '        }',
  '      ],',
  '      "Sadness": [',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_nayamu.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_tameiki.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_uru.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_kanashimu_w.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_nayamu.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_uru.motion3.json"',
  '        }',
  '      ],',
  '      "Surprise": [',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_bikkuri.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_odoroku.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_bikkuri.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_odoroku.motion3.json"',
  '        }',
  '      ],',
  '      "Thinking": [',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_kangaeru.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_act_unazuku.motion3.json"',
  '        },',
  '        {',
  '          "File": "motions/AC_rem_mizugi01_face_kangaeru.motion3.json"',
  '        }',
  '      ]',
  '    },'
)

$newLines = @()
$newLines += $lines[0..($start - 1)]
$newLines += $replacement
$newLines += $lines[$end..($lines.Count - 1)]

Set-Content -LiteralPath $path -Value $newLines
