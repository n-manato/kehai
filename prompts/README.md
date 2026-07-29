# Prompts

This directory contains utility prompts used in the Open-LLM-VTuber project. These are general-purpose prompts that are not specific to any character's persona.

## Examples of Utility Prompts

*   **Live2D Expressions:** Prompts that inform the LLM about available Live2D expressions.
*   **Tool Usage:** Prompts that guide the LLM on how to use available tools.
*   ... and many more.

## Character Persona Prompts

Character persona prompts (the prompts that define the personality of your AI
characters) can be provided in two ways:

1.  **Inline** — as `persona_prompt:` in your `conf.yaml`, or in a YAML file
    inside the `characters/` directory.
2.  **External file (recommended for long personas)** — store the persona text
    in a `.txt` file under `prompts/personas/` and reference it with
    `persona_prompt_file:` instead of `persona_prompt:`. The file content is
    loaded into `persona_prompt` at config-load time.

    ```yaml
    character_config:
      # ...
      persona_prompt_file: "prompts/personas/rem.txt"
    ```

    This keeps the (potentially long) persona under version control and out of
    `conf.yaml` — useful because `conf.yaml` is git-ignored and machine-specific.
    If both are set, the inline `persona_prompt` wins.