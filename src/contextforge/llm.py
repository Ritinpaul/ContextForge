from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    model: str
    task: str
    max_new_tokens: int = 256
    temperature: float = 0.7


class LLM:
    """Minimal text generation wrapper.

    Default implementation uses `transformers.pipeline` so the project is runnable
    on Windows without external API keys.

    The generated quality depends heavily on the chosen model.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._pipe = None

    @property
    def config(self) -> LLMConfig:
        return self._config

    def _ensure(self) -> None:
        if self._pipe is not None:
            return

        from transformers import pipeline

        self._pipe = pipeline(
            task=self._config.task,
            model=self._config.model,
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        self._ensure()

        temp = self._config.temperature if temperature is None else float(temperature)
        mnt = self._config.max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        task = self._config.task
        if task == "text2text-generation":
            out = self._pipe(
                prompt,
                max_new_tokens=mnt,
                temperature=temp,
            )
            # transformers returns: [{"generated_text": "..."}]
            return (out[0] or {}).get("generated_text", "").strip()

        if task == "text-generation":
            out = self._pipe(
                prompt,
                max_new_tokens=mnt,
                temperature=temp,
                do_sample=temp > 0,
                return_full_text=False,
            )
            # transformers returns: [{"generated_text": "..."}]
            return (out[0] or {}).get("generated_text", "").strip()

        out = self._pipe(prompt)
        if isinstance(out, list) and out:
            maybe = out[0]
            if isinstance(maybe, dict) and "generated_text" in maybe:
                return str(maybe["generated_text"]).strip()
        return str(out).strip()
