"""Shared solver/method tuple parsing for Estimator and Optimizer."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

MethodTuple = Tuple[str, str, str]


def parse_method(
    method: Union[str, MethodTuple, None],
    *,
    allowed_methods: Sequence[MethodTuple],
    default_methods: Sequence[MethodTuple],
    default_mode: str = "ad",
    default_none_method: Optional[MethodTuple] = None,
    allow_transcription: bool = False,
) -> Tuple[MethodTuple, Optional[str]]:
    """Parse a method string/tuple into ``(library, optimizer, mode)``.

    Returns:
        ``(method_tuple, transcription_or_None)``. Transcription is only set when
        ``allow_transcription`` is True and a 4-tuple was provided.
    """
    transcription = None
    if allow_transcription and isinstance(method, tuple) and len(method) == 4:
        transcription = method[3]
        allowed_transcriptions = ("single_shooting", "collocation")
        assert transcription in allowed_transcriptions, (
            "The 4th (transcription) element of the method tuple must be one "
            f"of {allowed_transcriptions} - \"{transcription}\" was provided."
        )
        method = tuple(method[:3])

    if isinstance(method, str):
        valid_methods = list(
            set([l[0] for l in allowed_methods] + [l[1] for l in allowed_methods])
        )
        assert (
            method in valid_methods
        ), f"If a string is provided, the \"method\" argument must be one of the following: {', '.join(valid_methods)} - \"{method}\" was provided."

        matched = False
        for t in default_methods:
            if t[0] == method:
                method = t
                matched = True
                break

        if not matched:
            candidates = []
            for m in allowed_methods:
                if m[1] == method:
                    candidates.append(m)

            if len(candidates) == 1:
                method = candidates[0]
            elif len(candidates) > 1:
                for c in candidates:
                    if c[2] == default_mode:
                        method = c
                        break

    elif isinstance(method, tuple):
        assert (
            len(method) == 3
        ), f'If a tuple is provided, it must contain three elements, corresponding to the library, method, and mode (e.g. ("scipy", "SLSQP", "ad")) - "{method}" was provided.'
        assert method[0] in [
            l[0] for l in allowed_methods
        ], f"If a tuple is provided, the first element must be one of the following: {', '.join(list(set([l[0] for l in allowed_methods])))} - \"{method}\" was provided."
        assert method[1] in [
            l[1] for l in allowed_methods
        ], f"If a tuple is provided, the second element must be one of the following: {', '.join(list(set([l[1] for l in allowed_methods])))} - \"{method}\" was provided."
        assert method[2] in [
            l[2] for l in allowed_methods
        ], f"If a tuple is provided, the third element must be one of the following: {', '.join(list(set([l[2] for l in allowed_methods])))} - \"{method}\" was provided."

        method_ = None
        for t in allowed_methods:
            if t[0] == method[0] and t[1] == method[1] and t[2] == method[2]:
                method_ = t
                break
        assert (
            method_ is not None
        ), f"The method {method} is not valid. Only the following methods are supported: {', '.join([str(t) for t in allowed_methods])}"
        method = method_
    elif method is None:
        if default_none_method is None:
            raise ValueError('The "method" argument must be a string or a tuple.')
        method = default_none_method
    else:
        raise ValueError(
            f'The "method" argument must be a string or a tuple - "{method}" was provided.'
        )

    return method, transcription
